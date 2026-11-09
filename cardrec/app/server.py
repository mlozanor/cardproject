"""Local web app (runs on your PC, opens in the browser).

    python -m cardrec.app.server              # real database (needs data/manifest.csv)
    python -m cardrec.app.server --demo       # fake cards, works with no data at all
    python -m cardrec.app.server --method orb --port 5000
"""
import argparse
import json
import sys
import threading
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from .. import config
from ..condition import combine
from . import prices
from .db import Database
from .engine import METHODS, Engine


def create_app(demo=False, default_method="sift", warm=True):
    static = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static), static_url_path="/static")
    engine = Engine(demo=demo, default_method=default_method)
    db = Database(config.DATA_DIR / ("collection_demo.db" if demo else "collection.db"))
    uploads = config.DATA_DIR / ("uploads_demo" if demo else "uploads")
    uploads.mkdir(parents=True, exist_ok=True)
    if warm:
        threading.Thread(target=engine.warmup, daemon=True).start()
    else:
        engine.ready = True

    def err(msg, code=400):
        return jsonify({"error": msg}), code

    @app.get("/")
    def index():
        return send_from_directory(static, "index.html")

    @app.get("/api/info")
    def info():
        return jsonify({"demo": demo, "default_method": default_method, "methods": METHODS,
                        "n_reference_cards": len(engine.ids), "ready": engine.ready})

    @app.post("/api/analyze")
    def analyze():
        f = request.files.get("image")
        if f is None:
            return err("No image uploaded")
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return err("Could not read that file as an image")
        method = request.form.get("method") or default_method
        if method not in METHODS:
            return err(f"Unknown method {method}")
        try:
            result = engine.analyze(img, method, detect_language=request.form.get("detect_language", "1") == "1")
        except ImportError as e:
            return err(f"Method '{method}' needs extra packages ({e}). The server runs with {sys.executable}; "
                       f"start it with the project venv (%USERPROFILE%\\.venvs\\cardproject\\Scripts\\python.exe) "
                       f"or pick SIFT/ORB.")
        uid = uuid.uuid4().hex[:12]
        cv2.imwrite(str(uploads / f"{uid}.png"), img)
        cv2.imwrite(str(uploads / f"{uid}_rect.png"), result.pop("rect"))
        result.update(upload_id=uid, upload_url=f"/uploads/{uid}.png", rect_url=f"/uploads/{uid}_rect.png")
        for c in result["candidates"]:
            c["image_url"] = f"/ref/{c['id']}"
        return jsonify(result)

    def photo_for(side):
        """A side's photo: a newly uploaded file '<side>' or an earlier upload '<side>_upload_id'."""
        f = request.files.get(side)
        if f:
            img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None, None
            uid = uuid.uuid4().hex[:12]
            cv2.imwrite(str(uploads / f"{uid}.png"), img)
            return img, uid
        uid = str(request.form.get(f"{side}_upload_id", ""))
        if uid.isalnum() and (uploads / f"{uid}.png").exists():
            return cv2.imread(str(uploads / f"{uid}.png")), uid
        return None, None

    @app.post("/api/condition")
    def condition():
        """Visual condition ESTIMATE from a front and/or back photo of the chosen card (multipart form:
        card_id, front / back files or front_upload_id / back_upload_id)."""
        card_id = request.form.get("card_id")
        if engine.meta(card_id) is None:
            return err("Unknown card")
        results, photos, problems = {}, {}, []
        for side in ("front", "back"):
            img, uid = photo_for(side)
            if img is None:
                continue
            result, card = engine.condition(img, card_id, side)
            if result is None:
                problems.append(f"{side}: the card outline was not found - retake it on a plain, contrasting background")
                continue
            cv2.imwrite(str(uploads / f"{uid}_cond_{side}.jpg"), card)      # straightened, nothing drawn on it
            results[side] = result
            photos[side] = f"/uploads/{uid}_cond_{side}.jpg"
        report = combine(results.get("front"), results.get("back"))
        if report is None:
            return err("; ".join(problems) or "Please add a front and/or a back photo")
        report.update(photos=photos, problems=problems)
        return jsonify(report)

    @app.get("/uploads/<name>")
    def upload_file(name):
        return send_from_directory(uploads, name)

    @app.get("/ref/<card_id>")
    def ref_image(card_id):
        p = engine.path(card_id)
        if p is None:
            abort(404)
        return send_file(p)

    @app.get("/api/price/<card_id>")
    def price(card_id):
        if demo:
            return jsonify({"available": False, "price": None, "error": "Demo mode has no price data"})
        meta = engine.meta(card_id)
        if meta is None:
            return err("Unknown card id", 404)
        return jsonify(prices.get_price(card_id, db, meta.get("tcgdex_id"), refresh=request.args.get("refresh") == "1",
                                        lang=meta.get("language")))

    @app.get("/api/collection")
    def collection():
        items = db.list()
        for it in items:
            it["tags"] = engine.tags(it["card_id"])           # from the database metadata
            it["condition"] = json.loads(it["condition"]) if it.get("condition") else None
            it["image_url"] = f"/ref/{it['card_id']}"
            it["photo_url"] = f"/uploads/{it['photo']}" if it.get("photo") else None
        return jsonify({"items": items, "summary": db.summary()})

    @app.post("/api/save")
    def save():
        d = request.get_json(force=True)
        meta = engine.meta(d.get("card_id"))
        if meta is None:
            return err("Unknown card id")
        uid = d.get("upload_id")
        photo = f"{uid}.png" if uid and (uploads / f"{uid}.png").exists() else None
        price_val, source = d.get("price"), d.get("price_source")
        if price_val is None and not demo:
            p = prices.get_price(meta["id"], db, meta.get("tcgdex_id"), lang=meta.get("language"))
            price_val, source = p.get("price"), p.get("source")
        item_id = db.add(meta, photo, d.get("method"), d.get("score"),
                         quantity=max(int(d.get("quantity", 1)), 1), price=price_val, price_source=source,
                         language=str(d.get("language") or "")[:10])
        if d.get("condition"):                                # keep the last condition estimate with the card
            db.update(item_id, condition=json.dumps(d["condition"])[:20000])
        return jsonify({"id": item_id})

    @app.patch("/api/card/<int:item_id>")
    def patch(item_id):
        if db.get(item_id) is None:
            return err("Not found", 404)
        d = request.get_json(force=True)
        fields = {}
        if "price" in d:
            fields["price"] = None if d["price"] in (None, "") else float(d["price"])
            fields["price_source"] = d.get("price_source", "manual")
        if "quantity" in d:
            fields["quantity"] = max(int(d["quantity"]), 1)
        if "notes" in d:
            fields["notes"] = str(d["notes"])[:500]
        if "language" in d:
            fields["language"] = str(d["language"] or "")[:10]
        db.update(item_id, **fields)
        return jsonify({"ok": True})

    @app.delete("/api/card/<int:item_id>")
    def delete(item_id):
        db.delete(item_id)
        return jsonify({"ok": True})

    return app


VENV_PYTHON = Path.home() / ".venvs" / "cardproject" / "Scripts" / "python.exe"


def relaunch_in_venv():
    """A plain `python` in a new terminal is often the system Python, which has no torch/transformers, so
    every CLIP/ResNet method would fail. If that happens and the project venv exists, run the server with
    the venv's Python instead (same arguments) and exit with its exit code."""
    try:
        import torch, transformers  # noqa: F401,E401
        return
    except ImportError:
        pass
    if not VENV_PYTHON.exists() or Path(sys.executable).resolve() == VENV_PYTHON.resolve():
        return                                          # nothing better available: CLIP/ResNet show an error
    import subprocess
    print(f"torch/transformers are missing in {sys.executable};\nrestarting with the project venv: {VENV_PYTHON}")
    sys.exit(subprocess.call([str(VENV_PYTHON), "-m", "cardrec.app.server", *sys.argv[1:]],
                             cwd=str(Path(__file__).resolve().parents[2])))


if __name__ == "__main__":
    relaunch_in_venv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--method", default="sift", choices=METHODS)
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()
    app = create_app(demo=a.demo, default_method=a.method)
    url = f"http://127.0.0.1:{a.port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"CardDex running at {url}  (Ctrl+C to stop)")
    app.run(port=a.port, debug=False, threaded=True)
