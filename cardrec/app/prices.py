"""Look up a card's market prices (on demand, cached 24 h).

Sources, checked against the live APIs on 2026-10-21:
  1. TCGdex  /v2/en/cards/<tcgdex_id>  ->  pricing.tcgplayer.<variant>.marketPrice (USD) and
     pricing.cardmarket.avg / avg30 (EUR). Has prices for all four of our sets.
  2. Fallback: Pokemon TCG API  /v2/cards/<id>  ->  tcgplayer.prices.<variant>.market (USD),
     cardmarket.prices.averageSellPrice (EUR). Has NO prices for Ascended Heroes and Pitch Black.
"Average price" shown in the app = mean of the TCGplayer market prices over the card's variants
(normal / holofoil / reverse ...), falling back to the Cardmarket average in EUR.
You can always overwrite it by hand in the card view. Cite the source in the report.
"""
import os

from ..database import TCGDEX, get_json

API = "https://api.pokemontcg.io/v2/cards/"


def _result(variants, eur_avg, eur_avg30, source_name):
    out = {"available": False, "price": None, "currency": "USD", "source": None,
           "variants": variants, "cardmarket_avg30_eur": eur_avg30}
    if variants:
        out.update(available=True, source=f"TCGplayer market (mean of variants, via {source_name})",
                   price=round(sum(v["market"] for v in variants) / len(variants), 2))
    elif eur_avg is not None:
        out.update(available=True, price=eur_avg, currency="EUR",
                   source=f"Cardmarket average (via {source_name})")
    return out


def summarize_tcgdex(card):
    pricing = card.get("pricing") or {}
    tp = pricing.get("tcgplayer") or {}
    variants = [{"name": k, "market": v.get("marketPrice")} for k, v in tp.items()
                if isinstance(v, dict) and v.get("marketPrice") is not None]   # skips "unit", "updated"
    cm = pricing.get("cardmarket") or {}
    return _result(variants, cm.get("avg"), cm.get("avg30"), "TCGdex")


def summarize(card):
    """Pokemon TCG API card object."""
    tp = (card.get("tcgplayer") or {}).get("prices") or {}
    variants = [{"name": k, "market": v.get("market")} for k, v in tp.items() if v.get("market") is not None]
    cm = (card.get("cardmarket") or {}).get("prices") or {}
    return _result(variants, cm.get("averageSellPrice"), cm.get("avg30"), "pokemontcg.io")


def get_price(card_id, database, tcgdex_id=None, refresh=False, lang="en"):
    """lang = printed language of the reference card. Japanese cards (database_ja.py) are looked up in
    TCGdex's Japanese catalogue, which has Cardmarket prices (EUR) but no TCGplayer prices; the
    Pokemon TCG API has English cards only, so it is not tried for them."""
    if not refresh:
        cached = database.cache_get(card_id)
        if cached is not None:
            return cached
    result, errors = None, []
    if tcgdex_id:
        base = TCGDEX.rsplit("/", 1)[0] + f"/{lang or 'en'}"
        try:
            result = summarize_tcgdex(get_json(f"{base}/cards/{tcgdex_id}", retries=3))
        except Exception as e:                          # offline, unknown id, ...
            errors.append(f"TCGdex: {e}")
    if (result is None or not result["available"]) and (lang or "en") == "en":
        headers = {"X-Api-Key": os.environ["POKEMONTCG_API_KEY"]} if os.environ.get("POKEMONTCG_API_KEY") else {}
        try:
            # this API often answers 500/502 for a moment: up to 4 tries (waits 3+6+9 s at most)
            fallback = summarize(get_json(API + card_id, headers=headers, retries=4)["data"])
            if result is None or fallback["available"]:
                result = fallback
        except Exception as e:
            errors.append(f"pokemontcg.io: {e}")
    if result is None:
        return {"available": False, "price": None, "error": "; ".join(errors)[:200]}
    database.cache_put(card_id, result)
    return result
