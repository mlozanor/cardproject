"""On-disk index cache shared by the app and the experiments.

Building the SIFT index for 785 cards takes about a minute, so the index (keypoints, descriptors,
or the embedding matrix) is pickled to <data>/cache/<signature>_<method>.pkl and reloaded next time.
The signature is an md5 of the card ids + image paths, so a different card list gets its own file.
A hybrid caches its two parts separately (hybrid-clip reuses the clip and sift files).
NOTE: if you change a method's settings (e.g. nfeatures) or replace card images, delete the cache folder.
"""
import hashlib
import pickle

from .. import config

_STATE_KEYS = {"ids", "kps", "descs", "_pos", "matrix"}     # the attributes that make up an index


def signature(paths, ids):
    return hashlib.md5("|".join(list(ids) + list(paths)).encode()).hexdigest()[:10]


def build_or_load(method, manifest=None, paths=None, ids=None):
    """Give `method` its index (from the cache if possible) and return it. Pass a manifest or paths+ids."""
    if manifest is not None:
        paths, ids = manifest["abs_path"].tolist(), manifest["id"].tolist()
    if hasattr(method, "embedder"):                              # hybrid: cache each part
        build_or_load(method.embedder, paths=paths, ids=ids)
        build_or_load(method.geometric, paths=paths, ids=ids)
        return method
    base_name = method.name.replace("+rot", "")                # "+rot" uses the same index
    path = config.DATA_DIR / "cache" / f"{signature(paths, ids)}_{base_name}.pkl"
    if path.exists():
        vars(method).update(pickle.loads(path.read_bytes()))
    else:
        method.build_index(paths, ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({k: v for k, v in vars(method).items() if k in _STATE_KEYS}))
    return method
