"""Load the reference-card manifest (one row per unique card)."""
import pandas as pd
from . import config


def load_manifest(path=config.MANIFEST) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["abs_path"] = df["image_path"].apply(lambda p: str(config.DATA_DIR / p))   # paths are relative to DATA_DIR
    return df


def load_app_manifest() -> pd.DataFrame:
    """The app's reference database: the English manifest plus every extra manifest that exists
    (config.APP_EXTRA_MANIFESTS, e.g. Japanese sets). The experiments use load_manifest() only."""
    parts = [load_manifest()] + [load_manifest(p) for p in config.APP_EXTRA_MANIFESTS if p.exists()]
    return pd.concat(parts, ignore_index=True).fillna("")
