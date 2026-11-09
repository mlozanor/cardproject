"""Method factory. Heavy imports (torch) happen only when that method is requested.

Names: sift, orb, pixel, resnet, clip, hybrid-<embedding> (e.g. hybrid-clip).
Suffixes, in this order:  <name>[@art][+rot]
  @art  compare only the artwork window (config.ART_BOX) of reference and query: language independent.
  +rot  (embedding / hybrid only) also try the query rotated by 180 degrees (SIFT/ORB are rotation
        invariant already, so they do not need it).
Examples: clip@art, hybrid-clip@art+rot, sift@art.
"""


def get_method(name, k=20):
    if name.endswith("@art"):
        m = get_method(name[:-4], k=k)
        parts = [m.embedder, m.geometric] if hasattr(m, "embedder") else [m]
        for p in parts:
            p.region = "art"
            p.name += "@art"                                # separate index file (different crop)
        if hasattr(m, "embedder"):
            m.name = f"hybrid_{m.embedder.name}_k{m.k}"
        return m
    if name.endswith("+rot"):
        m = get_method(name[:-4], k=k)
        emb = m.embedder if hasattr(m, "embedder") else m
        if not hasattr(emb, "both_orientations"):
            raise ValueError(f"+rot only applies to embedding methods, not {name}")
        emb.both_orientations = True
        emb.name += "+rot"                                  # cache.py strips "+rot": same index file
        if hasattr(m, "embedder"):
            m.name = f"hybrid_{emb.name}_k{m.k}"
        return m
    if name in ("sift", "orb"):
        from .sift import KeypointRetriever
        return KeypointRetriever(detector=name)
    if name == "pixel":
        from .embedding import TinyPixelRetriever
        return TinyPixelRetriever()
    if name == "resnet":
        from .cnn import ResNetRetriever
        return ResNetRetriever()
    if name == "clip":
        from .clip_method import CLIPRetriever
        return CLIPRetriever()
    if name.startswith("hybrid-"):                      # e.g. hybrid-clip, hybrid-resnet, hybrid-pixel
        from .hybrid import HybridRetriever
        from .sift import KeypointRetriever
        return HybridRetriever(get_method(name.split("-", 1)[1]), KeypointRetriever("sift"), k=k)
    raise ValueError(f"unknown method: {name}")
