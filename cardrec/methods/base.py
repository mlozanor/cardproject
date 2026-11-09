"""Common interface so every method can be run and compared the same way."""
from pathlib import Path

import cv2

from .. import config


def load_canonical(path):
    """Load a reference image resized to the canonical card size."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (config.CANON_W, config.CANON_H), interpolation=cv2.INTER_AREA)


class Retriever:
    """Subclasses implement build_index() and query().

    query() takes a rectified (canonical-size) BGR image and returns the FULL ranking as a
    list of (card_id, score), best first. Evaluation needs the full ranking for mAP.

    region = "full" (whole card) or "art" (only config.ART_BOX, "@art" in the method name). Subclasses
    call prepare() on every reference AND query image, so both are cropped the same way. "art"
    ignores all text, so a Spanish or Japanese copy looks like its English reference.
    """
    name = "base"
    region = "full"

    def prepare(self, img):
        if self.region == "art":
            y0, y1, x0, x1 = config.ART_BOX
            return img[y0:y1, x0:x1]
        return img

    def build_index(self, card_paths, card_ids):
        raise NotImplementedError

    def query(self, img_bgr, topk=None):
        raise NotImplementedError

    def index_size_bytes(self):
        return 0
