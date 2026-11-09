"""Global-embedding retrieval: one L2-normalised vector per card, cosine similarity search.

Subclasses only implement embed(list_of_bgr_images) -> (n, d) float32 array.
TinyPixelRetriever is a dependency-free sanity baseline (tiny grayscale thumbnail).
"""
import cv2
import numpy as np
from tqdm import tqdm

from .base import Retriever, load_canonical


class EmbeddingRetriever(Retriever):
    batch_size = 32
    # Rectification cannot tell up from down. With both_orientations=True ("+rot" in the method name)
    # the query is also embedded rotated by 180 degrees and each card keeps its better similarity.
    both_orientations = False

    def embed(self, imgs):
        raise NotImplementedError

    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=np.float32)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    def build_index(self, card_paths, card_ids):
        self.ids = list(card_ids)
        chunks = []
        for i in tqdm(range(0, len(card_paths), self.batch_size), desc=f"index {self.name}"):
            batch = [self.prepare(load_canonical(p)) for p in card_paths[i:i + self.batch_size]]
            chunks.append(self._normalize(self.embed(batch)))
        self.matrix = np.vstack(chunks)

    def query(self, img, topk=None):
        imgs = [img, cv2.rotate(img, cv2.ROTATE_180)] if self.both_orientations else [img]
        imgs = [self.prepare(im) for im in imgs]           # rotate first, then crop (art box is not centred)
        sims = (self.matrix @ self._normalize(self.embed(imgs)).T).max(axis=1)
        order = np.argsort(-sims)
        ranked = [(self.ids[i], float(sims[i])) for i in order]
        return ranked[:topk] if topk else ranked

    def index_size_bytes(self):
        return int(self.matrix.nbytes)


class TinyPixelRetriever(EmbeddingRetriever):
    name = "pixel"

    def embed(self, imgs):
        out = []
        for im in imgs:
            g = cv2.cvtColor(cv2.resize(im, (32, 44), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            v = g.astype(np.float32).ravel()
            out.append(v - v.mean())
        return np.array(out)
