"""Hybrid: embedding shortlist (top-k) re-ranked by SIFT+RANSAC inliers (ties -> embedding score)."""
from .base import Retriever


class HybridRetriever(Retriever):
    def __init__(self, embedder, geometric, k=20):
        self.embedder, self.geometric, self.k = embedder, geometric, k
        self.name = f"hybrid_{embedder.name}_k{k}"

    def build_index(self, card_paths, card_ids):
        self.embedder.build_index(card_paths, card_ids)
        self.geometric.build_index(card_paths, card_ids)

    def query(self, img, topk=None):
        emb_rank = self.embedder.query(img)
        shortlist = emb_rank[: self.k]
        inliers = self.geometric.score_candidates(img, [cid for cid, _ in shortlist])
        emb_score = dict(shortlist)
        reranked = sorted(shortlist, key=lambda t: (-inliers[t[0]], -emb_score[t[0]]))
        out = [(cid, float(inliers[cid])) for cid, _ in reranked] + emb_rank[self.k:]
        return out[:topk] if topk else out

    def index_size_bytes(self):
        return self.embedder.index_size_bytes() + self.geometric.index_size_bytes()
