"""Classical retrieval: local keypoints + ratio test + RANSAC homography (inlier count = score).

detector="sift" (float descriptors, L2) or "orb" (binary descriptors, Hamming).
Brute force over the whole database is slow for thousands of cards; that is exactly why the
hybrid method restricts this step to a shortlist (see score_candidates / hybrid.py).
"""
import cv2
import numpy as np
from tqdm import tqdm

from .base import Retriever, load_canonical


class KeypointRetriever(Retriever):
    def __init__(self, detector="sift", ratio=0.75, ransac_thresh=5.0, nfeatures=800):
        self.detector_name = detector
        self.name = detector
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        if detector == "sift":
            self.det = cv2.SIFT_create(nfeatures=nfeatures)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif detector == "orb":
            self.det = cv2.ORB_create(nfeatures=nfeatures)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError(detector)
        self.ids, self.kps, self.descs = [], [], []

    def _describe(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = self.det.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
        return pts, desc

    def build_index(self, card_paths, card_ids):
        self.ids, self.kps, self.descs = [], [], []
        for path, cid in tqdm(list(zip(card_paths, card_ids)), desc=f"index {self.name}"):
            pts, desc = self._describe(self.prepare(load_canonical(path)))
            self.ids.append(cid)
            self.kps.append(pts)
            self.descs.append(desc)
        self._pos = {cid: i for i, cid in enumerate(self.ids)}

    def _inliers(self, q_pts, q_desc, ref_i):
        r_desc = self.descs[ref_i]
        if q_desc is None or r_desc is None or len(q_desc) < 2 or len(r_desc) < 2:
            return 0
        knn = self.matcher.knnMatch(q_desc, r_desc, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < self.ratio * m[1].distance]
        if len(good) < 4:
            return 0
        src = self.kps[ref_i][[m.trainIdx for m in good]].reshape(-1, 1, 2)
        dst = q_pts[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
        return int(mask.sum()) if mask is not None else 0

    def score_candidates(self, img, candidate_ids):
        """Inlier counts for a subset of reference ids (used by the hybrid method)."""
        q_pts, q_desc = self._describe(self.prepare(img))
        return {cid: self._inliers(q_pts, q_desc, self._pos[cid]) for cid in candidate_ids}

    def query(self, img, topk=None):
        scores = self.score_candidates(img, self.ids)
        ranked = sorted(scores.items(), key=lambda t: -t[1])
        return ranked[:topk] if topk else ranked

    def index_size_bytes(self):
        return sum(k.nbytes for k in self.kps) + sum(d.nbytes for d in self.descs if d is not None)
