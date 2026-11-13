from typing import Any, List, Tuple
import cv2
import numpy as np
from base import KFSMethod


class FourierMethod(KFSMethod):
    def __init__(self, threshold: float = 0.40, blur_thresh: float = 15.0):
        super().__init__("Fourier (Freq Domain)", threshold)
        self.blur_thresh = blur_thresh

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (256, 256))
        fshift = np.fft.fftshift(np.fft.fft2(resized))
        mag = np.log1p(np.abs(fshift))
        return mag / (np.max(mag) + 1e-8)

    def compare_features(self, feat_ref: np.ndarray, feat_curr: np.ndarray) -> float:
        return float(np.mean(np.abs(feat_ref - feat_curr)) * 10.0)

    def is_Blur_or_Invalid(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.blur_thresh


class PixelIntensityMethod(KFSMethod):
    def __init__(self, threshold: float = 25.0):
        super().__init__("Pixel/Intensity (Smoothed SAD)", threshold)

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(cv2.resize(gray, (160, 120)), (5, 5), 0)

    def compare_features(self, feat_ref: np.ndarray, feat_curr: np.ndarray) -> float:
        return float(np.mean(cv2.absdiff(feat_ref, feat_curr)))


class ColorHistogramMethod(KFSMethod):
    def __init__(self, threshold: float = 0.35, grid_size: Tuple[int, int] = (2, 2)):
        super().__init__("Spatial HSV Histogram", threshold)
        self.grid_size = grid_size

    def extract_features(self, frame: np.ndarray) -> List[np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w, _ = hsv.shape
        gh, gw = h // self.grid_size[0], w // self.grid_size[1]
        hists = []
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                patch = hsv[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
                hist = cv2.calcHist([patch], [0, 1], None, [30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hists.append(hist)
        return hists

    def compare_features(
        self, feat_ref: List[np.ndarray], feat_curr: List[np.ndarray]
    ) -> float:
        return float(
            np.mean(
                [
                    cv2.compareHist(h_ref, h_curr, cv2.HISTCMP_BHATTACHARYYA)
                    for h_ref, h_curr in zip(feat_ref, feat_curr)
                ]
            )
        )


class LocalDescriptorMethod(KFSMethod):
    def __init__(self, threshold: float = 0.70, n_features: int = 500):
        super().__init__("Local Descriptors (ORB)", threshold)
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(
        self, frame: np.ndarray
    ) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.orb.detectAndCompute(gray, None)

    def compare_features(
        self,
        feat_ref: Tuple[List[cv2.KeyPoint], np.ndarray],
        feat_curr: Tuple[List[cv2.KeyPoint], np.ndarray],
    ) -> float:
        kp1, des1 = feat_ref
        kp2, des2 = feat_curr
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 1.0
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = [
            m
            for pair in matches
            if len(pair) == 2
            for m, n in [pair]
            if m.distance < 0.75 * n.distance
        ]
        return float(np.clip(1.0 - (len(good) / max(len(kp1), 1)), 0.0, 1.0))


class LucasKanadeOpticalFlowMethod(KFSMethod):
    def __init__(
        self,
        threshold: float = 45.0,
        max_corners: int = 250,
        min_survival_ratio: float = 0.45,
    ):
        super().__init__("Optical Flow (Sparse LK)", threshold)
        self.min_survival_ratio = min_survival_ratio
        self.feature_params = dict(
            maxCorners=max_corners, qualityLevel=0.01, minDistance=10, blockSize=7
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        self.prev_gray: np.ndarray | None = None
        self.anchor_pts: np.ndarray | None = None
        self.curr_pts: np.ndarray | None = None
        self.initial_pt_count: int = 1

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def set_reference(self, frame: np.ndarray, feat_curr: np.ndarray = None) -> None:
        gray = feat_curr if feat_curr is not None else self.extract_features(frame)
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        self.prev_gray = gray
        self.anchor_pts = pts
        self.curr_pts = pts.copy() if pts is not None else None
        self.initial_pt_count = len(pts) if pts is not None else 1
        self._ref_cache = gray

    def compare_features(self, feat_ref: Any, gray_curr: np.ndarray) -> float:
        if self.prev_gray is None or self.curr_pts is None or len(self.curr_pts) < 8:
            return float(self.threshold)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_curr, self.curr_pts, None, **self.lk_params
        )
        self.prev_gray = gray_curr
        if next_pts is None or status is None:
            return float(self.threshold)

        valid = status.flatten() == 1
        if np.sum(valid) < 8:
            return float(self.threshold)

        self.curr_pts = next_pts[valid].reshape(-1, 1, 2)
        self.anchor_pts = self.anchor_pts[valid].reshape(-1, 1, 2)

        if (
            len(self.curr_pts) / max(self.initial_pt_count, 1)
        ) < self.min_survival_ratio:
            return float(self.threshold)

        displacements = np.linalg.norm(self.curr_pts - self.anchor_pts, axis=2)
        return float(np.mean(displacements))


class FarnebackOpticalFlowMethod(KFSMethod):
    def __init__(
        self, threshold: float = 25.0, grid_size: Tuple[int, int] = (160, 120)
    ):
        super().__init__("Optical Flow (Dense Farneback)", threshold)
        self.grid_size = grid_size
        self.prev_small_gray: np.ndarray | None = None
        self.accum_flow: np.ndarray | None = None

    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.grid_size)

    def set_reference(self, frame: np.ndarray, feat_curr: np.ndarray = None) -> None:
        small_gray = (
            feat_curr if feat_curr is not None else self.extract_features(frame)
        )
        self.prev_small_gray = small_gray
        self.accum_flow = np.zeros(
            (self.grid_size[1], self.grid_size[0], 2), dtype=np.float32
        )
        self._ref_cache = small_gray

    def compare_features(self, feat_ref: Any, small_gray_curr: np.ndarray) -> float:
        if self.prev_small_gray is None or self.accum_flow is None:
            self.set_reference(small_gray_curr, feat_curr=small_gray_curr)
            return 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_small_gray,
            small_gray_curr,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        self.prev_small_gray = small_gray_curr
        self.accum_flow += flow
        return float(np.mean(np.linalg.norm(self.accum_flow, axis=2)))
