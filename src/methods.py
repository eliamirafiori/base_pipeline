from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple
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


class QuaternionFourierFusionMethod(KFSMethod):
    """
    Key Frame Extraction Based on Quaternion Fourier Transform with Multiple
    Features Fusion (Zhang et al., Expert Systems with Applications, 2023).

    Fuses 1 dynamic motion channel (temporal luminance difference) and 3 static
    perceptual color/intensity channels into a quaternion field, extracts the
    fused QFT phase spectrum, reconstructs a Gaussian-smoothed saliency feature
    map, and triggers keyframes using a motion-adaptive threshold.
    """

    def __init__(
        self,
        threshold: float = 0.36,
        grid_size: Tuple[int, int] = (96, 64),
        gaussian_sigma: float = 1.5,
        blur_thresh: float = 15.0,
        adaptive: bool = True,
        adapt_weight: float = 0.35,
        window_size: int = 30,
    ):
        super().__init__("Quaternion FFT Fusion", threshold)
        self.grid_size = grid_size
        self.gaussian_sigma = gaussian_sigma
        self.blur_thresh = blur_thresh
        self.adaptive = adaptive
        self.adapt_weight = adapt_weight

        # Precompute Gaussian kernel size once
        k = int(2 * round(3 * self.gaussian_sigma) + 1)
        self._gauss_ksize = (k, k)

        # Temporal state (reset automatically at frame 0 of each new video)
        self._prev_lum: Optional[np.ndarray] = None
        self._recent_scores: Deque[float] = deque(maxlen=window_size)

    def is_Blur_or_Invalid(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.blur_thresh

    def set_reference(
        self, frame: np.ndarray, feat_curr: Optional[Dict[str, Any]] = None
    ) -> None:
        # When feat_curr is None (frame 0 of a new video), reset temporal history
        if feat_curr is None:
            self._prev_lum = None
            self._recent_scores.clear()
        super().set_reference(frame, feat_curr)

    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        small_bgr = cv2.resize(frame, self.grid_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # 1. Static perceptual color-opponent & luminance channels (f1, f2, f3)
        rg = r - g
        by = b - 0.5 * (r + g)
        lum = (r + g + b) / 3.0

        # 2. Dynamic temporal motion channel (f0): frame-to-frame luminance diff
        if self._prev_lum is None:
            motion = np.zeros_like(lum)
        else:
            motion = np.abs(lum - self._prev_lum)
        self._prev_lum = lum.copy()
        local_motion = float(np.mean(motion))

        # 3. Symplectic quaternion decomposition: q = (f0 + f1*i) + (f2 + f3*i)*j
        c1 = motion + 1j * rg
        c2 = by + 1j * lum

        # 4. 2D Discrete Quaternion Fourier Transform (via two complex 2D FFTs)
        C1 = np.fft.fft2(c1)
        C2 = np.fft.fft2(c2)

        # 5. Fused QFT Phase Spectrum (normalize by quaternion modulus)
        q_modulus = np.sqrt(np.abs(C1) ** 2 + np.abs(C2) ** 2) + 1e-8
        P1 = C1 / q_modulus
        P2 = C2 / q_modulus

        # 6. Inverse QFT -> quaternion energy -> Gaussian filtering -> [0, 1] map
        q_inv1 = np.fft.ifft2(P1)
        q_inv2 = np.fft.ifft2(P2)
        fused_energy = (np.abs(q_inv1) ** 2 + np.abs(q_inv2) ** 2).astype(np.float32)

        fused_smooth = cv2.GaussianBlur(
            fused_energy, self._gauss_ksize, self.gaussian_sigma
        )
        f_min, f_max = float(fused_smooth.min()), float(fused_smooth.max())
        if f_max - f_min > 1e-8:
            feature_map = (fused_smooth - f_min) / (f_max - f_min)
        else:
            feature_map = np.zeros_like(fused_smooth)

        return {"feature_map": feature_map, "local_motion": local_motion}

    def compare_features(
        self, feat_ref: Dict[str, Any], feat_curr: Dict[str, Any]
    ) -> float:
        map_ref = feat_ref["feature_map"]
        map_curr = feat_curr["feature_map"]

        # Spatial L1 saliency difference + cosine structural distance
        l1_diff = float(np.mean(np.abs(map_ref - map_curr)))
        v_ref = map_ref.ravel()
        v_curr = map_curr.ravel()
        cos_sim = float(
            np.dot(v_ref, v_curr)
            / (np.linalg.norm(v_ref) * np.linalg.norm(v_curr) + 1e-8)
        )
        raw_score = 0.5 * l1_diff + 0.5 * (1.0 - cos_sim)

        if not self.adaptive:
            return raw_score

        # Adaptive local-motion normalization: scales score relative to recent
        # scene activity so slow scans still trigger while rapid jitter is smoothed
        self._recent_scores.append(raw_score)
        if len(self._recent_scores) < 5:
            return raw_score

        local_mean = float(np.mean(self._recent_scores))
        local_std = float(np.std(self._recent_scores))
        dynamic_baseline = local_mean + self.adapt_weight * local_std

        # Rescale around self.threshold so KFSMethod.evaluate_frame() stays plug-and-play
        if dynamic_baseline > 1e-6:
            return raw_score * (self.threshold / dynamic_baseline)
        return raw_score
