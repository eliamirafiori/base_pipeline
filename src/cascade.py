from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from base import KFSMethod


class EgoMotionAndCascadeKFS(KFSMethod):
    """
    4-Stage Hierarchical KFS Cascade + Ego-Motion Classifier:
      Stage 1: Pixel SAD (0.15 ms) -> Rejects stationary frames immediately.
      Stage 2: Dense Optical Flow + Radial Divergence -> Logs Robot Ego-Motion & integrates displacement.
      Stage 3: Laplacian High-Frequency Gate -> Rejects motion-blurred transition frames.
      Stage 4: ORB Keypoint Overlap -> Confirms 3D geometric change before promoting Keyframe.
    """

    def __init__(
        self,
        flow_trigger_thresh: float = 22.0,
        static_sad_thresh: float = 1.8,
        blur_thresh: float = 15.0,
        orb_thresh: float = 0.65,
        grid_size: Tuple[int, int] = (160, 120),
    ):
        super().__init__("Stacked Cascade + EgoMotion", threshold=flow_trigger_thresh)
        self.grid_w, self.grid_h = grid_size
        self.static_sad_thresh = static_sad_thresh
        self.blur_thresh = blur_thresh
        self.orb_thresh = orb_thresh

        # Precompute outward unit radial vectors from image center (Focus of Expansion)
        cx, cy = self.grid_w / 2.0, self.grid_h / 2.0
        x, y = np.meshgrid(np.arange(self.grid_w), np.arange(self.grid_h))
        rx, ry = x - cx, y - cy
        norms = np.hypot(rx, ry) + 1e-6
        self.unit_rx = (rx / norms).astype(np.float32)
        self.unit_ry = (ry / norms).astype(np.float32)

        # Stage 4 ORB verifier
        self.orb = cv2.ORB_create(nfeatures=400)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Internal state
        self.prev_small: np.ndarray | None = None
        self.accum_flow: np.ndarray | None = None
        self.ref_orb_des: np.ndarray | None = None
        self.ref_orb_kp_count: int = 1
        self.motion_history: List[Tuple[int, str]] = []

    def _classify_ego_motion(self, flow: np.ndarray, sad_step: float) -> str:
        if sad_step < self.static_sad_thresh:
            return "STATIONARY"

        u, v = flow[..., 0], flow[..., 1]
        # Invert optical flow sign to get physical camera motion direction
        cam_vx = -float(np.median(u))  # + = Robot turning/moving RIGHT, - = LEFT
        cam_vy = -float(np.median(v))  # + = Robot pitching DOWN, - = UP

        # Subtract global translation/rotation median to isolate radial expansion/contraction
        u_centered = u - np.median(u)
        v_centered = v - np.median(v)
        radial_div = float(
            np.mean(u_centered * self.unit_rx + v_centered * self.unit_ry)
        )

        lat_thresh = 0.40
        fwd_thresh = 0.20
        actions = []

        if radial_div > fwd_thresh:
            actions.append("MOVING FORWARD")
        elif radial_div < -fwd_thresh:
            actions.append("MOVING BACKWARD")

        if abs(cam_vx) > lat_thresh and abs(cam_vx) > abs(cam_vy) * 1.15:
            actions.append(
                "TURNING/MOVING RIGHT" if cam_vx > 0 else "TURNING/MOVING LEFT"
            )
        elif abs(cam_vy) > lat_thresh:
            actions.append("PITCHING DOWN" if cam_vy > 0 else "PITCHING UP")

        return " + ".join(actions) if actions else "MICRO-MOTION"

    def extract_features(self, frame: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (self.grid_w, self.grid_h))
        return {"gray": gray, "small": small, "orb_des": None, "orb_kp_count": 0}

    def set_reference(
        self, frame: np.ndarray, feat_curr: Dict[str, Any] = None
    ) -> None:
        feat = feat_curr if feat_curr is not None else self.extract_features(frame)
        self.prev_small = feat["small"]
        self.accum_flow = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)

        # Reuse ORB descriptors if Stage 4 already computed them on this frame
        if feat.get("orb_des") is not None:
            self.ref_orb_des = feat["orb_des"]
            self.ref_orb_kp_count = feat["orb_kp_count"]
        else:
            kp, des = self.orb.detectAndCompute(feat["gray"], None)
            self.ref_orb_des = des
            self.ref_orb_kp_count = len(kp) if kp else 1

        # Only set INITIALIZED on the initial frame (when feat_curr is None)
        if feat_curr is None:
            self.last_motion_label = "INITIALIZED (KF #0)"
            self.last_stage_reached = 0

        self._ref_cache = feat

    def compare_features(self, feat_ref: Any, feat_curr: Dict[str, Any]) -> float:
        # Unused directly because evaluate_frame is overridden for multi-stage gating
        return 0.0

    def evaluate_frame(
        self, frame_curr: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        feat_curr = self.extract_features(frame_curr)
        gray = feat_curr["gray"]
        small = feat_curr["small"]

        # --- STAGE 1: Ultra-Fast Pixel SAD Gate (~0.15 ms) ---
        sad_step = float(np.mean(cv2.absdiff(self.prev_small, small)))
        if sad_step < self.static_sad_thresh:
            self.prev_small = small
            curr_disp = float(np.mean(np.linalg.norm(self.accum_flow, axis=2)))
            self.last_motion_label = "STATIONARY"
            self.last_stage_reached = 1
            return curr_disp, False, feat_curr

        # --- STAGE 2: Dense Optical Flow & Ego-Motion Classification ---
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_small = small
        self.accum_flow += flow

        self.last_motion_label = self._classify_ego_motion(flow, sad_step)
        net_disp = float(np.mean(np.linalg.norm(self.accum_flow, axis=2)))

        if net_disp < self.threshold:
            self.last_stage_reached = 2
            return net_disp, False, feat_curr

        # --- STAGE 3: High-Frequency Laplacian Blur Gate ---
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < self.blur_thresh:
            self.last_motion_label += " [BLUR REJECTED]"
            self.last_stage_reached = 3
            return net_disp, False, feat_curr

        # --- STAGE 4: ORB Geometric Verification (Only runs when Stages 1-3 pass!) ---
        self.last_stage_reached = 4
        kp2, des2 = self.orb.detectAndCompute(gray, None)
        feat_curr["orb_des"] = des2
        feat_curr["orb_kp_count"] = len(kp2) if kp2 else 1

        if self.ref_orb_des is None or des2 is None or len(kp2) < 10:
            orb_dissim = 1.0
        else:
            matches = self.matcher.knnMatch(self.ref_orb_des, des2, k=2)
            good = [
                m
                for pair in matches
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]
            orb_dissim = 1.0 - (len(good) / max(self.ref_orb_kp_count, 1))

        if orb_dissim >= self.orb_thresh:
            return net_disp, True, feat_curr

        # If ORB says we still have high landmark overlap, scale score just below threshold
        return self.threshold * 0.9, False, feat_curr
