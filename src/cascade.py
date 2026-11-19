from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np
from base import KFSMethod


class EgoMotionAndCascadeKFS(KFSMethod):
    """
    4-Stage Hierarchical KFS Cascade + Ego-Motion Classifier:
      Stage 1: Pixel SAD (0.15 ms) -> Rejects stationary/micro-jitter frames immediately.
      Stage 2: Dense Optical Flow + Radial Divergence -> Logs Robot Ego-Motion & integrates displacement.
      Stage 3: Lazy Laplacian High-Frequency Gate -> Downscales & rejects motion-blurred frames.
      Stage 4: ORB Keypoint Overlap -> Confirms 3D geometric change without discarding flow progress.

    Args:
        flow_trigger_thresh: Stage 2 displacement gate (default: 28.0). Cumulative mean
            optical flow displacement (in pixels on `grid_size`) since the last accepted
            keyframe required to promote a frame to Stage 3. Lower values (18.0-22.0)
            capture smaller shifts and increase keyframes; higher values (35.0-45.0)
            enforce larger viewpoint changes.
        static_sad_thresh: Stage 1 early-exit gate (default: 5.5). Minimum frame-to-frame
            Sum of Absolute Differences (0-255 grayscale scale) needed to bypass the
            stationary check. Frames below this threshold exit immediately (~0.15 ms)
            without running dense optical flow, absorbing handheld micro-jitter (2.0-4.5)
            during pauses while still catching genuine motion.
        blur_thresh: Stage 3 Laplacian sharpness gate (default: 15.0). Minimum variance
            of the 2D Laplacian operator on `verify_gray`. Motion-blurred frames lose
            high-frequency edge energy and fall below this threshold, causing Stage 3
            to reject them (`[BLUR REJECTED]`). Higher values (25.0-40.0) enforce crisper
            frames; lower values (8.0-12.0) tolerate slight motion blur to preserve recall.
        orb_thresh: Stage 4 geometric dissimilarity gate (default: 0.52). Minimum landmark
            dissimilarity (`1.0 - good_matches / ref_keypoints`, range 0.0 to 1.0) between
            the candidate frame and active reference keyframe using Lowe's ratio test (0.75).
            A value of 0.52 requires at least 52% of reference ORB keypoints to leave the
            field of view (<= 48% overlap) before accepting a new keyframe. Higher values
            (0.65-0.75) suppress small rotational pans; lower values (0.40-0.45) allow
            denser viewpoint sampling.
        grid_size: Stages 1 & 2 motion resolution as `(width, height)` (default: (160, 120)).
            Proxy resolution used for pixel SAD, Farneback dense optical flow, and radial
            ego-motion classification. Changing `grid_size` proportionally scales pixel
            displacement values compared against `flow_trigger_thresh`.
        verify_size: Stages 3 & 4 verification resolution as `(width, height)`
            (default: (480, 360)). Caps the resolution used lazily inside Stage 3/4 when
            computing Laplacian blur variance and extracting 400 ORB keypoints, cutting
            CPU overhead by up to 16x on HD/1440p scans without losing corner geometry.
        orb_cooldown_frames: Stage 4 anti-thrashing cooldown (default: 2). Number of
            frames to wait before re-running Stage 4 ORB matching after an ORB rejection
            occurs, while keeping `accum_flow` intact so fast turns trigger immediately
            once landmark overlap drops.
        max_gap_frames: Temporal heartbeat guarantee (default: 60). Maximum number of
            frames allowed since the last keyframe before forcing a sharp frame
            (`[HEARTBEAT]`), even if `accum_flow` is below `flow_trigger_thresh` (set to
            `None` to disable). At 30 FPS, 60 frames guarantees at least one keyframe
            every 2.0 seconds during slow forward walks while still requiring the frame
            to pass the Stage 3 blur gate.
    """

    def __init__(
        self,
        flow_trigger_thresh: float = 28.0,  # Stage 2: cumulative flow displacement (px) to trigger
        static_sad_thresh: float = 5.5,  # Stage 1: frame-to-frame SAD threshold to exit early
        blur_thresh: float = 15.0,  # Stage 3: minimum Laplacian variance (sharpness gate)
        orb_thresh: float = 0.52,  # Stage 4: minimum ORB keypoint dissimilarity (0.0-1.0)
        grid_size: Tuple[int, int] = (
            160,
            120,
        ),  # Stages 1-2: proxy resolution for SAD & optical flow
        verify_size: Tuple[int, int] = (
            480,
            360,
        ),  # Stages 3-4: lazy downscale cap for blur & ORB
        orb_cooldown_frames: int = 2,  # Stage 4: frames to wait before retrying ORB after veto
        max_gap_frames: Optional[
            int
        ] = 60,  # Heartbeat: max frames without a KF before forcing one
    ):
        super().__init__("Stacked Cascade + EgoMotion", threshold=flow_trigger_thresh)
        self.grid_w, self.grid_h = grid_size
        self.verify_size = verify_size
        self.static_sad_thresh = static_sad_thresh
        self.blur_thresh = blur_thresh
        self.orb_thresh = orb_thresh
        self.orb_cooldown_frames = orb_cooldown_frames
        self.max_gap_frames = max_gap_frames

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
        self.prev_small: Optional[np.ndarray] = None
        self.accum_flow: Optional[np.ndarray] = None
        self.ref_orb_des: Optional[np.ndarray] = None
        self.ref_orb_kp_count: int = 1
        self.motion_history: List[Tuple[int, str]] = []
        self._orb_cooldown: int = 0
        self._frames_since_kf: int = 0

    def _get_verify_gray(self, gray: np.ndarray) -> np.ndarray:
        if gray.shape[1] > self.verify_size[0]:
            return cv2.resize(gray, self.verify_size, interpolation=cv2.INTER_AREA)
        return gray

    def _classify_ego_motion(self, flow: np.ndarray, sad_step: float) -> str:
        if sad_step < self.static_sad_thresh:
            return "STATIONARY"

        u, v = flow[..., 0], flow[..., 1]
        cam_vx = -float(np.median(u))
        cam_vy = -float(np.median(v))

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
        # Only compute the single 160x120 resize needed for Stages 1 & 2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(
            gray, (self.grid_w, self.grid_h), interpolation=cv2.INTER_AREA
        )
        return {
            "gray": gray,
            "small": small,
            "orb_des": None,
            "orb_kp_count": 0,
        }

    def set_reference(
        self, frame: np.ndarray, feat_curr: Optional[Dict[str, Any]] = None
    ) -> None:
        feat = feat_curr if feat_curr is not None else self.extract_features(frame)
        self.prev_small = feat["small"]
        self.accum_flow = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)
        self._orb_cooldown = 0
        self._frames_since_kf = 0

        if feat.get("orb_des") is not None:
            self.ref_orb_des = feat["orb_des"]
            self.ref_orb_kp_count = feat["orb_kp_count"]
        else:
            verify_gray = self._get_verify_gray(feat["gray"])
            kp, des = self.orb.detectAndCompute(verify_gray, None)
            self.ref_orb_des = des
            self.ref_orb_kp_count = len(kp) if kp else 1

        if feat_curr is None:
            self.last_motion_label = "INITIALIZED (KF #0)"
            self.last_stage_reached = 0

        self._ref_cache = feat

    def compare_features(self, feat_ref: Any, feat_curr: Dict[str, Any]) -> float:
        return 0.0

    def evaluate_frame(
        self, frame_curr: np.ndarray
    ) -> Tuple[float, bool, Dict[str, Any]]:
        self._frames_since_kf += 1
        if self._orb_cooldown > 0:
            self._orb_cooldown -= 1

        feat_curr = self.extract_features(frame_curr)
        small = feat_curr["small"]

        heartbeat_due = (
            self.max_gap_frames is not None
            and self._frames_since_kf >= self.max_gap_frames
        )

        # --- STAGE 1: Ultra-Fast Pixel SAD Gate (~0.15 ms) ---
        sad_step = float(np.mean(cv2.absdiff(self.prev_small, small)))
        if sad_step < self.static_sad_thresh and not heartbeat_due:
            self.prev_small = small
            self.last_motion_label = "STATIONARY"
            self.last_stage_reached = 1
            return 0.0, False, feat_curr

        # --- STAGE 2: Dense Optical Flow & Ego-Motion Classification ---
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_small = small
        self.accum_flow += flow

        self.last_motion_label = self._classify_ego_motion(flow, sad_step)
        mag = cv2.magnitude(self.accum_flow[..., 0], self.accum_flow[..., 1])
        net_disp = float(np.mean(mag))

        if net_disp < self.threshold and not heartbeat_due:
            self.last_stage_reached = 2
            return net_disp, False, feat_curr

        # --- STAGE 3: Lazy Downscale + High-Frequency Laplacian Blur Gate ---
        verify_gray = self._get_verify_gray(feat_curr["gray"])
        blur_var = float(cv2.Laplacian(verify_gray, cv2.CV_64F).var())
        if blur_var < self.blur_thresh:
            self.last_motion_label += " [BLUR REJECTED]"
            self.last_stage_reached = 3
            return net_disp, False, feat_curr

        if heartbeat_due and net_disp < self.threshold:
            self.last_motion_label += " [HEARTBEAT]"
            self.last_stage_reached = 3
            return net_disp, True, feat_curr

        if self._orb_cooldown > 0:
            self.last_stage_reached = 3
            return self.threshold * 0.9, False, feat_curr

        # --- STAGE 4: ORB Geometric Verification ---
        self.last_stage_reached = 4
        kp2, des2 = self.orb.detectAndCompute(verify_gray, None)
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

        # Keep accumulated displacement intact; set short cooldown so ORB doesn't thrash
        self._orb_cooldown = self.orb_cooldown_frames
        return self.threshold * 0.9, False, feat_curr
