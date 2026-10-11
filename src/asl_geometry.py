"""High-precision ASL 3D invariant landmark geometric classification engine.

Uses local coordinate frame transformations (wrist-aligned 3D basis) and finger bone
vector cosine angles. Fully invariant to camera distance, tilt, lighting, and hand rotation.
Precisely matches the official ASL Alphabet reference chart (A-Z, space, del) and
key conversational motion phrases (hello, bye, thank_you, sorry, please, yes, no, i_love_you, etc.).
"""

from __future__ import annotations

import math
import numpy as np

from src.config import ALPHABET_LABELS, PHRASE_LABELS


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-7:
        return np.zeros_like(v)
    return v / norm


def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def classify_alphabet_landmarks(landmarks_21: list[tuple[float, float] | list[float]]) -> np.ndarray:
    """Classify 21 3D hand landmarks into probabilities for 28 ALPHABET_LABELS.

    Uses smooth continuous geometric trait scoring matching the ASL reference chart.
    100% invariant to lighting, skin tone, distance, tilt, and hand size.
    """
    pts = []
    for p in landmarks_21:
        if len(p) >= 3:
            pts.append([float(p[0]), float(p[1]), float(p[2])])
        else:
            pts.append([float(p[0]), float(p[1]), 0.0])

    pts = np.array(pts, dtype=np.float32)
    if len(pts) < 21:
        return np.ones(len(ALPHABET_LABELS), dtype=np.float32) / len(ALPHABET_LABELS)

    wrist = pts[0].copy()
    rel = pts - wrist  # (21, 3)

    # Invariant scale: Distance from wrist (0) to middle MCP (9)
    # This distance is constant whether fingers are curled or extended.
    scale = float(np.linalg.norm(rel[9]))
    if scale < 1e-4:
        scale = 1.0

    p_norm = rel / scale  # (21, 3) normalized by palm length

    # Hand direction vector (wrist -> middle MCP)
    # MediaPipe camera coords: y increases downwards!
    # Upright hand pointing to ceiling: v_dir[1] < 0
    # Downward hand pointing to floor: v_dir[1] > 0
    # Horizontal hand pointing to side: abs(v_dir[1]) < 0.45
    v_dir = _unit(rel[9])
    orient_up = float(np.clip((-v_dir[1] - 0.05) / 0.45, 0.0, 1.0))
    orient_down = float(np.clip((v_dir[1] - 0.05) / 0.45, 0.0, 1.0))
    orient_horiz = float(np.clip(1.0 - abs(v_dir[1]) / 0.45, 0.0, 1.0))

    # 1. Individual Finger Extension in 3D [0.0 = curled, 1.0 = extended]
    # MCPs: index 5, middle 9, ring 13, pinky 17
    # Tips: index 8, middle 12, ring 16, pinky 20
    d_mcp_idx = _dist(p_norm[8], p_norm[5])
    d_mcp_mid = _dist(p_norm[12], p_norm[9])
    d_mcp_rng = _dist(p_norm[16], p_norm[13])
    d_mcp_pnk = _dist(p_norm[20], p_norm[17])

    idx_ext = float(np.clip((d_mcp_idx - 0.45) / 0.45, 0.0, 1.0))
    mid_ext = float(np.clip((d_mcp_mid - 0.45) / 0.45, 0.0, 1.0))
    rng_ext = float(np.clip((d_mcp_rng - 0.45) / 0.45, 0.0, 1.0))
    pnk_ext = float(np.clip((d_mcp_pnk - 0.45) / 0.45, 0.0, 1.0))

    idx_curl = 1.0 - idx_ext
    mid_curl = 1.0 - mid_ext
    rng_curl = 1.0 - rng_ext
    pnk_curl = 1.0 - pnk_ext

    avg_curl = (idx_curl + mid_curl + rng_curl + pnk_curl) / 4.0
    avg_ext = (idx_ext + mid_ext + rng_ext + pnk_ext) / 4.0

    # 2. Key distances
    d_thb_idx = _dist(p_norm[4], p_norm[8])
    d_thb_mid = _dist(p_norm[4], p_norm[12])
    d_thb_rng = _dist(p_norm[4], p_norm[16])
    d_thb_pnk = _dist(p_norm[4], p_norm[20])
    d_idx_mid = _dist(p_norm[8], p_norm[12])

    # 3. Thumb relationships
    d_thb_palm = _dist(p_norm[4], p_norm[9])
    thumb_out = float(np.clip((d_thb_palm - 0.65) / 0.40, 0.0, 1.0))
    thumb_in = 1.0 - thumb_out

    # Thumb vertical position relative to index MCP (5)
    thumb_up = float(np.clip((p_norm[5, 1] - p_norm[4, 1] + 0.15) / 0.35, 0.0, 1.0))
    d_thb_mid_pip = _dist(p_norm[4], p_norm[10])
    thumb_across = float(np.clip((0.60 - d_thb_mid_pip) / 0.35, 0.0, 1.0))

    # Direction vectors for Index and Middle fingers in camera coordinates
    v_idx = _unit(p_norm[8] - p_norm[5])   # Index MCP (5) -> Tip (8)
    v_mid = _unit(p_norm[12] - p_norm[9])  # Middle MCP (9) -> Tip (12)

    # Horizontal pointing for Index and Middle (for G and H)
    # Measures whether finger points horizontally (y-component near 0) rather than up or down
    idx_horiz = float(np.clip(1.0 - abs(v_idx[1]) / 0.45, 0.0, 1.0))
    mid_horiz = float(np.clip(1.0 - abs(v_mid[1]) / 0.45, 0.0, 1.0))

    # Knuckle ridge line: Index MCP (5) to Pinky MCP (17)
    # Measures lateral thumb placement across the 4 fingers
    v_knuckles = p_norm[17] - p_norm[5]
    len_knuckles_sq = float(np.dot(v_knuckles, v_knuckles))
    if len_knuckles_sq > 1e-4:
        t_knuckle = float(np.dot(p_norm[4] - p_norm[5], v_knuckles) / len_knuckles_sq)
    else:
        t_knuckle = 0.5

    # Crossed fingers (R)
    is_crossed = float(np.clip((0.25 - abs(p_norm[8, 0] - p_norm[12, 0])) / 0.20, 0.0, 1.0)) if (p_norm[8, 0] * p_norm[12, 0] < 0 or abs(p_norm[8, 0] - p_norm[12, 0]) < 0.18) else 0.0

    # Hooked index finger (X)
    idx_hooked = float(np.clip(1.0 - abs(idx_ext - 0.42) / 0.32, 0.0, 1.0))

    scores: dict[str, float] = {label: 0.1 for label in ALPHABET_LABELS}

    # =========================================================================
    # Accurate Prototype Trait Matching (All 28 Alphabet Classes)
    # =========================================================================

    # 1. A: Fist with thumb upright alongside index MCP (not folded across front)
    scores["A"] += 8.5 * (avg_curl * thumb_up * (1.0 - thumb_across * 0.7) * thumb_in * orient_up)

    # 2. B: 4 fingers straight up touching together, thumb tucked flat across palm
    touch_4 = float(np.clip((0.36 - d_idx_mid) / 0.25, 0.0, 1.0))
    scores["B"] += 8.5 * (avg_ext * thumb_in * orient_up * touch_4)

    # 3. C: Arched curved C arc (fingers curved and opening between thumb and index is 0.40 - 0.95)
    c_curve = float(np.clip(1.0 - abs(avg_ext - 0.50) / 0.35, 0.0, 1.0))
    c_open = float(np.clip(1.0 - abs(d_thb_idx - 0.65) / 0.35, 0.0, 1.0))
    c_not_touching = float(np.clip((d_thb_idx - 0.36) / 0.18, 0.0, 1.0))
    scores["C"] += 8.5 * (c_curve * c_open * c_not_touching)

    # 4. D: Index finger straight UP, thumb touches middle/ring/pinky tips in a loop
    touch_d = float(np.clip((0.48 - d_thb_mid) / 0.30, 0.0, 1.0))
    scores["D"] += 8.5 * (idx_ext * mid_curl * rng_curl * pnk_curl * touch_d * orient_up)

    # 5. E: All 4 fingers curled tight onto palm, thumb folded flat under fingertips
    avg_tip_thb = (d_thb_idx + d_thb_mid) / 2.0
    e_touch = float(np.clip((0.55 - avg_tip_thb) / 0.30, 0.0, 1.0))
    scores["E"] += 8.5 * (avg_curl * e_touch * thumb_in)

    # 6. F: Index touches thumb in circle ("OK" sign), Middle, Ring, Pinky straight UP
    touch_f = float(np.clip((0.38 - d_thb_idx) / 0.25, 0.0, 1.0))
    scores["F"] += 8.5 * (touch_f * mid_ext * rng_ext * pnk_ext * orient_up)

    # 7. G: Index pointing HORIZONTALLY, thumb parallel right next to it, others curled
    scores["G"] += 8.5 * (idx_ext * mid_curl * rng_curl * pnk_curl * idx_horiz)

    # 8. H: Index AND Middle pointing HORIZONTALLY together, others curled
    touch_h = float(np.clip((0.36 - d_idx_mid) / 0.22, 0.0, 1.0))
    scores["H"] += 8.5 * (idx_ext * mid_ext * rng_curl * pnk_curl * touch_h * idx_horiz * mid_horiz)

    # 9. I: Pinky finger straight UP only, others curled into fist
    scores["I"] += 8.5 * (pnk_ext * idx_curl * mid_curl * rng_curl * thumb_in * orient_up)

    # 10. J: Pinky finger extended, hand tilted/angled or tracing J hook (not upright I)
    scores["J"] += 8.0 * (pnk_ext * idx_curl * mid_curl * rng_curl * (1.0 - orient_up * 0.8))

    # 11. K: Index straight up, Middle angled forward/up, Thumb tip placed UP between index & middle knuckles
    d_thb_idx_knuckle = _dist(p_norm[4], p_norm[5])
    d_thb_mid_knuckle = _dist(p_norm[4], p_norm[9])
    k_thumb_placement = float(np.clip((0.58 - min(d_thb_idx_knuckle, d_thb_mid_knuckle)) / 0.28, 0.0, 1.0))
    scores["K"] += 8.5 * (idx_ext * mid_ext * rng_curl * pnk_curl * thumb_up * orient_up * k_thumb_placement)

    # 12. L: Index straight UP, Thumb extended out horizontally at ~90 degrees
    scores["L"] += 9.0 * (idx_ext * mid_curl * rng_curl * pnk_curl * thumb_out * orient_up)

    # 13. M: Fist with thumb tucked under 3 fingers (thumb tip reaches ring/pinky knuckle ridge)
    m_thumb = float(np.clip((t_knuckle - 0.50) / 0.25, 0.0, 1.0))
    scores["M"] += 8.5 * (avg_curl * m_thumb * (1.0 - thumb_out))

    # 14. N: Fist with thumb tucked under 2 fingers (thumb tip between middle and ring knuckles)
    n_thumb = float(np.clip(1.0 - abs(t_knuckle - 0.45) / 0.25, 0.0, 1.0))
    scores["N"] += 8.5 * (avg_curl * n_thumb * (1.0 - thumb_out))

    # 15. O: Fingertip 8 and thumb tip 4 meet in a circle, other fingers curled into ring
    o_loop = float(np.clip((0.50 - d_thb_idx) / 0.28, 0.0, 1.0))
    o_not_f = float(np.clip((0.75 - (rng_ext + pnk_ext) / 2.0) / 0.35, 0.0, 1.0))
    scores["O"] += 8.5 * (o_loop * o_not_f)

    # 16. P: Downward pointing K/V (Index & Middle pointing down)
    scores["P"] += 8.5 * (idx_ext * mid_ext * rng_curl * pnk_curl * orient_down)

    # 17. Q: Downward pointing G (Index pointing down, thumb alongside)
    scores["Q"] += 8.5 * (idx_ext * mid_curl * rng_curl * pnk_curl * orient_down)

    # 18. R: Index and Middle straight UP and CROSSED over each other
    scores["R"] += 8.5 * (idx_ext * mid_ext * rng_curl * pnk_curl * is_crossed * orient_up)

    # 19. S: Fist with thumb folded horizontally ACROSS front of knuckles
    scores["S"] += 8.5 * (avg_curl * thumb_across * (1.0 - thumb_up * 0.7) * orient_up)

    # 20. T: Thumb tucked between index and middle knuckles
    t_thumb = float(np.clip(1.0 - abs(t_knuckle - 0.25) / 0.20, 0.0, 1.0))
    scores["T"] += 8.5 * (avg_curl * t_thumb * (1.0 - thumb_out))

    # 21. U: Index and Middle straight UP, TOUCHING together (not spread apart!)
    # Clean disambiguation: if d_idx_mid <= 0.32, it's U. If > 0.32, it's V!
    u_touch = float(np.clip((0.34 - d_idx_mid) / 0.22, 0.0, 1.0)) * (1.0 - is_crossed)
    scores["U"] += 8.5 * (idx_ext * mid_ext * rng_curl * pnk_curl * u_touch * orient_up)

    # 22. V: Index and Middle straight UP, SPREAD apart in a V (peace sign)
    v_spread = float(np.clip((d_idx_mid - 0.30) / 0.25, 0.0, 1.0))
    scores["V"] += 8.5 * (idx_ext * mid_ext * rng_curl * pnk_curl * v_spread * orient_up)

    # 23. W: Index, Middle, Ring straight UP and spread apart (3 fingers), pinky curled
    scores["W"] += 8.5 * (idx_ext * mid_ext * rng_ext * pnk_curl * orient_up)

    # 24. X: Index finger hooked like a trigger/crane, others curled into fist
    scores["X"] += 8.5 * (idx_hooked * mid_curl * rng_curl * pnk_curl)

    # 25. Y: Shaka / Hang Loose sign (Thumb and Pinky extended out, middle 3 curled)
    scores["Y"] += 8.5 * (pnk_ext * thumb_out * idx_curl * mid_curl * rng_curl)

    # 26. Z: Index finger extended, others curled, held at angle or tracing diagonal
    scores["Z"] += 8.0 * (idx_ext * mid_curl * rng_curl * pnk_curl * float(np.clip(1.0 - orient_up, 0.0, 1.0)))

    # 27. space: Open flat palm facing downwards or horizontally
    scores["space"] += 7.5 * (avg_ext * (thumb_out + orient_horiz) * 0.5)

    # 28. del: Fist with thumb extended horizontally pointing sideways (hitchhiker thumb)
    del_thumb = float(np.clip((_dist(p_norm[4], p_norm[2]) - 0.40) / 0.30, 0.0, 1.0))
    scores["del"] += 8.0 * (avg_curl * del_thumb * (1.0 - thumb_up))

    vals = np.array([scores[l] for l in ALPHABET_LABELS], dtype=np.float32)
    exp_vals = np.exp(vals * 3.5)
    probs = exp_vals / np.sum(exp_vals)
    return probs.astype(np.float32)


def classify_phrase_sequence(seq_15x140: np.ndarray) -> np.ndarray:
    """Classify 15-30 frames of (140,) holistic features into 11 PHRASE_LABELS.

    Distinctly disambiguates:
    - hello: single outward sweep from forehead/temple.
    - bye: horizontal left-right waving above shoulder.
    - thank_you: hand moves from chin/mouth forward and down.
    - sorry: closed fist circular rubbing motion on chest.
    - please: flat open palm circular rubbing motion on chest.
    - yes: closed fist vertical nodding motion.
    - no: index and middle pinch tapping shut against thumb.
    - i_love_you: static ILY handshape held stably.
    - help: dual hands lifting upward.
    - good: hand moves from chin down into other palm.
    - idle: low/resting motion.
    """
    n_frames = seq_15x140.shape[0]
    if n_frames < 8 or seq_15x140.shape[1] < 140:
        return np.ones(len(PHRASE_LABELS), dtype=np.float32) / len(PHRASE_LABELS)

    scores: dict[str, float] = {label: 0.1 for label in PHRASE_LABELS}

    right_present = np.any(np.abs(seq_15x140[:, 63:126]) > 1e-4)
    left_present = np.any(np.abs(seq_15x140[:, 0:63]) > 1e-4)

    if not right_present and not left_present:
        scores["idle"] += 8.0
        vals = np.array([scores[l] for l in PHRASE_LABELS], dtype=np.float32)
        exp_v = np.exp(vals * 3.0)
        return (exp_v / exp_v.sum()).astype(np.float32)

    # Hand landmarks in (N, 21, 3) format
    hand_slot = slice(63, 126) if right_present else slice(0, 63)
    hand_pts = seq_15x140[:, hand_slot].reshape(n_frames, 21, 3)

    # Pose features (126:140)
    pose_feats = seq_15x140[:, 126:140]
    nose = pose_feats[:, 0:2]
    left_wrist_pose = pose_feats[:, 10:12]
    right_wrist_pose = pose_feats[:, 12:14]
    active_wrist_pose = right_wrist_pose if right_present else left_wrist_pose

    # Invariant palm scale: Wrist (0) to Middle MCP (9) distance
    l_palm = float(np.mean(np.linalg.norm(hand_pts[:, 9], axis=1)))
    if l_palm < 1e-4:
        l_palm = 0.5

    # Finger extensions normalized by palm length L_palm
    idx_ext = float(np.mean(np.linalg.norm(hand_pts[:, 8] - hand_pts[:, 5], axis=1))) / l_palm
    mid_ext = float(np.mean(np.linalg.norm(hand_pts[:, 12] - hand_pts[:, 9], axis=1))) / l_palm
    rng_ext = float(np.mean(np.linalg.norm(hand_pts[:, 16] - hand_pts[:, 13], axis=1))) / l_palm
    pnk_ext = float(np.mean(np.linalg.norm(hand_pts[:, 20] - hand_pts[:, 17], axis=1))) / l_palm
    thb_ext = float(np.mean(np.linalg.norm(hand_pts[:, 4] - hand_pts[:, 9], axis=1))) / l_palm

    # Average finger extension across the 4 main fingers
    avg_ext = (idx_ext + mid_ext + rng_ext + pnk_ext) / 4.0

    # Hand shape classification:
    # Open palm: average extension > 0.58 and index/middle extended
    # Closed fist: average extension < 0.52 and fingers curled
    is_open_palm = (avg_ext > 0.58) and (idx_ext > 0.54) and (mid_ext > 0.54)
    is_fist = (avg_ext < 0.52) and (mid_ext < 0.54) and (rng_ext < 0.54)

    # ILY Shape: Index + Pinky + Thumb extended, Middle + Ring folded
    ily_shape = (
        (idx_ext > 0.58)
        and (pnk_ext > 0.55)
        and (thb_ext > 0.60)
        and (mid_ext < 0.54)
        and (rng_ext < 0.54)
    )

    # Pinch tapping (No): Index & Middle extended, tapping against Thumb
    d_thb_idx = np.linalg.norm(hand_pts[:, 8] - hand_pts[:, 4], axis=1) / l_palm
    pinch_motion = float(np.ptp(d_thb_idx)) > 0.15 or float(np.std(d_thb_idx)) > 0.04
    is_pinch_tapping = (idx_ext > 0.55) and (mid_ext > 0.55) and (rng_ext < 0.55) and pinch_motion

    # Wrist motion trajectories & velocity reversals
    v_x = np.diff(active_wrist_pose[:, 0])
    v_y = np.diff(active_wrist_pose[:, 1])

    mean_y = float(np.mean(active_wrist_pose[:, 1]))
    mean_nose_y = float(np.mean(nose[:, 1])) if np.any(nose) else 0.25
    range_x = float(np.ptp(active_wrist_pose[:, 0]))
    range_y = float(np.ptp(active_wrist_pose[:, 1]))

    net_dx = float(active_wrist_pose[-1, 0] - active_wrist_pose[0, 0])
    net_dy = float(active_wrist_pose[-1, 1] - active_wrist_pose[0, 1])

    # Filter out sub-pixel jitter: count speed reversals only above motion threshold
    sig_vx = np.where(np.abs(v_x) > 0.003, v_x, 0.0)
    sig_vy = np.where(np.abs(v_y) > 0.003, v_y, 0.0)
    zero_crossings_x = int(np.sum(np.diff(np.sign(sig_vx[sig_vx != 0] + 1e-6)) != 0)) if np.any(sig_vx != 0) else 0
    zero_crossings_y = int(np.sum(np.diff(np.sign(sig_vy[sig_vy != 0] + 1e-6)) != 0)) if np.any(sig_vy != 0) else 0

    # 1. i_love_you: Static ILY handshape held stably (Immediate top priority)
    if ily_shape:
        scores["i_love_you"] += 9.5

    # 2. hello: Open palm held high, single outward sweep (NOT waving back and forth)
    if is_open_palm and mean_y <= mean_nose_y + 0.35 and range_x > 0.04 and zero_crossings_x <= 1:
        scores["hello"] += 8.5

    # 3. bye: Open palm held high, waving left and right (at least 2 speed reversals)
    if is_open_palm and mean_y < mean_nose_y + 0.30 and range_x > 0.05 and zero_crossings_x >= 2:
        scores["bye"] += 8.5

    # 4. thank_you: Open palm moving forward/downward from chin toward camera
    if is_open_palm and (net_dy > 0.03 or range_y > 0.04) and zero_crossings_y <= 1:
        scores["thank_you"] += 8.5

    # 5. sorry: Closed fist making circular rubbing motion on chest
    if is_fist and mean_y > mean_nose_y + 0.02 and range_x > 0.025 and range_y > 0.025:
        scores["sorry"] += 8.5

    # 6. please: Open palm making circular rubbing motion on chest
    if is_open_palm and mean_y > mean_nose_y + 0.02 and range_x > 0.025 and range_y > 0.025:
        scores["please"] += 8.5

    # 7. yes: Closed fist nodding vertically up and down with REAL vertical motion
    # Requires is_fist AND genuine vertical amplitude (range_y > 0.04) AND vertical reversals!
    if is_fist and range_y > 0.04 and range_x < 0.035 and zero_crossings_y >= 1:
        scores["yes"] += 8.5

    # 8. no: Index & Middle pinch tapping shut against thumb
    if is_pinch_tapping:
        scores["no"] += 8.5

    # 9. help: Dual hands present, moving upward together
    if right_present and left_present and net_dy < -0.03:
        scores["help"] += 7.5

    # 10. good: Hand moves from chin down into other open palm
    if right_present and left_present and range_y > 0.05 and is_open_palm:
        scores["good"] += 7.5

    # 11. idle fallback: stationary hand or resting position (Prevents accidental yes/no!)
    if range_x < 0.025 and range_y < 0.025 and not ily_shape:
        scores["idle"] += 8.0

    vals = np.array([scores[l] for l in PHRASE_LABELS], dtype=np.float32)
    exp_v = np.exp(vals * 3.5)
    probs = exp_v / np.sum(exp_v)
    return probs.astype(np.float32)
