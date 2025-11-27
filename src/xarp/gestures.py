from xarp import XR, run_xr_app
from xarp.storage.local_file_system import SessionRepositoryLocalFileSystem
from xarp.data_models.spatial import distance, cosine_similarity
import numpy as np

PALM = 0
WRIST = 1

THUMB_METACARPAL = 2
THUMB_PROXIMAL = 3
THUMB_DISTAL = 4
THUMB_TIP = 5

INDEX_METACARPAL = 6
INDEX_PROXIMAL = 7
INDEX_INTERMEDIATE = 8
INDEX_DISTAL = 9
INDEX_TIP = 10

MIDDLE_METACARPAL = 11
MIDDLE_PROXIMAL = 12
MIDDLE_INTERMEDIATE = 13
MIDDLE_DISTAL = 14
MIDDLE_TIP = 15

RING_METACARPAL = 16
RING_PROXIMAL = 17
RING_INTERMEDIATE = 18
RING_DISTAL = 19
RING_TIP = 20

PINKY_METACARPAL = 21
PINKY_PROXIMAL = 22
PINKY_INTERMEDIATE = 23
PINKY_DISTAL = 24
PINKY_TIP = 25

THUMB = (
    THUMB_METACARPAL,
    THUMB_PROXIMAL,
    THUMB_DISTAL,
    THUMB_TIP,
)

INDEX = (
    INDEX_METACARPAL,
    INDEX_PROXIMAL,
    INDEX_INTERMEDIATE,
    INDEX_DISTAL,
    INDEX_TIP,
)

MIDDLE = (
    MIDDLE_METACARPAL,
    MIDDLE_PROXIMAL,
    MIDDLE_INTERMEDIATE,
    MIDDLE_DISTAL,
    MIDDLE_TIP,
)

RING = (
    RING_METACARPAL,
    RING_PROXIMAL,
    RING_INTERMEDIATE,
    RING_DISTAL,
    RING_TIP,
)

PINKY = (
    PINKY_METACARPAL,
    PINKY_PROXIMAL,
    PINKY_INTERMEDIATE,
    PINKY_DISTAL,
    PINKY_TIP
)

FINGERS = THUMB, INDEX, MIDDLE, RING, PINKY
DIGITS = INDEX, MIDDLE, RING, PINKY


def finger_extension(hand, chain):
    """
    chain: list of joint indices from base (metacarpal) to tip.
    Returns E in [0,1], where:
      1 → perfectly straight
      0 → maximally curled (practically ~0.3–0.6 for real hands)
    """
    pts = [hand[i].position for i in chain]

    path_len = 0.0
    for a, b in zip(pts, pts[1:]):
        path_len += distance(a, b)

    chord_len = distance(pts[0], pts[-1])

    ext = chord_len / path_len
    ext = float(max(0.0, min(1.0, ext)))  # clamp numeric noise
    return ext


def finger_flexion(hand, chain):
    """
    Complement of finger_extension:
      1 → strongly flexed
      0 → perfectly straight
    """
    return float(1.0 - finger_extension(hand, chain))


def palm_normal(hand):
    """
    Palm normal from wrist–index–middle metacarpals.
    Sign depends on coordinate convention.
    """
    wrist = hand[WRIST].position
    idx_m = hand[INDEX_METACARPAL].position
    mid_m = hand[MIDDLE_METACARPAL].position
    return np.cross(idx_m - wrist, mid_m - wrist)


# ----------------- pinch gestures -----------------

def pinch(hand, threshold=0.015):
    """
    Thumb–index pinch. Metric: distance thumb_tip–index_tip (m).
    Default threshold ~1.5 cm.
    """
    thumb_tip = hand[THUMB_TIP].position
    index_tip = hand[INDEX_TIP].position
    dist = distance(index_tip, thumb_tip)
    return dist if threshold is None else dist < threshold


def pinch_middle(hand, threshold=0.015):
    thumb_tip = hand[THUMB_TIP].position
    middle_tip = hand[MIDDLE_TIP].position
    dist = distance(middle_tip, thumb_tip)
    return dist if threshold is None else dist < threshold


def pinch_ring(hand, threshold=0.015):
    thumb_tip = hand[THUMB_TIP].position
    ring_tip = hand[RING_TIP].position
    dist = distance(ring_tip, thumb_tip)
    return dist if threshold is None else dist < threshold


def double_pinch(hands, threshold=None):
    if threshold is not None:
        if not pinch(hands.left, threshold) or not pinch(hands.right, threshold):
            return None

    l_thumb = hands.left[THUMB_TIP].position
    l_index = hands.left[INDEX_TIP].position
    l_center = (l_thumb + l_index) / 2

    # Right hand pinch point
    r_thumb = hands.right[THUMB_TIP].position
    r_index = hands.right[INDEX_TIP].position
    r_center = (r_thumb + r_index) / 2

    return distance(l_center, r_center)


# ----------------- pose / extension gestures -----------------

def fist(hand, threshold=0.6):
    """
    Metric: mean flexion of the four long fingers in [0,1].
    Higher → more fist-like.
    Reasonable default: threshold=0.6
    """
    flex_vals = [finger_flexion(hand, chain) for chain in DIGITS]
    metric = sum(flex_vals) / len(flex_vals)
    return metric if threshold is None else metric > threshold


def open_hand(hand, threshold=0.8):
    """
    Metric: mean extension of the four long fingers in [0,1].
    Higher → more open.
    Reasonable default: threshold=0.8
    """
    ext_vals = [finger_extension(hand, chain) for chain in DIGITS]
    metric = sum(ext_vals) / len(ext_vals)
    return metric if threshold is None else metric > threshold


def point(hand, threshold=0.3):
    """
    Index extended, others less extended.
    Metric = ext(index) - max(ext(middle, ring, pinky)).
    Range roughly [-1,1]. Point-like when > 0.3.
    """
    idx = finger_extension(hand, INDEX)
    mid = finger_extension(hand, MIDDLE)
    rng = finger_extension(hand, RING)
    pnk = finger_extension(hand, PINKY)

    metric = idx - max(mid, rng, pnk)
    return metric if threshold is None else metric > threshold


def victory(hand, threshold=0.5):
    """
    Index + middle extended; ring + pinky more flexed.
    Metric = (ext(index)+ext(middle)) - (ext(ring)+ext(pinky)).
    Straight V should easily exceed 0.5.
    """
    idx = finger_extension(hand, INDEX)
    mid = finger_extension(hand, MIDDLE)
    rng = finger_extension(hand, RING)
    pnk = finger_extension(hand, PINKY)

    metric = (idx + mid) - (rng + pnk)
    return metric if threshold is None else metric > threshold


# ----------------- thumb orientation -----------------

def thumbs_up(hand, threshold=0.7):
    """
    Metric: cosine similarity between thumb direction and palm normal.
    Range [-1,1]; > 0.7 means roughly aligned.
    """
    thumb_base = hand[THUMB_METACARPAL].position
    thumb_tip = hand[THUMB_TIP].position
    thumb_vec = thumb_tip - thumb_base

    n = palm_normal(hand)
    metric = cosine_similarity(thumb_vec, n)
    return metric if threshold is None else metric > threshold


# ----------------- flat palm -----------------

def flat_palm(hand, threshold=0.015):
    """
    Metric: max distance (m) of metacarpal joints (index/middle/ring/pinky)
    from their best-fit plane. Smaller → flatter palm.
    For a hand ~7–9 cm wide and decent tracking, 0.015 m (~1.5 cm)
    is a reasonable flatness threshold.
    """
    pts = np.array([
        hand[INDEX_METACARPAL].position,
        hand[MIDDLE_METACARPAL].position,
        hand[RING_METACARPAL].position,
        hand[PINKY_METACARPAL].position,
    ])

    pts_centered = pts - pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts_centered, full_matrices=False)
    normal = vh[-1]

    dists = np.abs(pts_centered @ normal)
    metric = float(np.max(dists))
    return metric if threshold is None else metric < threshold


# ----------------- coarse grab -----------------

def coarse_grab(hand, threshold=0.035):
    """
    Metric: mean distance (m) from thumb tip to all other fingertips.
    Lower → more closed / grab-like (no contact/force semantics).
    For an adult hand, open spread ≈ 5–7 cm; closed fist ≈ 1.5–3 cm.
    Default threshold 3.5 cm (0.035 m) is a reasonable mid-point.
    """
    thumb_tip = hand[THUMB_TIP].position
    tips = [
        hand[INDEX_TIP].position,
        hand[MIDDLE_TIP].position,
        hand[RING_TIP].position,
        hand[PINKY_TIP].position,
    ]
    dists = [distance(t, thumb_tip) for t in tips]
    metric = float(np.mean(dists))
    return metric if threshold is None else metric < threshold
