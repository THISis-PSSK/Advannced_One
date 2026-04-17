from __future__ import annotations

import cv2
from typing import Optional, Tuple


def open_camera(preferred_index: int = 0, max_index: int = 3) -> Tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """
    Open a webcam in a robust way (Linux-friendly).
    Tries indices [preferred_index, 0..max_index] with common backends.
    Returns (cap, index) or (None, None).
    """
    indices = [preferred_index] + [i for i in range(0, max_index + 1) if i != preferred_index]
    backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

    for idx in indices:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                return cap, idx
            cap.release()

    return None, None

