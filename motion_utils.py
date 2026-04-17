from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class MotionConfig:
    window: int = 10
    min_dx: float = 0.22
    max_dy: float = 0.18
    cooldown_ms: int = 700


class MotionDetector:
    """
    Simple motion detector using the wrist landmark (id 0) in normalized coords.
    Detects:
      - swipe_left  -> DELETE
      - swipe_right -> SPACE
    """

    def __init__(self, config: MotionConfig | None = None):
        self.cfg = config or MotionConfig()
        self._pts: Deque[Tuple[float, float, int]] = deque(maxlen=self.cfg.window)
        self._last_fire_ms: int = -10**9

    def reset(self) -> None:
        self._pts.clear()
        self._last_fire_ms = -10**9

    def update(self, result, timestamp_ms: int) -> Optional[str]:
        if result is None or not getattr(result, "hand_landmarks", None):
            self._pts.clear()
            return None

        # Use first detected hand
        wrist = result.hand_landmarks[0][0]
        self._pts.append((float(wrist.x), float(wrist.y), int(timestamp_ms)))

        if len(self._pts) < self._pts.maxlen:
            return None

        # Cooldown
        if timestamp_ms - self._last_fire_ms < self.cfg.cooldown_ms:
            return None

        x0, y0, _ = self._pts[0]
        x1, y1, _ = self._pts[-1]
        dx = x1 - x0
        dy = y1 - y0

        if abs(dy) > self.cfg.max_dy:
            return None

        if dx >= self.cfg.min_dx:
            self._last_fire_ms = timestamp_ms
            self._pts.clear()
            return "SPACE"

        if dx <= -self.cfg.min_dx:
            self._last_fire_ms = timestamp_ms
            self._pts.clear()
            return "DELETE"

        return None
