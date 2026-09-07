"""Motion vectors for the temporal path.

DLSS is a temporal upscaler: it is told where each pixel came from and reuses
detail across frames. A decoded video carries no motion vectors, so they are
estimated here with DIS optical flow — the same way the reference application
does it (``src/video/guides.py``).

Without them DLSS still runs, treating every frame as a still, and both temporal
stability and detail on video are noticeably worse. A batch of unrelated images
must NOT go through this: it would carry detail from one picture into the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._deps import TSDependencyManager

#: Above this mean absolute difference the frames are a cut, not motion.
SCENE_CUT_SCORE = 0.24


@dataclass
class GuideFrame:
    """What the worker needs to know about this frame's relation to the last."""

    motion: np.ndarray
    reset: bool
    scene_score: float


class TemporalGuideGenerator:
    """Estimate the guide buffers an encoded video does not contain."""

    def __init__(self, width: int, height: int, flow_width: int = 640) -> None:
        cv2 = TSDependencyManager.import_optional("cv2")
        if cv2 is None:
            raise RuntimeError(
                "[TS DLSS Upscaler] opencv-python-headless is required for the temporal "
                "path. Install it, or switch 'temporal' off."
            )
        self._cv2 = cv2
        self.width = width
        self.height = height
        scale = min(1.0, flow_width / width)
        # Even sizes and never below 64: the flow solver needs both.
        self.flow_width = max(64, int(round(width * scale / 2) * 2))
        self.flow_height = max(64, int(round(height * scale / 2) * 2))
        self.previous_gray: np.ndarray | None = None
        self.zero_motion = np.zeros((height, width, 2), dtype=np.float16)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.dis.setUseSpatialPropagation(True)
        self.dis.setFinestScale(1)

    def _small_gray(self, rgba: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        return cv2.resize(
            gray, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA
        )

    def process(self, rgba: np.ndarray) -> GuideFrame:
        cv2 = self._cv2
        current = self._small_gray(rgba)
        if self.previous_gray is None:
            # The first frame has nothing to reuse: reset and zero motion.
            motion = self.zero_motion
            reset = True
            scene_score = 1.0
        else:
            scene_score = float(np.mean(cv2.absdiff(current, self.previous_gray))) / 255.0
            reset = scene_score > SCENE_CUT_SCORE
            if reset:
                motion = self.zero_motion
            else:
                # ⚠️ Current -> previous: the worker asks "where did this pixel
                # come from", which is the previous position minus this one.
                motion = self.dis.calc(current, self.previous_gray, None)
                motion = cv2.resize(
                    motion, (self.width, self.height), interpolation=cv2.INTER_LINEAR
                )
                motion[..., 0] *= self.width / self.flow_width
                motion[..., 1] *= self.height / self.flow_height
                motion = np.ascontiguousarray(motion.astype(np.float16))
        self.previous_gray = current
        return GuideFrame(motion=motion, reset=reset, scene_score=scene_score)
