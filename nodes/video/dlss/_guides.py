"""Where the cuts are — the only thing a video batch still has to tell the engine.

Until the upstream v9 runtime the caller also estimated motion vectors with DIS
optical flow and sent them with every frame. The Neuroframe Engine estimates
motion itself, on the GPU (NVIDIA optical flow, with a bundled Lucas-Kanade
fallback), so all that is left here is the question the pictures answer and the
engine cannot: is this frame the continuation of the last one, or a new shot?

⚠️ Getting that wrong is visible. A missed cut drags the previous shot's detail
into the first frames of the new one; a cut declared on every frame throws away
the temporal detail the whole mode exists for. The threshold is the reference
application's own (``src/video/guides.py``), measured on real footage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..._deps import TSDependencyManager

#: Above this mean absolute difference the frames are a cut, not motion.
SCENE_CUT_SCORE = 0.24


@dataclass
class GuideFrame:
    """How this frame relates to the last one."""

    reset: bool
    scene_score: float


class TemporalGuideGenerator:
    """Compare consecutive frames and say where the temporal history must restart."""

    def __init__(self, width: int, height: int, compare_width: int = 640) -> None:
        cv2 = TSDependencyManager.import_optional("cv2")
        if cv2 is None:
            raise RuntimeError(
                "[TS DLSS Upscaler] opencv-python-headless is required for the temporal "
                "path. Install it, or switch 'temporal' off."
            )
        self._cv2 = cv2
        self.width = width
        self.height = height
        scale = min(1.0, compare_width / width)
        # Even sizes and never below 64, the same grid the flow solver used.
        self.compare_width = max(64, int(round(width * scale / 2) * 2))
        self.compare_height = max(64, int(round(height * scale / 2) * 2))
        self.previous_gray: np.ndarray | None = None

    def _small_gray(self, rgba: np.ndarray) -> np.ndarray:
        cv2 = self._cv2
        gray = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
        return cv2.resize(
            gray, (self.compare_width, self.compare_height), interpolation=cv2.INTER_AREA
        )

    def process(self, rgba: np.ndarray) -> GuideFrame:
        cv2 = self._cv2
        current = self._small_gray(rgba)
        if self.previous_gray is None:
            # The first frame has nothing to reuse.
            reset, scene_score = True, 1.0
        else:
            scene_score = float(np.mean(cv2.absdiff(current, self.previous_gray))) / 255.0
            reset = scene_score > SCENE_CUT_SCORE
        self.previous_gray = current
        return GuideFrame(reset=reset, scene_score=scene_score)
