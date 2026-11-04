from abc import ABC, abstractmethod
import numpy as np
from privacy_vision.schemas import Detection

class BaseDetector(ABC):
    source: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Return detections in original BGR image coordinates. Errors propagate."""
