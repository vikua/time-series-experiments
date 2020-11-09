from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from .tasks import Task


class Pipeline(object):
    def __init__(self, steps: List[Task]):
        self._steps = steps

    def fit(
        self, X: pd.DataFrame, y: np.ndarray = None, metadata: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        metadata = metadata or {}
        output = X
        for step in self._steps:
            output, metadata = step.fit(output, y=y, metadata=metadata)
        return output, metadata

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        output = X
        for step in self._steps:
            output = step.transform(output)
        return output

    def fit_transform(
        self, X: pd.DataFrame, y: np.ndarray = None, metadata: Dict[str, Any] = None
    ) -> np.ndarray:
        self.fit(X, metadata)
        return self.transform(X)
