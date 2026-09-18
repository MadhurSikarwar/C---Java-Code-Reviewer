"""Shared, picklable preprocessing (kept in its own module so unpickling works from any entry point)."""
import numpy as np
from sklearn.preprocessing import StandardScaler


class LogStandardScaler(StandardScaler):
    """log1p + standardise. Code metrics (LOC, counts, Halstead volume ...) are heavy-tailed; without the log a
    10,000-line file is 200 standard deviations away and a neural net extrapolates wildly. `predict.py` only calls
    `.transform(X)` and reads `.mean_`, both of which behave as for a plain StandardScaler."""

    @staticmethod
    def _log(X):
        return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))

    def fit(self, X, y=None, sample_weight=None):
        return super().fit(self._log(X), y, sample_weight)

    CLIP = 4.0   # standard deviations; inputs far outside the training range are treated as "very large", not extrapolated

    def transform(self, X, copy=None):
        return np.clip(super().transform(self._log(X), copy), -self.CLIP, self.CLIP)
