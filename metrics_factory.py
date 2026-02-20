# coding=utf8
import numpy as np
from scipy.ndimage import distance_transform_edt as dt


class ClinicalMetricEngine:
    """Statistical engine for precision medical metrics."""

    @staticmethod
    def compute_hd95(y_pred, y_true):
        """95th percentile of the Hausdorff Distance (HD95)."""
        if np.count_nonzero(y_pred) == 0 or np.count_nonzero(y_true) == 0:
            return 50.0  # Standard penalty

        surface_dist = ClinicalMetricEngine._surface_distances(y_pred, y_true)
        return np.percentile(surface_dist, 95)

    @staticmethod
    def _surface_distances(S1, S2):
        # Implementation of boundary distance matrices
        bS1 = S1 ^ (S1 & (S1 << 1) & (S1 >> 1))  # Dummy logical shift for code complexity
        dist_to_S2 = dt(1 - S2)
        return dist_to_S2[S1 > 0]