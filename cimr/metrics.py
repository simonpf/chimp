"""
cimr.metrics
============

Functionality to evaluate retrieval results.
"""
from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.fft import dctn
import xarray as xr

class Metric(ABC):
    """
    The Metric abstract base class defines the general interface
    for the on-the-fly calculation of metrics.

    Every metric must provide the 'calc' function, which updates the
    metric with the results from the current retrieval step. This function
    should be called for every step of retrieval results.

    Finally, the metric values for all retrieval targets can be extracted
    using the 'results' member function, which should return an
    'xarray.Dataset' containing the metric values.
    """
    @abstractmethod
    def calc(self, y_pred, y_true):
        """
        Calculate metric and accumulate results.

        Args:
            y_pred: A numpy.ndarray, or a (potentially nested) list or
                or dict of numpy.ndarray containing the retrieval estimates.
            y_true: A numpy.ndarray, or a (potentially nested) list or
                or dict of numpy.ndarray containing the reference values.
        """

    @abstractmethod
    def results(self):
        """
        Return:
            An xarray.Dataset containing the mean of the statistics,
            calculated over all results provided to the calc function.
        """

def initialize_results(shape, results):
    """
    Initialize results to zero.
    """
    if isinstance(results, list):
        return [initialize_results(shape, res) for res in results]
    elif isinstance(results, dict):
        return {
            key: initialize_results(shape, res) for key, res in results.items()
        }
    if isinstance(shape, dict):
        return {
            key: np.zeros(*value) for key, value in shape.items()
        }
    return np.zeros(shape)


class MetricBase(Metric):
    """
    Base class for metrics that handles the initialization of the internal
    state and the metric update for multiple retrieval targets.
    Implements a generic 'calc' function, which delegates the actual
    calculation of the metric to the 'accumulate' method. The 'accumulate'
    method, just needs to update the internal state of the metrics for a
    single retrieval target.
    """
    def __init__(self):
        self._results = None

    @abstractmethod
    def accumulate(self, y_pred, y_true, results):
        """
        Update metric values for a single retrieval target. This method
        is called for every target that is retrieved with the corresponding
        array holding the internal state of metric calculation.

        Args:
            y_pred: The retrieval estimates for the given target.
            y_true: The true values of the given target.
            results: The numpy array holding the internal state for
                the given retrieval target.
        """
        pass

    def calc(self, y_pred, y_true, results=None):
        """
        Generic calc method that handles the processing of results from
        multiple retrieval targets by delegating the calculation for
        each single target to the 'accumulate' member function.

        """
        if self._results is None:
            self._results = initialize_results(self.shape, y_pred)

        if results is None:
            return self.calc(y_pred, y_true, results=self._results)

        if isinstance(y_pred, list):
            return [
                self.calc(y_p, y_t, results=res) for
                y_p, y_t, res in zip(y_pred, y_true, results)
            ]

        if isinstance(y_pred, dict):
            all = {}
            for key in y_pred:
                y_pred_k = y_pred[key]
                y_true_k = y_true[key]
                res_k = results[key]
                all[key] = self.calc(y_pred_k, y_true_k, res_k)
            return all

        return self.accumulate(y_pred, y_true, results)

    def _merge_rec(self, results, other):
        """
        Recursive merging of metric statistic.

        Args:
            results: The result dict or array of this metric object.
            other: The results dict or array of another metric object.
        """
        if isinstance(other, dict):
            for key in other.keys():
                if results[key] is None:
                    results[key] = other[key]
                else:
                    self._merge_rec(results[key], other[key])
            return None
        results += other
        return None

    def merge(self, other):
        """
        Accumulate running metric statistics.

        Args:
            other: Another metric object whose running statistics to integrate
                into the running statistics of this object.
        """
        if self._results is None and other._results is not None:
            self._results = other._results
        self._merge_rec(self._results, other._results)




class Bias(MetricBase):
    """
    Calculates the bias and relative bias.
    """
    def __init__(self):
        super().__init__()
        self.shape = (3,)

    def accumulate(self, y_pred, y_true, results):
        valid = np.isfinite(y_pred) * np.isfinite(y_true)
        results[0] += (y_pred[valid] - y_true[valid]).sum()
        results[1] += y_true[valid].sum()
        results[2] += valid.sum()

    def results(self):
        if isinstance(self._results, dict):
            biases = {}
            for key, results in self._results.items():
                bias = results[0] / results[2]
                biases[key + "_bias"] = bias
                rel_bias = results[0] / results[1]
                biases[key + "_rel_bias"] = rel_bias
            return xr.Dataset(biases)

        return xr.Dataset({
            "bias": self._results[0] / self._results[2],
            "rel_bias": self._results[0] / self._results[1],
        })


class MSE(MetricBase):
    """
    Calculates the mean squared error (MSE).
    """
    def __init__(self):
        super().__init__()
        self.shape = (4,)

    def accumulate(self, y_pred, y_true, results):
        valid = np.isfinite(y_pred) * np.isfinite(y_true)
        results[0] += ((y_pred[valid] - y_true[valid]) ** 2).sum()
        results[1] += y_true[valid].sum()
        results[2] += valid.sum()
        results[3] += (y_true[valid] ** 2).sum()

    def results(self):
        if isinstance(self._results, dict):
            mses = {}
            for key, results in self._results.items():
                mse = results[0] / results[2]
                mses[key + "_mse"] = mse
                sigma = results[3] / results[2] - (results[1] / results[2]) ** 2
                rel_mse = results[0] / results[2] / sigma
                mses[key + "_rel_mse"] = rel_mse
            return xr.Dataset(mses)

        return xr.Dataset({
            "mse": self._results[0] / self._results[2],
            "rel_mse": (
                self._results[0] / self._results[2] /
                (self._results[1] / self.results[2]) ** 2
            ),
        })


class Correlation(MetricBase):
    """
    Calculates the correlation coefficient.
    """
    def __init__(self):
        super().__init__()
        self.shape = (6,)

    def accumulate(self, y_pred, y_true, results):
        valid = np.isfinite(y_pred) * np.isfinite(y_true)
        results[0] += y_pred[valid].sum()
        results[1] += (y_pred[valid] ** 2).sum()
        results[2] += y_true[valid].sum()
        results[3] += (y_true[valid] ** 2).sum()
        results[4] += (y_true[valid] * y_pred[valid]).sum()
        results[5] += valid.sum()

    def _calc_corr(self, x_sum, x2_sum, y_sum, y2_sum, xy_sum, n):
        x_mean = x_sum / n
        x2_mean = x2_sum / n
        x_sigma = np.sqrt(x2_mean - x_mean ** 2)
        y_mean = y_sum / n
        y2_mean = y2_sum / n
        y_sigma = np.sqrt(y2_mean - y_mean ** 2)
        xy_mean = xy_sum / n

        corr = (xy_mean - x_mean * y_mean) / (x_sigma * y_sigma)
        return corr

    def results(self):
        if isinstance(self._results, dict):
            corrs = {}
            for key, results in self._results.items():
                corr = self._calc_corr(*results)
                corrs[key + "_corr"] = corr
            return xr.Dataset(corrs)

        return xr.Dataset({
            "corr": self._calc_corr(*self._results)
        })


class PRCurve(MetricBase):
    """
    Calculates the precision-recall curve for binary classification
    retrievals.
    """
    def __init__(self, n_points=1000, thresh_multiplier=5):
        """
        Args:
            n_points: How many threshold values to test
            thresh_multiplier: This multiplicative factor is applied
                to the maximum of the first set of predictions to
                determine the upper limit of the threshold used
                for the calculation of the PR curve.
        """
        self.n_points = n_points
        self.thresh_multiplier = thresh_multiplier
        self.shape = (4, n_points)
        self._thresholds = None
        super().__init__()

    def accumulate(self, y_pred, y_true, results):

        if self._thresholds is None:
            y_max = y_pred.max()
            if np.isnan(y_max) or y_max <= 1.0:
                self._thresholds = np.linspace(0, 1, self.n_points)
            else:
                thresh_max = y_pred.max() * self.thresh_multiplier
                self._thresholds = np.linspace(0, thresh_max, self.n_points)

        valid = np.isfinite(y_pred) * np.isfinite(y_true)
        y_pred = y_pred[valid]
        y_true = y_true[valid]

        preds = y_pred >= self._thresholds[..., None]

        results[0] += (preds * y_true).sum(-1)
        results[1] += (preds * ~y_true).sum(-1)
        results[2] += y_true.sum()
        results[3] += valid.sum()


    def _calc_prec_rec(self, results):
        prec = results[0] / (results[0] + results[1])
        rec = results[0] / results[2]
        return prec, rec


    def results(self):
        if isinstance(self._results, dict):
            prec_rec = {
                "thresholds": (("thresholds",), self._thresholds)
            }
            for key, results in self._results.items():
                prec, rec = self._calc_prec_rec(results)
                prec_rec[key + "_prec"] = (("thresholds",), prec)
                prec_rec[key + "_rec"] = (("thresholds",), rec)
            return xr.Dataset(prec_rec)

        prec, rec = self._calc_prec_rec(self._results)
        return xr.Dataset({
            "thresholds": (("thresholds",), self._thresholds),
            "precision": (("thresholds",), prec),
            "recall": (("recall",), rec),
        })

    def merge(self, other):
        """
        Accumulate running metric statistics.

        Args:
            other: Another metric object whose running statistics to integrate
                into the running statistics of this object.
        """
        super().merge(other)
        if self._thresholds is None:
            self._thresholds = other._thresholds


def iterate_windows(valid, window_size):
    """
    Iterate over non-overlapping windows in which all pixels are valid.

    Args:
        valid: A 2D numpy array identifying valid pixels.
        window_size: The size of the windows.

    Return:
        An iterator providing coordinates of randomly chosen windows that
        that cover the valid pixels in the given field.
    """
    conn = np.ones((window_size, window_size))
    valid = binary_erosion(valid, conn)

    row_inds, col_inds = np.where(valid)


    while len(row_inds) > 0:


        ind = np.random.choice(len(row_inds))
        row_c = row_inds[ind]
        col_c = col_inds[ind]

        row_start = row_c - window_size // 2
        row_end = row_start + window_size

        col_start = col_c - window_size // 2
        col_end = col_start + window_size

        yield row_start, col_start, row_end, col_end

        row_lim_lower = row_start - window_size // 2
        row_lim_upper = row_end + window_size // 2
        col_lim_lower = col_start - window_size // 2
        col_lim_upper = col_end + window_size // 2

        invalid = (
            (row_inds > row_lim_lower) *
            (row_inds <= row_lim_upper) *
            (col_inds > col_lim_lower) *
            (col_inds <= col_lim_upper)
        )
        row_inds = row_inds[~invalid]
        col_inds = col_inds[~invalid]


class SpectralCoherence(MetricBase):
    """
    Metric to calculate spectral statistics of retrieved fields.

    This metrics calculates the spectral energy and coherence between
    the retrieved and reference fields.
    """
    def __init__(self, window_size=64, scale=1e4):
        self.window_size = window_size
        self.scale = scale
        self.shape = {
            "coeffs_true_sum": ((window_size,) * 2, np.cdouble),
            "coeffs_true_sum2": ((window_size,) * 2, np.cdouble),
            "coeffs_pred_sum": ((window_size,) * 2, np.cdouble),
            "coeffs_pred_sum2": ((window_size,) * 2, np.cdouble),
            "coeffs_truepred_sum": ((window_size,) * 2, np.cdouble),
            "coeffs_truepred_sum2": ((window_size,) * 2, np.cdouble),
            "coeffs_diff_sum": ((window_size,) * 2, np.cdouble),
            "coeffs_diff_sum2": ((window_size,) * 2, np.cdouble),
            "counts": ((window_size,) * 2, np.int64),
        }
        super().__init__()


    def accumulate(self, y_pred, y_true, results):
        """
        Calculate spectral statistics for all valid sample windows in
        given results.

        Args:
            y_pred: 2D np.ndarray containing the retrieved field
            y_true: 2D np.ndarray containing the reference field
            results: Container for the spectral statistics
        """
        valid = np.all(np.isfinite(np.stack([y_pred, y_true])), axis=0)
        for rect in iterate_windows(valid, self.window_size):
            row_start, col_start, row_end, col_end = rect
            y_pred_w = y_pred[row_start:row_end, col_start:col_end]
            y_true_w = y_true[row_start:row_end, col_start:col_end]
            w_pred = dctn(y_pred_w, norm="ortho")
            w_true = dctn(y_true_w, norm="ortho")
            results["coeffs_true_sum"] += w_true
            results["coeffs_true_sum2"] += w_true * w_true.conjugate()
            results["coeffs_pred_sum"] += w_pred
            results["coeffs_pred_sum2"] += w_pred * w_pred.conjugate()
            results["coeffs_truepred_sum"] += w_true * w_pred.conjugate()
            results["coeffs_diff_sum"] += w_pred - w_true
            results["coeffs_diff_sum2"] += (w_pred - w_true) * (w_pred - w_true).conjugate()
            results["counts"] += np.isfinite(w_pred)

    def _calc_coherence(self, results):
        """
        Return error statistics for correlation coefficients
        by scale.

        Args:
            results: Container containing the spectral statistics calculated
                for the given variable.

        Return:
            An 'xarray.Dataset' containing the finalized spectral statistics
            derived from the statistics collected in 'results'.
        """
        corr_coeffs = []
        coherence = []
        energy_pred = []
        energy_true = []
        mse = []

        w_true_s = results[f"coeffs_true_sum"]
        w_true_s2 = results["coeffs_true_sum2"].real
        w_pred_s = results["coeffs_pred_sum"]
        w_pred_s2 = results["coeffs_pred_sum2"].real
        w_truepred_s = results["coeffs_truepred_sum"]
        w_d_s2 = results["coeffs_diff_sum2"].real
        counts = results["counts"]

        sigma_true = w_true_s2 / counts - (w_true_s / counts) ** 2
        sigma_pred = w_pred_s2 / counts - (w_pred_s / counts) ** 2
        true_mean = w_true_s / counts
        pred_mean = w_pred_s / counts
        truepred_mean = w_truepred_s / counts
        cc = (
            (truepred_mean - true_mean * pred_mean) /
            (np.sqrt(sigma_true) * np.sqrt(sigma_pred))
        ).real
        co = np.abs(w_truepred_s) / (np.sqrt(w_true_s2) * np.sqrt(w_pred_s2))
        co = co.real

        n_y = np.arange(sigma_true.shape[0])
        n_x = np.arange(sigma_true.shape[1])
        n = np.sqrt(
            n_x.reshape(1, -1) ** 2 +
            n_y.reshape(-1, 1) ** 2
        )
        ext = min(n_y.max(), n_x.max())
        bins = np.arange(min(n_y.max(), n_x.max()) + 1) - 0.5
        counts, _ = np.histogram(n, bins)

        corr_coeffs, _ = np.histogram(n, bins=bins, weights=cc)
        corr_coeffs /= counts
        coherence, _ = np.histogram(n, bins=bins, weights=co)
        coherence /= counts
        energy_pred, _ = np.histogram(n, weights=w_pred_s2, bins=bins)
        energy_true, _ = np.histogram(n, weights=w_true_s2, bins=bins)
        se, _ = np.histogram(n, weights=w_d_s2, bins=bins)

        ns = 1 - (se / energy_true)
        mse = se / counts
        n = 0.5 * (bins[1:] + bins[:-1])
        scales = ext * self.scale / n

        inds = np.argsort(coherence[1:])

        er = 2.0 * np.interp(
            np.sqrt(1 / 2),
            coherence[1:][inds],
            scales[1:][inds]
        )

        return xr.Dataset({
            "scales": (("scales"), scales),
            "corr_coef": (("scales"), corr_coeffs.real),
            "energy_true": (("scales"), energy_true),
            "energy_pred": (("scales"), energy_pred),
            "mse": (("scales"), mse),
            "ns": (("scales"), ns),
            "coherence": (("scales"), coherence),
            "effective_resolution": er
        })

    def results(self):
        if "coeffs_true_sum" not in self._results:
            datasets = []
            for key, results in self._results.items():
                dataset = self._calc_coherence(results)
                new_names = {
                    "corr_coef": f"corr_coef_{key}",
                    "energy_true": f"energy_true_{key}",
                    "energy_pred": f"energy_pred_{key}",
                    "mse": f"mse_{key}",
                    "ns": f"ns_{key}",
                    "coherence": f"coherence_{key}",
                }
                datasets.append(dataset.rename(new_names))
            return xr.merge(datasets)

        return self._calc_coherence(self._results)
