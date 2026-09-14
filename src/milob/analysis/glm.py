# analysis/glm.py
from .base import BaseAnalysis
from ..processing.fitting import run_parallel_fits
import numpy as np
import xarray as xr
import statsmodels.api as sm
from scipy.stats import t
from scipy.linalg import pinv, solve_triangular
from scipy.signal import lfilter
from sklearn.decomposition import PCA
from typing import Optional, List, Union, Dict, Any


def _lagmatrix(y, pmax):
    """Return a matrix whose column j is y lagged by j+1 samples, zero-padded."""
    n = len(y)
    cols = np.zeros((n, pmax))
    for lag in range(1, pmax + 1):
        cols[lag:, lag - 1] = y[:n - lag]
    return cols


def _ar_fit(y, pmax):
    """
    Fit an autoregressive model of order at most ``pmax`` by BIC search.

    Parameters
    ----------
    y : np.ndarray
        Series to fit.
    pmax : int
        Highest order considered.

    Returns
    -------
    np.ndarray
        Coefficients, with an intercept first and the selected lag
        coefficients after it.
    """
    n = len(y)
    pmax = max(min(pmax, n - 1), 1)
    lags_f = _lagmatrix(y, pmax)
    y_b = y[::-1]
    lags_b = _lagmatrix(y_b, pmax)

    X = np.column_stack([np.ones(2 * n), np.vstack([lags_f, lags_b])])
    yy = np.concatenate([y, y_b])

    Q, R = np.linalg.qr(X, mode='reduced')
    n_obs = X.shape[0]
    n_params = X.shape[1]

    bics = np.full(n_params, np.inf)
    for i in range(1, n_params + 1):
        b = solve_triangular(R[:i, :i], Q[:, :i].T @ yy)
        r = yy - X[:, :i] @ b
        mse = np.mean(r ** 2)
        if mse <= 0:
            continue
        ll = -n_obs / 2 * np.log(2 * np.pi * mse) - n_obs / 2
        bics[i - 1] = -2 * ll + i * np.log(n_obs)

    best_i = int(np.argmin(bics)) + 1
    coef = solve_triangular(R[:best_i, :best_i], Q[:, :best_i].T @ yy)
    return coef


def _ar_whiten_filter(y, X, pmax):
    """Return the AR whitening filter fitted to the residual of a given fit."""
    coef = _ar_fit(y, pmax)
    ar_order = len(coef) - 1
    if ar_order == 0:
        return np.array([1.0]), 0
    return np.r_[1.0, -coef[1:]], ar_order


def _my_filter(f, y):
    """Filter a series, zeroing the first sample and restoring the DC offset."""
    y1 = y[0, ...]
    out = lfilter(f, [1.0], y - y1, axis=0)
    return out + np.sum(f) * y1


def _ar_irls_fit(y, X, pmax, tune=4.685, maxiter=10):
    """
    Fit a model by autoregressive iteratively reweighted least squares.

    Alternates between whitening the residual of the current fit and a robust
    Tukey-biweight refit on the whitened data, until convergence. Follows the
    implementation in the NIRS Brain AnalyzIR toolbox.

    Parameters
    ----------
    y : np.ndarray
        Observations.
    X : np.ndarray
        Design matrix.
    pmax : int
        Highest autoregressive order considered.
    tune : float
        Tukey biweight tuning constant. Default 4.685.
    maxiter : int
        Maximum iterations. Default 10.

    Returns
    -------
    statsmodels results object
        The final robust fit on the whitened data.

    References
    ----------
    Barker, J. W., Aarabi, A., & Huppert, T. J. (2013). Autoregressive model
    based algorithm for correcting motion and serially correlated errors in
    fNIRS. Biomedical Optics Express, 4(8), 1366-1379.
    """
    B = np.linalg.lstsq(X, y, rcond=None)[0]
    B0 = np.full_like(B, 1e6)
    results = None
    ar_order = 0
    n_iter = 0
    while np.linalg.norm(B - B0) / np.linalg.norm(B0) > 1e-2 and n_iter < maxiter:
        B0 = B
        res = y - X @ B
        f, ar_order = _ar_whiten_filter(res, X, pmax)
        Xf = _my_filter(f, X)
        yf = _my_filter(f, y)
        results = sm.RLM(yf, Xf, M=sm.robust.norms.TukeyBiweight(c=tune)).fit()
        B = results.params
        n_iter += 1
    return results, ar_order


def _fit_channel(y_slices, X, method, max_p, tune, sc_slices, const_idx,
                 want_residuals):
    """
    Fit every slice of one channel.

    Parameters
    ----------
    y_slices : np.ndarray
        This channel's data, shape (n_times, n_types, n_4d).
    X : np.ndarray
        Normalised design matrix, shape (n_times, n_regressors), shared by
        every channel.
    method : {'ols', 'robust', 'ar-irls'}
        Estimator to use.
    max_p : int
        Autoregressive order ceiling for 'ar-irls'.
    tune : float
        Tukey biweight tuning constant.
    sc_slices : np.ndarray or None
        Nearest short-channel data appended as an extra design column, for
        ``nuisance_method='sc_nearest'``. None for every other method.
    const_idx : int or None
        Column index of the intercept, or None when there is none.
    want_residuals : bool
        Also return the per-slice residuals.

    Returns
    -------
    betas, covs, ps, residuals : tuple
        Coefficients, their covariances, the selected autoregressive order per
        slice, and the residuals or None. Skipped slices stay NaN.
    """
    n_times, n_types, n_4d = y_slices.shape
    n_regs = X.shape[1] + (1 if sc_slices is not None else 0)

    betas = np.full((n_regs, n_types, n_4d), np.nan)
    covs = np.full((n_types, n_4d, n_regs, n_regs), np.nan)
    ps = np.full((n_types, n_4d), np.nan)
    residuals = np.full((n_times, n_types, n_4d), np.nan) if want_residuals else None

    for t_idx in range(n_types):
        for b_idx in range(n_4d):
            y = y_slices[:, t_idx, b_idx]

            # Skip channels with NaNs within the signal
            if np.any(np.isnan(y)):
                continue

            # nuisance_method='sc_nearest': build this channel/type's own
            # design matrix (shared task+drift columns, plus its specific
            # nearest-good-short-channel regressor, type-matched so an HbO
            # channel gets its nearest channel's own HbO signal). Skip if
            # that type-matched signal is unusable.
            X_iter = X
            if sc_slices is not None:
                sc_signal = sc_slices[:, t_idx, b_idx]
                if np.any(np.isnan(sc_signal)):
                    continue
                sc_std = np.std(sc_signal)
                if sc_std == 0:
                    continue
                sc_signal_z = (sc_signal - np.mean(sc_signal)) / sc_std
                X_iter = np.column_stack([X, sc_signal_z])

            # *** Fitting Logic ***

            # CASE 1: OLS
            if method == 'ols':
                results = sm.OLS(y, X_iter).fit()

            # CASE 2: Robust (no pre-whitening)
            elif method == 'robust':
                results = sm.RLM(y, X_iter, M=sm.robust.norms.TukeyBiweight()).fit()

            # CASE 3: AR-IRLS (Barker, Aarabi & Huppert, 2013 -- ported from
            # nirs-toolbox's ar_irls.m: iteratively AR-whiten the residual of
            # the current fit and robust-refit on the whitened data until
            # convergence, rather than fitting the AR order jointly with the
            # regressors -- joint fitting lets a high-order AR term absorb
            # slow-varying regressors (nuisance PCs, task blocks) into the
            # "noise" model instead of attributing that variance to the
            # regressors themselves.)
            else:
                results, ar_order = _ar_irls_fit(y, X_iter, max_p, tune=tune)
                ps[t_idx, b_idx] = ar_order

            # Store estimators
            betas[:, t_idx, b_idx] = results.params
            covs[t_idx, b_idx] = results.cov_params()  # parameter covariance matrix

            if want_residuals:
                # The part of the data explained by the model is y_hat =
                # X @ beta, so residuals = y - y_hat. For SCR we keep the
                # intercept: results.params are estimated in the whitened
                # space but apply to the original scale of the regressors,
                # so residuals in the original data space always use the
                # unwhitened X.
                res_val = y - X_iter @ results.params
                if const_idx is not None:
                    res_val += X_iter[:, const_idx] * results.params[const_idx]
                residuals[:, t_idx, b_idx] = res_val

    return betas, covs, ps, residuals


class GLM(BaseAnalysis):
    #: Stream states the GLM is defined on. 'od' and 'conc' are both
    #: linear in the underlying hemodynamics, so a fit is interpretable
    #: on either (`CW_Stream.regress_nuisance` uses the OD case). 'raw'
    #: intensity is not -- Beer-Lambert is logarithmic, so betas fit on
    #: it have no physical meaning -- and neither is 'processed', which
    #: says only that *something* was done to raw intensity.
    _VALID_STATUS = ('od', 'conc')

    #: Dimension names this GLM accepts as the spatial index, in priority
    #: order. Everything downstream of the fit (contrasts, averaging,
    #: `GLMOutput`) is already written against `beta.dims` generically, so
    #: supporting a new spatial discretisation is a matter of naming it
    #: here rather than of new machinery.
    _SPATIAL_DIMS = ('channel',)

    def __init__(self, dataset): 
        """
        Build a GLM for one preprocessed stream.

        Parameters
        ----------
        dataset : Datastream
            Preprocessed stream, whose status must be 'od' or 'conc'. Task and
            nuisance regressors are configured through
            :meth:`create_task_regressors` and :meth:`create_nuisance_regressors`.
        """
        status = getattr(dataset, 'status', None)
        if status not in self._VALID_STATUS:
            raise ValueError(
                f"GLM requires a stream with status in {self._VALID_STATUS}, got "
                f"{status!r} (stream {getattr(dataset, 'name', '?')!r}). Convert "
                f"first -- e.g. .to_od() and .mbll(), or run the stream through "
                f"Session.preprocess() with a pipeline that does."
            )

        # Call parent class constructor
        super().__init__(dataset)   # This handles self.dataset = dataset

        # Fail here rather than deep inside fit(): resolving the spatial
        # dim is what every later step indexes against.
        self.spatial_dim
        
        # GLM-specific initializations
        self.task_regressors = None
        self.nuisance_regressors = None
        self.design_matrix = None
        self.X_scaled = None
        # Meta-information for labeling
        self.task_names = []
        self.nuisance_names = []
        self.regressor_names = []
        # The two type-matched extra-regressor modes (see fit()): at most
        # one is ever non-None, set by create_nuisance_regressors().
        self.nearest_sc_assignment = None
        self.sc_average_signal = None
        

    @property
    def spatial_dim(self):
        """Name of the dataset's spatial dimension."""
        for dim in self._SPATIAL_DIMS:
            if dim in self.dataset.data.dims:
                return dim
        raise ValueError(
            f"GLM needs one of {self._SPATIAL_DIMS} as a spatial dimension; "
            f"stream {getattr(self.dataset, 'name', '?')!r} has dims "
            f"{self.dataset.data.dims}."
        )

    def _get_hrf_kernel(self, kernel_type, fs):
        """
        Build the haemodynamic response kernel used for convolution.

        Parameters
        ----------
        kernel_type : {'canonical', 'gamma'}
            Shape of the response.
        fs : float
            Sampling frequency in Hz.

        Returns
        -------
        np.ndarray
            Kernel normalised to unit peak.
        """
        t = np.arange(0, 30, 1/fs)
        from scipy.stats import gamma
        if kernel_type == 'canonical':
            # Simplified double-gamma
            hrf = gamma.pdf(t, 6) - 0.16 * gamma.pdf(t, 16)
        else:   # gamma
            hrf = gamma.pdf(t, 6)
        return hrf / np.max(hrf)
    
    
    
    def create_task_regressors(self, 
                               basis: str = 'canonical',
                               fir_duration: float = 20.0,
                               source: Union[Dict, np.ndarray, None] = None):
        """
        Build the task part of the design matrix.

        Parameters
        ----------
        basis : {'canonical', 'gamma', 'fir'}
            Basis convolved with each condition's stimulus vector.
        fir_duration : float
            Seconds after onset to model, for the FIR basis. Default 20.0.
        source : dict or np.ndarray, optional
            Events to use instead of the stream's own: a mapping of condition to
            onsets and durations, or pre-computed regressors.

        Returns
        -------
        GLM
            This object, for chaining.
        """
        n_times = len(self.dataset.data.time)

        if isinstance(source, np.ndarray):
            # Case: Direct regressor input (e.g., video analysis output)
            self.task_regressors = source
            self.task_names = [f"task_{i+1}" for i in range(source.shape[1])]
            return self.task_regressors
        
        # Case: Event-based convolution. Event onsets are in SECONDS, so the
        # sampling rate is what places them on the time axis -- getting it
        # wrong silently shifts every regressor and quietly invalidates the
        # whole fit. Refuse rather than assume a default (this used to fall
        # back to 1.0 Hz, which is how a reconstructed stream that had lost
        # its 'sampling_rate' attr produced a plausible-looking but wrong
        # design matrix).
        fs = self.dataset.data.attrs.get('sampling_rate')
        if fs is None or not np.isfinite(fs) or fs <= 0:
            raise ValueError(
                f"create_task_regressors() needs a valid sampling rate in "
                f"data.attrs['sampling_rate'] to place event onsets (which are "
                f"in seconds) onto the time axis; got {fs!r} for stream "
                f"{getattr(self.dataset, 'name', '?')!r}. Set it explicitly, "
                f"e.g. stream.data.attrs['sampling_rate'] = fs."
            )
        fs = float(fs)

        events_obj = source if source is not None else self.dataset.events
                
        reg_list = []
        self.task_names = []
        
        for label in events_obj.conditions:
            # Get a dataframe for the i-th condition
            df_cond = events_obj.get_events_by_label(label)
            
            # Create a stim/onset vector (delta functions)
            stim_vec = np.zeros(n_times)
            for onset in df_cond['onset'].values:
                idx = int(np.round(onset * fs))
                if idx < n_times:
                    stim_vec[idx] = 1
            
            if basis == 'fir':
                # FIR / Deconvolution Model
                # Creates a series of shifted identity pulses
                n_lags = int(fir_duration * fs)
                for lag in range(n_lags):
                    fir_reg = np.zeros(n_times)
                    # Shift the impulse vector by 'lag'
                    if lag < n_times:
                        fir_reg[lag:] = stim_vec[:n_times - lag]
                    reg_list.append(fir_reg)
                    self.task_names.append(f"{label}_lag{lag}")
            else:
                # Convolved Models
                hrf = self._get_hrf_kernel(basis, fs)
                # If duration exists, use boxcar; otherwise, use impulses
                if (df_cond['duration'] > 0).any():
                    boxcar = np.zeros(n_times)
                    for _, row in df_cond.iterrows():
                        start = int(np.round(row['onset']*fs))
                        end = int(np.round((row['onset'] + row['duration']) * fs))
                        boxcar[start:min(end, n_times)] = 1
                    stim_vec = boxcar
                
                # Convolve and truncate to original length
                convolved = np.convolve(stim_vec, hrf)[:n_times]
                reg_list.append(convolved)
                self.task_names.append(label)

        if not reg_list:
            raise ValueError(
                "create_task_regressors() found no event conditions to build "
                "regressors from (self.dataset.events is empty or 'source' had "
                "no conditions). If you don't want task regressors at all -- "
                "e.g. a nuisance-only fit -- call "
                "create_design_matrix(include_tasks=False, ...) explicitly "
                "instead of relying on fit()'s auto-build."
            )

        self.task_regressors = np.column_stack(reg_list)
        return self.task_regressors
    
    
    _DRIFT_OPTIONS = ('none', 'intercept', 'intercept+trend')
    _NUISANCE_METHODS = ('none', 'sc_average', 'sc_pca', 'sc_nearest',
                         'global_pca', 'global_average')
    _FIT_METHODS = ('ols', 'robust', 'ar-irls')

    def _pca_from_pool(self, data, mask, n_times, n_components, label):
        """
        Extract principal components from a pool of channels.

        Parameters
        ----------
        data : np.ndarray
            Stream data to draw from.
        mask : np.ndarray of bool
            Which channels form the pool.
        n_times : int
            Number of time points.
        n_components : int or float
            Components to keep, or the variance fraction to explain.
        label : str
            Name used when recording the result.

        Returns
        -------
        tuple of (np.ndarray, dict) or (None, None)
            Components and their metadata, or None when the pool has no usable
            data.
        """
        pool_data = data.sel(channel=mask).values
        # Flatten (Time, Channels, Types) -> (Time, Features)
        pool_data = pool_data.reshape(n_times, -1)

        # Remove columns that are entirely NaN
        pool_data = pool_data[:, ~np.all(np.isnan(pool_data), axis=0)]
        if pool_data.shape[1] == 0:
            print(f"Warning: all good {label} channels contain only NaN data; skipping PCA.")
            return None, None

        # Fill remaining NaNs with column mean; fall back to 0 if needed
        col_means = np.nanmean(pool_data, axis=0)
        inds = np.where(np.isnan(pool_data))
        pool_data[inds] = np.take(col_means, inds[1])
        pool_data = np.nan_to_num(pool_data)

        # Standardize (PCA usually performs better with scaled data)
        from sklearn.preprocessing import StandardScaler
        pool_data = StandardScaler().fit_transform(pool_data)

        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(pool_data)
        info = {
            "n_components": pca.n_components_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "total_variance_explained": pca.explained_variance_ratio_.sum(),
        }
        return pcs, info

    def create_nuisance_regressors(self,
                                   nuisance_method: str = 'sc_pca',
                                   n_components: Union[int, float] = 0.8,
                                   add_drift: str = 'intercept',
                                   external_regressors: Optional[np.ndarray] = None):
        """
        Build the nuisance part of the design matrix.

        Parameters
        ----------
        nuisance_method : {'none', 'sc_average', 'sc_pca', 'sc_nearest', 'global_pca', 'global_average'}
            Source of the nuisance regressors, chosen once for the whole fit.
            'sc_average' shares the mean of the good short channels, matched to
            the slice being fitted. 'sc_pca' (default) shares principal components
            of the pooled short channels. 'sc_nearest' gives each long channel the
            nearest good short channel. 'global_pca' and 'global_average' draw
            from the long channels instead, for probes with no short separation;
            a long channel is not guaranteed to be neurally silent, so a
            widespread response can be partly regressed away.
        n_components : int or float
            Components kept for 'sc_pca' and 'global_pca'. An integer keeps that
            many; a float in (0, 1) keeps enough to explain that fraction of the
            variance. Default 0.8.
        add_drift : {'none', 'intercept', 'intercept+trend'}
            Drift terms to include. Default 'intercept'.
        external_regressors : np.ndarray, optional
            Further regressors, such as heart rate or respiration.

        Returns
        -------
        GLM
            This object, for chaining.

        References
        ----------
        Gregg, N. M. et al. (2010). Frontiers in Neuroenergetics, 2, 14.
        Zhang, Y. et al. (2005). Journal of Biomedical Optics, 10(1), 011014.
        Novi, S. L. et al. (2020). Neurophotonics, 7(1), 015001.
        """
        if add_drift not in self._DRIFT_OPTIONS:
            raise ValueError(
                f"add_drift must be one of {self._DRIFT_OPTIONS}, got {add_drift!r}"
            )
        if nuisance_method not in self._NUISANCE_METHODS:
            raise ValueError(
                f"nuisance_method must be one of {self._NUISANCE_METHODS}, got {nuisance_method!r}"
            )

        n_times = len(self.dataset.data.time)
        regs = []
        self.nuisance_names = []
        self.sc_pca_info = None
        self.nuisance_method = nuisance_method
        self.add_drift = add_drift
        self.n_components = n_components
        self.nearest_sc_assignment = None
        self.sc_average_signal = None

        # Intercept and Drift
        if add_drift in ('intercept', 'intercept+trend'):
            regs.append(np.ones(n_times))
            self.nuisance_names.append("constant")
        if add_drift == 'intercept+trend':
            regs.append(np.linspace(-1, 1, n_times))
            self.nuisance_names.append("linear_drift")

        data = self.dataset.data
        has_masks = 'is_short' in data.coords and 'is_bad' in data.coords

        if nuisance_method == 'none':
            pass

        elif nuisance_method in ('sc_average', 'sc_pca', 'sc_nearest'):
            if not has_masks:
                print("Warning: stream missing is_short/is_bad coords; cannot compute short-channel regressors.")
            else:
                sc_mask = data.coords['is_short'] & ~data.coords['is_bad']

                if not sc_mask.any():
                    print("Warning: no good short channels available.")
                elif nuisance_method == 'sc_average':
                    # Kept on the type-matched per-slice path (fit()'s
                    # sc_slices) rather than added to the shared design
                    # matrix X, because the average is per (wavelength |
                    # chromophore): one column of X cannot hold a
                    # different signal for each slice. Same mechanism as
                    # 'sc_nearest', but one shared signal for every long
                    # channel instead of a per-channel pick.
                    vals = data.values
                    vals_4d = vals if vals.ndim == 4 else vals[..., None]
                    sc_block = vals_4d[:, sc_mask.values]  # (time, n_sc, type, 4d)
                    with np.errstate(invalid='ignore'):
                        avg = np.nanmean(sc_block, axis=1)  # (time, type, 4d)
                    if np.all(np.isnan(avg)):
                        print("Warning: all good short channels contain only NaN data; "
                              "skipping short-channel average.")
                    else:
                        self.sc_average_signal = avg
                        self.nuisance_names.append("SC_avg")
                elif nuisance_method == 'sc_pca':
                    pcs, info = self._pca_from_pool(data, sc_mask, n_times, n_components, label='short')
                    if pcs is not None:
                        self.sc_pca_info = info
                        regs.append(pcs)
                        self.nuisance_names.extend([f"SC_PC{i+1}" for i in range(pcs.shape[1])])
                else:  # sc_nearest
                    probe = getattr(self.dataset, 'probe', None)
                    if probe is None:
                        print("Warning: stream has no Probe; cannot compute channel distances for nuisance_method='sc_nearest'.")
                    else:
                        all_labels = list(data.channel.values)
                        sc_indices = np.where(sc_mask.values)[0]
                        dist_matrix = probe.channel_distance_matrix()
                        # Reindex the probe's distance matrix (covers all probe
                        # channels) onto this dataset's channel order/subset.
                        probe_labels = list(probe.channel_labels)
                        label_to_probe_idx = {lbl: i for i, lbl in enumerate(probe_labels)}

                        n_chans = len(all_labels)
                        assignment = np.full(n_chans, -1, dtype=int)
                        for c, label in enumerate(all_labels):
                            if sc_mask.values[c] or label not in label_to_probe_idx:
                                continue  # short channels themselves are never modelled
                            p_c = label_to_probe_idx[label]
                            candidates = [
                                (dist_matrix[p_c, label_to_probe_idx[all_labels[sc]]], sc)
                                for sc in sc_indices
                                if all_labels[sc] in label_to_probe_idx
                            ]
                            if candidates:
                                candidates.sort(key=lambda pair: pair[0])
                                assignment[c] = candidates[0][1]

                        self.nearest_sc_assignment = assignment
                        # Placeholder name: the actual channel behind this slot
                        # varies per long channel -- see nearest_sc_assignment
                        # for the concrete per-channel mapping.
                        self.nuisance_names.append("SC_nearest")

        else:  # global_pca, global_average
            if not has_masks:
                print("Warning: stream missing is_short/is_bad coords; cannot compute global regressors.")
            else:
                global_mask = ~data.coords['is_short'] & ~data.coords['is_bad']

                if not global_mask.any():
                    print("Warning: no good long channels available for global regression.")
                elif nuisance_method == 'global_pca':
                    pcs, info = self._pca_from_pool(data, global_mask, n_times, n_components, label='long')
                    if pcs is not None:
                        self.sc_pca_info = info
                        regs.append(pcs)
                        self.nuisance_names.extend([f"Global_PC{i+1}" for i in range(pcs.shape[1])])
                else:  # global_average
                    pool_data = data.sel(channel=global_mask).values.reshape(n_times, -1)
                    pool_data = pool_data[:, ~np.all(np.isnan(pool_data), axis=0)]
                    if pool_data.shape[1] == 0:
                        print("Warning: all good long channels contain only NaN data; skipping global average.")
                    else:
                        col_means = np.nanmean(pool_data, axis=0)
                        inds = np.where(np.isnan(pool_data))
                        pool_data[inds] = np.take(col_means, inds[1])
                        pool_data = np.nan_to_num(pool_data)

                        from sklearn.preprocessing import StandardScaler
                        pool_data = StandardScaler().fit_transform(pool_data)

                        regs.append(pool_data.mean(axis=1))
                        self.nuisance_names.append("Global_avg")

        self.nuisance_regressors = np.column_stack(regs)
        return self.nuisance_regressors
        
    
    def create_design_matrix(self, include_tasks=True, include_nuisance=True):
        """
        Assemble the full design matrix.

        Parameters
        ----------
        include_tasks : bool
            Include the task regressors. Default True.
        include_nuisance : bool
            Include the nuisance regressors. Default True.

        Returns
        -------
        np.ndarray
            Design matrix of shape (n_times, n_regressors).
        """
        regs = []
        self.regressor_names = []
        
        # Only create/add tasks if requested
        if include_tasks:
            if self.task_regressors is None: 
                self.create_task_regressors()
            regs.append(self.task_regressors)
            self.regressor_names.extend(self.task_names)
        
        # Only create/add nuisance if requested
        if include_nuisance:
            if self.nuisance_regressors is None: 
                self.create_nuisance_regressors()
            regs.append(self.nuisance_regressors)
            self.regressor_names.extend(self.nuisance_names)
            
        if not regs:
            raise ValueError("Design matrix is empty. Select tasks, nuisance, or both.")
        
        self.design_matrix = np.column_stack(regs)
        return self.design_matrix
        
        
        
    def fit(self, 
            method: str = 'ar-irls',
            max_ar_order: Optional[int] = None,
            return_residuals=False,
            n_jobs: int = 1,
            **kwargs: Any):
        """
        Fit the design matrix to every channel and chromophore.

        Parameters
        ----------
        method : {'ols', 'robust', 'ar-irls'}
            'ols' is ordinary least squares, 'robust' is iteratively reweighted
            least squares without pre-whitening, and 'ar-irls' (default) adds
            autoregressive pre-whitening.
        max_ar_order : int, optional
            Ceiling on the autoregressive order search for 'ar-irls'. The order
            used is chosen per slice by BIC. Defaults to four times the sampling
            rate.
        return_residuals : bool
            Also return the per-channel residuals. Default False.
        n_jobs : int
            Parallel jobs across channels. Default 1, meaning serial. Worth using
            for 'ar-irls' on many channels.

        Returns
        -------
        GLMOutput
            Coefficients, their covariances and the selected autoregressive order
            per slice.
        """
        if method not in self._FIT_METHODS:
            raise ValueError(f"method must be one of {self._FIT_METHODS}, got {method!r}")
        self.fit_method = method
        self.max_ar_order = max_ar_order

        # Data shape management
        data_obj = self.dataset.data
        is_4d = len(data_obj.dims) == 4
        values = data_obj.values
        n_times, n_chans, n_types = values.shape[:3]
        n_4d = values.shape[3] if is_4d else 1
        
        # Design Matrix preparation
        
        # If the a matrix hasn't been built yet, assume a standard Full GLM.
        # (but for SCR, matrix must be built manually before calling fit).
        if self.design_matrix is None:
            self.create_design_matrix(include_tasks=True, include_nuisance=True)
            
        # Normalize design matrix before fitting
        if self.X_scaled is None:
            self.X_scaled = self._normalize_design_matrix(self.design_matrix)
        
        X = self.X_scaled
        # Two nuisance methods append a type-matched column that X cannot
        # carry (it differs per wavelength/chromophore slice, and for
        # 'sc_nearest' per channel too) -- see create_nuisance_regressors.
        nearest_mode = getattr(self, 'nearest_sc_assignment', None) is not None
        sc_avg_signal = getattr(self, 'sc_average_signal', None)
        n_regs = X.shape[1] + 1 if (nearest_mode or sc_avg_signal is not None) else X.shape[1]
        fs = self.dataset.data.attrs.get('sampling_rate', 1.0)
            
        # Initialize core estimators
        if is_4d:
            betas = np.full((n_regs, n_chans, n_types, n_4d), np.nan)
            covs = np.full((n_chans, n_types, n_4d, n_regs, n_regs), np.nan)
        else:
            betas = np.full((n_regs, n_chans, n_types), np.nan)
            covs = np.full((n_chans, n_types, n_regs, n_regs), np.nan)
        
        # One AR order per fitted slice. The 4D case carries the 4th-dim axis
        # too -- each (type, 4th-dim) slice runs its own BIC order search, and
        # `_package_results` declares ar_order with that axis.
        optimal_ps = (np.full((n_chans, n_types, n_4d), np.nan) if is_4d
                      else np.full((n_chans, n_types), np.nan))
        
        # Initialize residuals array if needed
        if return_residuals:
            res_shape = (n_times, n_chans, n_types, n_4d) if is_4d else (n_times, n_chans, n_types)
            residuals = np.full(res_shape, np.nan)
            
        # Pre-calculate masks. A voxel stream carries neither coord -- there
        # are no bad or short voxels -- so both default to all-False and
        # every voxel is fitted.
        is_bad = self.dataset.data.coords.get('is_bad', np.zeros(n_chans, dtype=bool))
        is_short = self.dataset.data.coords.get('is_short', np.zeros(n_chans, dtype=bool))
        fit_mask = ~(is_bad | is_short)
        
        # Build one job per fittable channel. A channel's (type, 4th-dim)
        # slices stay inside its job: they share the AR/RLM setup cost and
        # keep the job count at n_chans, so process-based parallelism isn't
        # paying pickling overhead per individual slice.
        const_idx = (self.regressor_names.index("constant")
                     if "constant" in self.regressor_names else None)
        max_p = int(np.round(4 * fs)) if max_ar_order is None else max_ar_order
        tune = kwargs.get('tune', 4.685)

        # Treat 3D as 4D-with-one-slice so the per-channel worker has a
        # single shape to handle; this is a view, not a copy.
        values_4d = values if is_4d else values[..., None]

        jobs = []
        job_channels = []
        for c in range(n_chans):
            # Skip if the whole channel is marked bad or short
            if not fit_mask[c]:
                continue

            sc_slices = None
            if nearest_mode:
                sc_idx = self.nearest_sc_assignment[c]
                # No good short channel available for this channel
                if sc_idx == -1:
                    continue
                sc_slices = np.ascontiguousarray(values_4d[:, sc_idx])
            elif sc_avg_signal is not None:
                # One shared signal for every long channel -- still passed
                # per job so _fit_channel's single type-matched code path
                # serves both modes unchanged.
                sc_slices = sc_avg_signal

            jobs.append((_fit_channel,
                         (np.ascontiguousarray(values_4d[:, c]), X, method,
                          max_p, tune, sc_slices, const_idx, return_residuals)))
            job_channels.append(c)

        results = run_parallel_fits(jobs, n_jobs=n_jobs)

        # Scatter each channel's block back into the pre-allocated arrays
        for c, (betas_c, covs_c, ps_c, res_c) in zip(job_channels, results):
            if is_4d:
                betas[:, c, :, :] = betas_c
                covs[c, :, :, :, :] = covs_c
                optimal_ps[c, :, :] = ps_c
                if return_residuals:
                    residuals[:, c, :, :] = res_c
            else:
                betas[:, c, :] = betas_c[..., 0]
                covs[c, :, :, :] = covs_c[:, 0]
                optimal_ps[c, :] = ps_c[:, 0]
                if return_residuals:
                    residuals[:, c, :] = res_c[..., 0]

        if return_residuals:
            return residuals
        
        # Package into the Output container
        return self._package_results(betas, covs, optimal_ps)
    

    def _normalize_design_matrix(self, X):
        """Check and rescale the design matrix so its columns are comparable."""
        X_scaled = X.copy()
        
        # Identify columns that are NOT the constant intercept
        # We find them by looking for columns with a variance > 0
        for i in range(X_scaled.shape[1]):
            col = X_scaled[:, i]
            col_std = np.std(col)
            
            if col_std == 0:
                # This is a constant (intercept). Do not touch.
                continue
            
            # Is this a Task or a Nuisance?
            # We can use the names we stored earlier
            name = self.regressor_names[i]
            
            if name in self.task_names:
                # For Tasks: We often scale by the peak to keep the beta 
                # in 'Hb Concentration units'.
                peak = np.max(np.abs(col))
                X_scaled[:, i] = col / peak
            else:
                # For Nuisance: Z-score (Mean 0, Std 1) is best practice.
                X_scaled[:, i] = (col - np.mean(col)) / col_std
                
        return X_scaled
    
    

    def _package_results(self, betas, covariances, optimal_ps):
        """Wrap the fitted coefficients and covariances into a GLMOutput."""
        # Determine chromophore vs wavelength in dimension name
        # Payload axis, in priority order: 'chromophore' (CW_Stream after
        # mbll), 'component' (TissueStream, channel- or voxel-indexed),
        # else 'wavelength' (an un-unmixed stream). The output keeps the
        # input's own name rather than renaming it -- a TissueStream can
        # carry scattering parameters ('A', 'b') on `component`, which are
        # not chromophores, so relabelling would assert something false.
        type_dim = next(
            (d for d in ('chromophore', 'component') if d in self.dataset.data.dims),
            'wavelength',
        )
        space_dim = self.spatial_dim
        is_4d = (betas.ndim == 4)
        
        # Define coordinates for the square covariance matrix
        coords = {
            'regressor': self.regressor_names,
            'regressor_i': self.regressor_names,
            'regressor_j': self.regressor_names,
            space_dim: self.dataset.data.coords[space_dim].values,
            type_dim: self.dataset.data.coords[type_dim].values
        }
        
        # Pack into Dataset
        if is_4d:
            dim4_name = self.dataset.data.dims[3]
            coords[dim4_name] = self.dataset.data.coords[dim4_name].values
            
            data_vars = {
               'beta': (['regressor', space_dim, type_dim, dim4_name], betas), 
               'covariance': ([space_dim, type_dim, dim4_name, 'regressor_i', 'regressor_j'], covariances),
               'ar_order': ([space_dim, type_dim, dim4_name], optimal_ps)
            }
        else:
            data_vars = {
                'beta': (['regressor', space_dim, type_dim], betas),
                'covariance': ([space_dim, type_dim, 'regressor_i', 'regressor_j'], covariances),
                'ar_order': ([space_dim, type_dim], optimal_ps)
            } 
        
        res_xr = xr.Dataset(data_vars, coords=coords)
        
        # Store metadata for future inference
        res_xr.attrs['df'] = self.dataset.data.time.size - len(self.regressor_names)
        res_xr.attrs['analysis_type'] = 'GLM_Estimators'

        from ..outputs.output_glm import GLMOutput
        from .. import __version__
        fit_entry = {
            'operation': 'GLM.fit',
            'params': {
                'method': getattr(self, 'fit_method', None),
                'max_ar_order': getattr(self, 'max_ar_order', None),
                'task_names': list(self.task_names),
                'nuisance_method': getattr(self, 'nuisance_method', None),
                'add_drift': getattr(self, 'add_drift', None),
                'n_components': getattr(self, 'n_components', None),
                'source_stream_name': self.dataset.name,
            },
            'version': __version__,
        }
        history = list(self.dataset.history) + [fit_entry]
        return GLMOutput(data=res_xr, probe=self.dataset.probe, analysis_type='GLM',
                         history=history,
                         )
