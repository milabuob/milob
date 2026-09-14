# MILOB

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A Python library for analysing diffuse optical data — continuous-wave NIRS (CW-NIRS),
time-domain NIRS (TD-NIRS), frequency-domain NIRS (FD-NIRS) and diffuse correlation
spectroscopy (DCS) — from raw recordings through to group-level statistics.

MILOB is built around a single idea: a measurement, whatever the modality, is a forward
model of the medium composed with an observation operator of the instrument. Recovering
tissue properties is inverting that composition. One fitting engine therefore serves
FD-DOS, TD-DOS and DCS alike, and the same parameter-space containers hold the result
regardless of which instrument produced it.

---

## Features

**Modalities.** CW-NIRS, TD-NIRS (gated TPSFs and moments), FD-NIRS (complex AC/phase) and
CW-DCS (g₂ autocorrelation), plus auxiliary physiological streams (PPG, ECG, accelerometer).
Conversions between them where they are physically meaningful (TD→CW, FD→CW).

**File I/O.** SNIRF reading and writing across all four modalities; NIRx; Homer-style
`.nirs` MATLAB files; ISS OxiplexTS and ISS Imagent "BOXY"; OpenMotion SCOS. BIDS datasets
can be both read and written.

**Preprocessing.** Detrending, band-pass/high-pass/low-pass filtering, resampling, epoching
around events, and channel pruning. Motion correction by TDDR, spline interpolation (MARA)
and wavelet filtering. Signal quality via scalp coupling index (SCI), peak spectral power
(PSP) and SNR, with screening helpers that flag channels in one call.

**Optical property recovery.** Nonlinear least-squares fitting for FD-DOS, TD-DOS (full
TPSFs or moments) and DCS against any forward model, with multi-start optimisation,
parameter covariance, and optional parallel execution. Multi-distance slope fitting for
absolute µa/µs′. Joint fitting solves one shared parameter vector across several
measurement types at once — an FD-DOS and a DCS recording fitted together, for example.

**Forward models.** Semi-infinite, two-layer and general N-layer diffusion kernels, each
available as FD/CW fluence, time-domain fluence, and DCS field autocorrelation. Scatterer
dynamics as Brownian, random-flow or Langevin motion. Spectral parameterisation lets you
fit chromophore concentrations directly rather than per-wavelength absorption. Every
geometry also ships a simulator that returns a ready-to-analyse stream.

**Statistics.** A GLM with canonical HRF regressors, short-separation and PCA nuisance
regression, and three fitting methods (OLS, robust, AR-IRLS with pre-whitening). Group
inference by one-sample t-test or DerSimonian–Laird random-effects meta-analysis.
Functional connectivity by Pearson correlation, coherence, wavelet coherence and geodesic
distance, with phase-randomised surrogate nulls. Graph-theoretic network metrics on top.

**Anatomy and visualisation.** Landmark-based coregistration to MNI space, 10-20/10-10
optode labelling, and scalp/cortical surface meshes. Plotting is matplotlib throughout —
probe layouts, topographic maps, connectivity matrices and networks, time series and
spectra — with optional 3-D rendering via PyVista.

---

## Installation

MILOB is not yet on PyPI. Install it directly from GitHub:

```bash
pip install git+https://github.com/milabuob/milob.git
```

We recommend installing into a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install git+https://github.com/milabuob/milob.git
```

### Optional extras

Some features need dependencies that are not installed by default:

| Extra | Installs | Needed for |
|---|---|---|
| `threed` | PyVista | 3-D probe and surface rendering |
| `image` | NiBabel | reading NIfTI volumes |
| `brain` | Nilearn | fetching fsaverage cortical surfaces |
| `docs` | Sphinx, RTD theme | building the API documentation locally |

```bash
pip install "milob[threed,image,brain] @ git+https://github.com/milabuob/milob.git"
```

Requires Python 3.10.4 or newer.

---

## Quick start

### Simulate and fit

Every forward model has a matching simulator, so you can exercise the whole pipeline
without any data on disk:

```python
import numpy as np
import milob
from milob import forward
from milob.core.probe import Probe

# A one-source, three-detector, two-wavelength FD probe
probe = Probe(
    s_pos=np.array([[0.0, 0.0, 0.0]]),
    d_pos=np.array([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    wavelengths=np.array([690.0, 830.0]),
    channels={'sources':     [1, 1, 1, 1, 1, 1],
              'detectors':   [1, 2, 3, 1, 2, 3],
              'wavelengths': [690.0, 690.0, 690.0, 830.0, 830.0, 830.0]},
    sc_threshold=None,
    lengthUnit='cm',
)

# Simulate a frequency-domain measurement of a homogeneous medium
fd = forward.simulate_fd_stream(
    probe, mua=[0.10, 0.12], musp=[11.0, 9.0],
    modulation_freq=110e6, n=1.4,
    noise_level=0.005, rng=np.random.default_rng(0),
)

# Recover optical properties, then chromophore concentrations
op = fd.fit_to_op()                 # -> OptPropStream (mua, musp)
tissue = op.to_concentration()      # -> TissueStream  (HbO, HbR, ...)

print(op.select('mua').values.ravel()[:2])     # ~[0.094, 0.109]  (true 0.10, 0.12)
print(float(tissue.sto2().values.ravel()[0]))  # ~0.64
```

The same shape of call fits DCS:

```python
probe_dcs = Probe(
    s_pos=np.array([[0.0, 0.0, 0.0]]),
    d_pos=np.array([[2.5, 0.0, 0.0]]),
    wavelengths=np.array([785.0]),
    channels={'sources': [1], 'detectors': [1], 'wavelengths': [785.0]},
    sc_threshold=None,
    lengthUnit='cm',
)

taus = np.logspace(-7, -2, 60)
dcs = forward.simulate_dcs_stream(
    probe_dcs, mua=0.1, musp=10.0, taus=taus,
    wavelength=785.0, aDb=1e-8, beta=0.5,
    noise_level=0.01, rng=np.random.default_rng(0),
)

bfi = dcs.fit_to_bfi(fixed_params={'mua': 0.1, 'musp': 10.0, 'n': 1.4})
print(float(bfi.data.sel(op='bfi').values.ravel()[0]))   # ~9.0e-9 (true 1e-8)
```

### Load a recording

```python
from milob import Session

session = Session.from_snirf("sub-01_task-rest_nirs.snirf", sc_threshold=12.0)
stream = session.get_stream("nirs")

conc = (stream.to_od()                                    # intensity -> optical density
              .tddr()                                     # motion correction
              .frequency_filter(lowcut=0.01, highcut=0.5) # band-pass
              .mbll(dpf=6.0))                             # -> HbO / HbR
```

Every step returns a new stream and records itself in the stream's history, so `conc`
carries an account of how it was produced.

### Task analysis across subjects

Batch work is expressed as pipelines — a list of `(step_name, kwargs)` tuples applied to
every run of every subject:

```python
from milob import Study

study = Study(name="fingertapping", data_path="/path/to/bids-dataset")
study.load_bids_data(sc_threshold=12.0)

preprocess = [
    ('to_od', {}),
    ('tddr', {}),
    ('frequency_filter', {'lowcut': 0.01, 'highcut': 0.5}),
    ('mbll', {'dpf': 6.0}),
]
glm = [
    ('create_task_regressors', {'basis': 'canonical'}),
    ('create_nuisance_regressors', {'nuisance_method': 'sc_average'}),
]

results = study.run_glm(preprocess_pipeline=preprocess,
                        glm_pipeline=glm,
                        method='ar-irls')       # {subject_id: GLMOutput}

group = study.get_group_stats(method='weighted')  # random-effects meta-analysis
```

---

## Data model

MILOB organises an analysis into four nested containers, so the same code works on one
recording or on several hundred:

| Level | Class | Holds |
|---|---|---|
| Project | `Study` | A set of subjects; batch processing, group statistics, BIDS import/export |
| Subject | `Session` | One recording session; a dictionary of streams, one per modality |
| Stream | `Datastream` subclasses | One time series, as an [Xarray](https://xarray.dev) array plus its `Probe` geometry, `Events` and processing history |
| Output | `Output` subclasses | Statistical results — betas, t-statistics, connectivity matrices, network metrics |

Streams divide into **measurement** streams, which hold what the instrument recorded
(`CW_Stream`, `TD_Stream`, `FD_Stream`, `DCS_Stream`), and **parameter** streams, which hold
what was recovered from it (`OptPropStream` for µa/µs′/BFi, `TissueStream` for HbO/HbR/StO₂
and scattering). Every operation stamps an entry on the stream's history, so a result
carries a record of how it was produced.

---

## Documentation

Every public class and function carries a NumPy-style docstring, so `help(...)` and your
editor's tooltips are the quickest reference:

```python
from milob import forward
help(forward.si_fd_fluence)
```

The same docstrings build into browsable HTML with Sphinx. From a copy of the repository:

```bash
pip install sphinx sphinx-rtd-theme
cd docs
make html
```

The rendered pages appear in `docs/build/html/`.

---

## License

<!-- TODO: add a LICENSE file and name it here before making the repository public. -->
No license has been chosen yet. Until one is added, all rights are reserved and the code
carries no permission to use, copy, modify or distribute it.

---

## Citation

<!-- TODO: replace with the paper or Zenodo DOI once available. -->
If you use MILOB in published work, please cite this repository until a citable reference
is available.

---

## Background

MILOB — **M**edical **I**maging & **L**aboratório de **Ó**ptica **B**iomédica — began as a
collection of analysis routines written at the Laboratório de Óptica Biomédica, University
of Campinas, Brazil, and grew into a library at the Medical Imaging lab, University of
Birmingham, UK. This is the public release: a subset of the internal codebase covering the
methods that are established and documented. Work that is still experimental or awaiting
publication is not included here.
