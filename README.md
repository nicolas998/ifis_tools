# ifis_tools

Set of tools to do different tasks related to the IFIS ASYNCH model and
streamflow series analysis.

The package contains a small Fortran extension (`read_dat`) that is compiled at
install time with [f2py](https://numpy.org/doc/stable/f2py/). Building now uses
the modern [`meson-python`](https://mesonbuild.com/meson-python/) backend, so it
installs cleanly on Python 3.12+ on both Windows and Linux.

## Build requirements

You need a Fortran (and C) compiler available on your `PATH` before installing:

- **Linux**: `gfortran` (e.g. `sudo apt install gfortran` or `conda install gfortran`).
- **Windows**: easiest is conda-forge:
  `conda install -c conda-forge fortran-compiler`
  (alternatively install gfortran via MSYS2 / WinLibs and add it to `PATH`).

The Python build dependencies (`meson-python`, `ninja`, `numpy`) are pulled in
automatically by pip from `pyproject.toml`.

## Install

```bash
pip install .
```

For an editable / development install:

```bash
pip install --no-build-isolation -e .
```

(`--no-build-isolation` requires `meson-python`, `ninja` and `numpy` to be
installed in the current environment.)

## Optional dependencies

```bash
pip install ".[hdf]"    # h5py, for reading HLM hdf snapshots
pip install ".[eval]"   # hydroeval, for KGE / NSE / pbias metrics
```
