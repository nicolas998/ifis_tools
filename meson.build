project(
    'ifis_tools',
    ['c', 'fortran'],
    version: '0.6.0',
    meson_version: '>=1.1.0',
    default_options: ['warning_level=1'],
)

# Python installation (pure: false -> platform-specific wheel, allows compiled ext)
py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# Locate NumPy and f2py include directories at configure time.
incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check: true,
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check: true,
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)

# fortranobject.c ships with f2py and must be compiled into the extension.
fortranobject_c = incdir_f2py / 'fortranobject.c'

# On Windows + MinGW (gfortran/gcc), statically link the GCC/gfortran runtime so
# the resulting .pyd does not depend on libgfortran/libgcc/libquadmath/
# libwinpthread DLLs being on PATH at import time.
cc = meson.get_compiler('c')
ext_link_args = []
if host_machine.system() == 'windows' and cc.get_id() == 'gcc'
    ext_link_args += ['-static', '-static-libgcc', '-static-libgfortran', '-lquadmath']
endif

# Generate the C wrapper + Fortran module wrappers from the .f90 source.
read_dat_wrappers = custom_target('read_dat_module',
    input: ['ifis_tools/read_dat.f90'],
    output: ['read_datmodule.c', 'read_dat-f2pywrappers2.f90'],
    command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'read_dat', '--lower'],
)

# Build the compiled extension inside the ifis_tools package.
py.extension_module('read_dat',
    [
        'ifis_tools/read_dat.f90',
        read_dat_wrappers,
        fortranobject_c,
    ],
    include_directories: inc_np,
    dependencies: py_dep,
    link_args: ext_link_args,
    subdir: 'ifis_tools',
    install: true,
)

# Pure-Python sources.
py.install_sources(
    [
        'ifis_tools/__init__.py',
        'ifis_tools/asynch_manager.py',
        'ifis_tools/auxiliar.py',
        'ifis_tools/database_tools.py',
        'ifis_tools/from_taudem.py',
        'ifis_tools/plot_tools.py',
        'ifis_tools/series_tools.py',
    ],
    subdir: 'ifis_tools',
)

# Package data files (templates, base configs, etc.).
install_data(
    [
        'ifis_tools/evap.mon',
        'ifis_tools/60XBaseGlobal.gbl',
        'ifis_tools/254BaseGlobal.gbl',
        'ifis_tools/190BaseGlobal.gbl',
        'ifis_tools/XXXBaseGlobal.gbl',
        'ifis_tools/BaseInitial.dbc',
        'ifis_tools/BaseInitial_link.dbc',
        'ifis_tools/BaseRun.sh',
        'ifis_tools/ControlPoints.sav',
    ],
    install_dir: py.get_install_dir() / 'ifis_tools',
)
