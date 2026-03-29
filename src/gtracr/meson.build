py.install_sources(
  [
    '__init__.py',
    '_fallback.py',
    'constants.py',
    'geomagnetic_cutoffs.py',
    'location.py',
    'particle.py',
    'plotting.py',
    'trajectory.py',
    'utils.py',
  ],
  subdir: 'gtracr',
)

install_subdir('data', install_dir: py.get_install_dir() / 'gtracr')

subdir('bfield')
subdir('scripts')
