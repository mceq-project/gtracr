gpu_cpp_args = []
if has_gpu
  gpu_cpp_args = ['-DGTRACR_HAS_GPU']
endif

py.extension_module(
  '_libgtracr',
  sources: [
    'src/TrajectoryTracer.cpp',
    'src/BatchGMRC.cpp',
    'src/igrf.cpp',
    'src/pybind11_wrapper.cpp',
    'gpu/igrf_table.cpp',
  ],
  include_directories: [
    include_directories('include'),
    include_directories('extern'),   # for nlohmann/json.hpp
    include_directories('gpu'),      # for igrf_table.hpp
  ],
  dependencies: [py.dependency(), pybind11_dep],
  cpp_args: extra_cpp_args + gpu_cpp_args,
  install: true,
  subdir: 'gtracr',
)
