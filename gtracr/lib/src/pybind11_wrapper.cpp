
#include "MagneticField.hpp"
#include "TrajectoryTracer.hpp"
#include "igrf.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(_libgtracr, M) {
  py::class_<TrajectoryTracer>(M, "TrajectoryTracer", py::module_local())
      .def(py::init<>())
      // Full constructor with optional solver_type / atol / rtol.
      // solver_type: 'r' = frozen-field RK4, 'b' = Boris, 'a' = adaptive RK45.
      .def(py::init<double, double, double, double, double,
                    int, const char,
                    const std::pair<std::string, double> &,
                    const char, double, double>(),
           py::arg("charge"),
           py::arg("mass")           = 1.67e-27,
           py::arg("start_altitude") = 100e3,
           py::arg("escape_radius")  = 10. * 6371.2e3,
           py::arg("stepsize")       = 1e-5,
           py::arg("max_iter")       = 10000,
           py::arg("bfield_type")    = 'i',
           py::arg("igrf_params")    = std::pair<std::string, double>{"", 2020.},
           py::arg("solver_type")    = 'r',
           py::arg("atol")           = 1e-3,
           py::arg("rtol")           = 1e-6)
      .def_property_readonly("charge",           &TrajectoryTracer::charge)
      .def_property_readonly("mass",             &TrajectoryTracer::mass)
      .def_property_readonly("start_altitude",   &TrajectoryTracer::start_altitude)
      .def_property_readonly("escape_radius",    &TrajectoryTracer::escape_radius)
      .def_property_readonly("step_size",        &TrajectoryTracer::stepsize)
      .def_property_readonly("max_iter",         &TrajectoryTracer::max_iter)
      .def_property_readonly("particle_escaped", &TrajectoryTracer::particle_escaped)
      .def_property_readonly("final_time",       &TrajectoryTracer::final_time)
      .def_property_readonly("final_sixvector",  &TrajectoryTracer::final_sixvector)
      .def_property_readonly("nsteps",           &TrajectoryTracer::nsteps)
      .def_property_readonly("solver_type",      &TrajectoryTracer::solver_type)
      .def("reset",                              &TrajectoryTracer::reset)
      .def("set_start_altitude",                 &TrajectoryTracer::set_start_altitude)
      .def("evaluate",                           &TrajectoryTracer::evaluate)
      .def("evaluate_and_get_trajectory",        &TrajectoryTracer::evaluate_and_get_trajectory);

  py::class_<MagneticField>(M, "MagneticField", py::module_local())
      .def(py::init<>())
      .def("values", &MagneticField::values);

  py::class_<IGRF>(M, "IGRF", py::module_local())
      .def(py::init<const std::string &, const double>())
      .def_property_readonly("sdate",            &IGRF::sdate)
      .def_property_readonly("nmax",             &IGRF::nmax)
      .def_property_readonly("cartesian_values", &IGRF::cartesian_values)
      .def("values",                             &IGRF::values);
}
