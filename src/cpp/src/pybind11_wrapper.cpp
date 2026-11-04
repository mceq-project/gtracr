
#include "BatchGMRC.hpp"
#include "MagneticField.hpp"
#include "TrajectoryTracer.hpp"
#include "igrf.hpp"
#include "igrf_table.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

PYBIND11_MODULE(_libgtracr, M) {
  // Bind TableParams so Python can pass it to the shared-table constructor.
  py::class_<TableParams>(M, "TableParams", py::module_local())
      .def(py::init<>())
      .def_readwrite("r_min", &TableParams::r_min)
      .def_readwrite("r_max", &TableParams::r_max)
      .def_readwrite("log_r_min", &TableParams::log_r_min)
      .def_readwrite("log_r_max", &TableParams::log_r_max)
      .def_readwrite("Nr", &TableParams::Nr)
      .def_readwrite("Ntheta", &TableParams::Ntheta)
      .def_readwrite("Nphi", &TableParams::Nphi);

  // Standalone function: generate the IGRF lookup table once in Python,
  // return (numpy array, TableParams) so threads can share the table.
  M.attr("DEFAULT_ESCAPE_FACTOR") = integrator_control::DEFAULT_ESCAPE_FACTOR;
  M.attr("DEFAULT_TIME_FACTOR") = integrator_control::DEFAULT_TIME_FACTOR;
  M.attr("DEFAULT_ATOL") = integrator_control::DEFAULT_ATOL;
  M.attr("DEFAULT_RTOL") = integrator_control::DEFAULT_RTOL;

  M.def(
      "generate_igrf_table",
      [](const std::string& data_dir, double decimal_year) {
        IGRF igrf(data_dir + "/" + igrf_const::JSON_FILE, decimal_year);
        TableParams params;
        std::vector<float> table = generate_igrf_table(igrf, params);

        // Move data into a numpy array (zero-copy via capsule).
        auto* data = new std::vector<float>(std::move(table));
        py::capsule free_when_done(data,
                                   [](void* p) { delete static_cast<std::vector<float>*>(p); });
        py::array_t<float> arr({static_cast<py::ssize_t>(data->size())}, data->data(),
                               free_when_done);

        return py::make_tuple(arr, params);
      },
      py::arg("data_dir"), py::arg("decimal_year"));

  // Bind BatchGMRCParams struct.
  py::class_<BatchGMRCParams>(M, "BatchGMRCParams", py::module_local())
      .def(py::init<>())
      .def_readwrite("latitude", &BatchGMRCParams::latitude)
      .def_readwrite("longitude", &BatchGMRCParams::longitude)
      .def_readwrite("detector_alt", &BatchGMRCParams::detector_alt)
      .def_readwrite("particle_alt", &BatchGMRCParams::particle_alt)
      .def_readwrite("escape_radius", &BatchGMRCParams::escape_radius)
      .def_readwrite("charge", &BatchGMRCParams::charge)
      .def_readwrite("mass", &BatchGMRCParams::mass)
      .def_readwrite("min_rigidity", &BatchGMRCParams::min_rigidity)
      .def_readwrite("max_rigidity", &BatchGMRCParams::max_rigidity)
      .def_readwrite("delta_rigidity", &BatchGMRCParams::delta_rigidity)
      .def_readwrite("dt", &BatchGMRCParams::dt)
      .def_readwrite("max_time", &BatchGMRCParams::max_time)
      .def_readwrite("solver_type", &BatchGMRCParams::solver_type)
      .def_readwrite("bfield_type", &BatchGMRCParams::bfield_type)
      .def_readwrite("atol", &BatchGMRCParams::atol)
      .def_readwrite("rtol", &BatchGMRCParams::rtol)
      .def_readwrite("n_samples", &BatchGMRCParams::n_samples)
      .def_readwrite("n_threads", &BatchGMRCParams::n_threads)
      .def_readwrite("max_attempts_factor", &BatchGMRCParams::max_attempts_factor)
      .def_readwrite("base_seed", &BatchGMRCParams::base_seed)
      .def_readwrite("escape_factor", &BatchGMRCParams::escape_factor)
      .def_readwrite("time_factor", &BatchGMRCParams::time_factor);

  // Bind batch_gmrc_evaluate: takes numpy table (or None for direct IGRF),
  // TableParams, igrf_params, BatchGMRCParams.
  // Returns (zenith, azimuth, rcutoff, above_max, total_trajectories);
  // above_max is a bool array, True where the cutoff exceeds max_rigidity.
  M.def(
      "batch_gmrc_evaluate",
      [](py::object shared_table_obj, const TableParams& table_params,
         const std::pair<std::string, double>& igrf_params, const BatchGMRCParams& params) {
        // Keep the array wrapper alive until after batch_gmrc_evaluate returns,
        // so that `tbl_ptr` is never dangling (forcecast may create a copy whose
        // data buffer would be freed if the wrapper went out of scope too early).
        py::array_t<float, py::array::c_style | py::array::forcecast> shared_table;
        const float* tbl_ptr = nullptr;
        if (!shared_table_obj.is_none()) {
          shared_table = shared_table_obj
                             .cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
          tbl_ptr = shared_table.data();
        }

        BatchGMRCResult result;
        {
          py::gil_scoped_release release;
          result = batch_gmrc_evaluate(tbl_ptr, table_params, igrf_params, params);
        }

        // Convert to numpy arrays via zero-copy capsules.
        auto make_array = [](std::vector<double>&& vec) -> py::array_t<double> {
          auto* data = new std::vector<double>(std::move(vec));
          py::capsule free_when_done(data,
                                     [](void* p) { delete static_cast<std::vector<double>*>(p); });
          return py::array_t<double>({static_cast<py::ssize_t>(data->size())}, data->data(),
                                     free_when_done);
        };

        py::array_t<bool> above_max(static_cast<py::ssize_t>(result.above_max.size()));
        auto above_max_view = above_max.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < above_max_view.shape(0); ++i) {
          above_max_view(i) = result.above_max[static_cast<size_t>(i)] != 0;
        }

        return py::make_tuple(
            make_array(std::move(result.zenith)), make_array(std::move(result.azimuth)),
            make_array(std::move(result.rcutoff)), above_max, result.total_trajectories);
      },
      py::arg("shared_table"), py::arg("table_params"), py::arg("igrf_params"), py::arg("params"));

  py::class_<PenumbraParams>(M, "PenumbraParams", py::module_local())
      .def(py::init<>())
      .def_readwrite("r_lo", &PenumbraParams::r_lo)
      .def_readwrite("r_hi", &PenumbraParams::r_hi)
      .def_readwrite("delta", &PenumbraParams::delta)
      .def_readwrite("coarse", &PenumbraParams::coarse)
      .def_readwrite("guard", &PenumbraParams::guard)
      .def_readwrite("store_ladder", &PenumbraParams::store_ladder);

  // Bind batch_gmrc_penumbra. Returns a dict with zenith, azimuth, r_lower, r_upper, r_eff,
  // flag, ladder, total_trajectories and (if store_ladder) allowed of shape (n, K).
  M.def(
      "batch_gmrc_penumbra",
      [](py::object shared_table_obj, const TableParams& table_params,
         const std::pair<std::string, double>& igrf_params, const BatchGMRCParams& params,
         const PenumbraParams& pparams, std::vector<double> zenith, std::vector<double> azimuth) {
        if (zenith.size() != azimuth.size())
          throw std::invalid_argument("zenith and azimuth must have the same length");
        if (!(pparams.r_lo > 0.0) || !(pparams.r_hi > pparams.r_lo) || !(pparams.delta > 0.0))
          throw std::invalid_argument("penumbra ladder needs 0 < r_lo < r_hi and delta > 0");
        py::array_t<float, py::array::c_style | py::array::forcecast> shared_table;
        const float* tbl_ptr = nullptr;
        if (!shared_table_obj.is_none()) {
          shared_table = shared_table_obj
                             .cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
          tbl_ptr = shared_table.data();
        }
        if (params.bfield_type == 't' && tbl_ptr == nullptr)
          throw std::invalid_argument("bfield_type 't' requires a shared table");

        PenumbraResult r;
        {
          py::gil_scoped_release release;
          r = batch_gmrc_penumbra(tbl_ptr, table_params, igrf_params, params, pparams, zenith,
                                  azimuth);
        }
        auto arr = [](const auto& v) {
          using T = typename std::decay_t<decltype(v)>::value_type;
          py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
          std::copy(v.begin(), v.end(), a.mutable_data());
          return a;
        };
        py::dict d;
        d["zenith"] = arr(r.zenith);
        d["azimuth"] = arr(r.azimuth);
        d["r_lower"] = arr(r.r_lower);
        d["r_upper"] = arr(r.r_upper);
        d["r_eff"] = arr(r.r_eff);
        d["flag"] = arr(r.flag);
        d["ladder"] = arr(r.ladder);
        d["total_trajectories"] = r.total_trajectories;
        if (pparams.store_ladder) {
          py::array_t<bool> a({static_cast<py::ssize_t>(r.zenith.size()),
                               static_cast<py::ssize_t>(r.ladder.size())});
          std::copy(r.allowed.begin(), r.allowed.end(), a.mutable_data());
          d["allowed"] = a;
        }
        return d;
      },
      py::arg("shared_table"), py::arg("table_params"), py::arg("igrf_params"), py::arg("params"),
      py::arg("penumbra_params"), py::arg("zenith") = std::vector<double>{},
      py::arg("azimuth") = std::vector<double>{});

  py::class_<TrajectoryTracer>(M, "TrajectoryTracer", py::module_local())
      .def(py::init<>())
      // Full constructor with optional solver_type / atol / rtol.
      // solver_type: 'r' = frozen-field RK4, 'b' = Boris, 'a' = adaptive RK45.
      .def(py::init<double, double, double, double, double, int, const char,
                    const std::pair<std::string, double>&, const char, double, double>(),
           py::arg("charge"), py::arg("mass") = 1.67e-27, py::arg("start_altitude") = 100e3,
           py::arg("escape_radius") = 10. * 6371.2e3, py::arg("stepsize") = 1e-5,
           py::arg("max_iter") = 10000, py::arg("bfield_type") = 'i',
           py::arg("igrf_params") = std::pair<std::string, double>{"", 2020.},
           py::arg("solver_type") = 'r', py::arg("atol") = integrator_control::DEFAULT_ATOL,
           py::arg("rtol") = integrator_control::DEFAULT_RTOL)
      // Shared-table constructor: borrows a numpy array as the IGRF table.
      .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> shared_table,
                       const TableParams& table_params, double charge, double mass,
                       double start_altitude, double escape_radius, double stepsize, int max_iter,
                       const std::pair<std::string, double>& igrf_params, char solver_type,
                       double atol, double rtol) {
             const float* ptr = shared_table.data();
             return new TrajectoryTracer(ptr, table_params, charge, mass, start_altitude,
                                         escape_radius, stepsize, max_iter, igrf_params,
                                         solver_type, atol, rtol);
           }),
           py::arg("shared_table"), py::arg("table_params"), py::arg("charge"), py::arg("mass"),
           py::arg("start_altitude"), py::arg("escape_radius"), py::arg("stepsize"),
           py::arg("max_iter"), py::arg("igrf_params"), py::arg("solver_type") = 'r',
           py::arg("atol") = integrator_control::DEFAULT_ATOL,
           py::arg("rtol") = integrator_control::DEFAULT_RTOL,
           // prevent the numpy array from being garbage-collected while tracer lives
           py::keep_alive<1, 2>())
      .def_property_readonly("charge", &TrajectoryTracer::charge)
      .def_property_readonly("mass", &TrajectoryTracer::mass)
      .def_property_readonly("start_altitude", &TrajectoryTracer::start_altitude)
      .def_property_readonly("escape_radius", &TrajectoryTracer::escape_radius)
      .def_property_readonly("step_size", &TrajectoryTracer::stepsize)
      .def_property_readonly("max_iter", &TrajectoryTracer::max_iter)
      .def_property_readonly("particle_escaped", &TrajectoryTracer::particle_escaped)
      .def_property_readonly("final_time", &TrajectoryTracer::final_time)
      .def_property_readonly("final_sixvector", &TrajectoryTracer::final_sixvector)
      .def_property_readonly("nsteps", &TrajectoryTracer::nsteps)
      .def_property_readonly("solver_type", &TrajectoryTracer::solver_type)
      .def_property_readonly("escape_factor", &TrajectoryTracer::escape_factor)
      .def_property_readonly("time_factor", &TrajectoryTracer::time_factor)
      .def_property_readonly("run_escape_radius", &TrajectoryTracer::run_escape_radius)
      .def_property_readonly("run_max_time", &TrajectoryTracer::run_max_time)
      .def_property_readonly("run_max_iter", &TrajectoryTracer::run_max_iter)
      .def("dipole_b0", &TrajectoryTracer::dipole_b0)
      .def("stormer_length", &TrajectoryTracer::stormer_length, py::arg("momentum"))
      .def("set_escape_scaling", &TrajectoryTracer::set_escape_scaling, py::arg("escape_factor"),
           py::arg("time_factor"))
      .def("reset", &TrajectoryTracer::reset, py::call_guard<py::gil_scoped_release>())
      .def("set_start_altitude", &TrajectoryTracer::set_start_altitude,
           py::call_guard<py::gil_scoped_release>())
      .def("evaluate", &TrajectoryTracer::evaluate, py::call_guard<py::gil_scoped_release>())
      .def("evaluate_and_get_trajectory", &TrajectoryTracer::evaluate_and_get_trajectory)
      .def("find_cutoff_rigidity", &TrajectoryTracer::find_cutoff_rigidity,
           py::call_guard<py::gil_scoped_release>())
      .def("find_cutoff_rigidity_bisect", &TrajectoryTracer::find_cutoff_rigidity_bisect,
           py::call_guard<py::gil_scoped_release>());

  py::class_<MagneticField>(M, "MagneticField", py::module_local())
      .def(py::init<>())
      .def("values", &MagneticField::values);

  py::class_<IGRF>(M, "IGRF", py::module_local())
      .def(py::init<const std::string&, const double>())
      .def_property_readonly("sdate", &IGRF::sdate)
      .def_property_readonly("nmax", &IGRF::nmax)
      .def_property_readonly("cartesian_values", &IGRF::cartesian_values)
      .def("values", &IGRF::values);
}
