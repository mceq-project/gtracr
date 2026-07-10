
#include "BatchGMRC.hpp"
#include "MagneticField.hpp"
#include "MuonTracer.hpp"
#include "TrajectoryTracer.hpp"
#include "igrf.hpp"
#include "igrf_table.hpp"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace {

// Move a std::vector<double> into a numpy array (zero-copy via capsule).
py::array_t<double> to_array(std::vector<double>&& vec, std::vector<py::ssize_t> shape) {
  auto* data = new std::vector<double>(std::move(vec));
  py::capsule free_when_done(data, [](void* p) { delete static_cast<std::vector<double>*>(p); });
  return py::array_t<double>(shape, data->data(), free_when_done);
}

py::array_t<int8_t> to_array_i8(std::vector<int8_t>&& vec) {
  auto* data = new std::vector<int8_t>(std::move(vec));
  py::capsule free_when_done(data, [](void* p) { delete static_cast<std::vector<int8_t>*>(p); });
  return py::array_t<int8_t>({static_cast<py::ssize_t>(data->size())}, data->data(),
                             free_when_done);
}

std::vector<double> as_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
  return std::vector<double>(arr.data(), arr.data() + arr.size());
}

}  // namespace

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
  M.def(
      "generate_igrf_table",
      [](const std::string& data_dir, double decimal_year) {
        IGRF igrf(data_dir + "/igrf13.json", decimal_year);
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
      .def_readwrite("base_seed", &BatchGMRCParams::base_seed);

  // Bind batch_gmrc_evaluate: takes numpy table (or None for direct IGRF),
  // TableParams, igrf_params, BatchGMRCParams.
  // Returns (zenith, azimuth, rcutoff, total_trajectories).
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

        return py::make_tuple(make_array(std::move(result.zenith)),
                              make_array(std::move(result.azimuth)),
                              make_array(std::move(result.rcutoff)), result.total_trajectories);
      },
      py::arg("shared_table"), py::arg("table_params"), py::arg("igrf_params"), py::arg("params"));

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
           py::arg("solver_type") = 'r', py::arg("atol") = 1e-3, py::arg("rtol") = 1e-6)
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
           py::arg("atol") = 1e-3, py::arg("rtol") = 1e-6,
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

  // -------------------------------------------------------------------------
  // MuonTracer ("mutracr") — batch atmospheric-muon transport.
  // Constructed via keyword-style factory; the ergonomic Python API lives
  // in gtracr.mutracr.MuonTracer (this binding is the raw engine).
  // Units: cm, s, GeV, Tesla, g/cm^3 (see MuonTracer.hpp).
  // -------------------------------------------------------------------------
  py::class_<muon::MuonTracer>(M, "MuonTracer", py::module_local())
      .def(py::init([](const std::string& bfield_type, std::array<double, 3> b_uniform,
                       double dipole_g10, double dipole_r_ref,
                       const std::pair<std::string, double>& igrf_params, int shell_n_h,
                       int shell_n_colat, int shell_n_lon, double shell_h_max,
                       const std::string& atmo_type, double atmo_rho0, py::object atmo_h,
                       py::object atmo_rho, double atmo_h_top, double dedx_const,
                       py::object dedx_e_kin, py::object dedx_a, double r_ground, double h_escape) {
             // field
             std::unique_ptr<muon::MuonField> field;
             if (bfield_type == "none") {
               field.reset(new muon::MuonField(std::array<double, 3>{0.0, 0.0, 0.0}));
             } else if (bfield_type == "uniform") {
               field.reset(new muon::MuonField(b_uniform));
             } else if (bfield_type == "dipole") {
               field.reset(new muon::MuonField(dipole_g10, dipole_r_ref));
             } else if (bfield_type == "shell") {
               field.reset(new muon::MuonField(igrf_params.first + "/igrf13.json",
                                               igrf_params.second, r_ground, shell_n_h,
                                               shell_n_colat, shell_n_lon, shell_h_max));
             } else {
               throw std::invalid_argument("bfield_type must be none|uniform|dipole|shell");
             }
             // atmosphere
             std::unique_ptr<muon::MuonAtmosphere> atmo;
             if (atmo_type == "none") {
               atmo.reset(new muon::MuonAtmosphere());
             } else if (atmo_type == "usstd") {
               atmo.reset(new muon::MuonAtmosphere(muon::MuonAtmosphere::us_std()));
             } else if (atmo_type == "uniform") {
               atmo.reset(new muon::MuonAtmosphere(atmo_rho0, atmo_h_top));
             } else if (atmo_type == "table") {
               auto h = as_vector(
                   atmo_h.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
               auto r = as_vector(
                   atmo_rho.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
               atmo.reset(new muon::MuonAtmosphere(h, r, atmo_h_top));
             } else {
               throw std::invalid_argument("atmo_type must be none|usstd|uniform|table");
             }
             // dE/dX
             std::unique_ptr<muon::MuonDedx> dedx;
             if (dedx_e_kin.is_none()) {
               dedx.reset(new muon::MuonDedx(dedx_const));
             } else {
               auto e = as_vector(
                   dedx_e_kin
                       .cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
               auto a = as_vector(
                   dedx_a.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
               dedx.reset(new muon::MuonDedx(e, a));
             }
             return new muon::MuonTracer(std::move(*field), std::move(*atmo), std::move(*dedx),
                                         r_ground, h_escape);
           }),
           py::arg("bfield_type") = "none",
           py::arg("b_uniform") = std::array<double, 3>{0.0, 0.0, 0.0},
           py::arg("dipole_g10") = 29404.8e-9, py::arg("dipole_r_ref") = 6.3712e8,
           py::arg("igrf_params") = std::pair<std::string, double>{"", 2020.},
           py::arg("shell_n_h") = 13, py::arg("shell_n_colat") = 91, py::arg("shell_n_lon") = 181,
           py::arg("shell_h_max") = 1.3e7, py::arg("atmo_type") = "none",
           py::arg("atmo_rho0") = 1.2e-3, py::arg("atmo_h") = py::none(),
           py::arg("atmo_rho") = py::none(), py::arg("atmo_h_top") = 1.128e7,
           py::arg("dedx_const") = 2.0e-3, py::arg("dedx_e_kin") = py::none(),
           py::arg("dedx_a") = py::none(), py::arg("r_ground") = 6.3712e8,
           py::arg("h_escape") = 1.20e7)
      .def_property_readonly("r_ground", &muon::MuonTracer::r_ground)
      .def_property_readonly("h_escape", &muon::MuonTracer::h_escape)
      .def(
          "bfield",
          [](const muon::MuonTracer& self,
             py::array_t<double, py::array::c_style | py::array::forcecast> x) {
            if (x.ndim() != 2 || x.shape(1) != 3)
              throw std::invalid_argument("x must have shape (N, 3)");
            const py::ssize_t n = x.shape(0);
            std::vector<double> out(3 * n);
            const double* xd = x.data();
            for (py::ssize_t i = 0; i < n; ++i) {
              const auto b = self.bfield_at({xd[3 * i], xd[3 * i + 1], xd[3 * i + 2]});
              out[3 * i] = b[0];
              out[3 * i + 1] = b[1];
              out[3 * i + 2] = b[2];
            }
            return to_array(std::move(out), {n, 3});
          },
          py::arg("x"), "B [T] at Cartesian positions x [cm], shape (N, 3).")
      .def(
          "rho",
          [](const muon::MuonTracer& self,
             py::array_t<double, py::array::c_style | py::array::forcecast> h) {
            std::vector<double> out(h.size());
            const double* hd = h.data();
            for (py::ssize_t i = 0; i < h.size(); ++i) out[i] = self.rho_at(hd[i]);
            return to_array(std::move(out), {static_cast<py::ssize_t>(out.size())});
          },
          py::arg("h"), "Density [g/cm^3] at altitudes h [cm] above the ground sphere.")
      .def(
          "dedx",
          [](const muon::MuonTracer& self,
             py::array_t<double, py::array::c_style | py::array::forcecast> e_tot) {
            std::vector<double> out(e_tot.size());
            const double* ed = e_tot.data();
            for (py::ssize_t i = 0; i < e_tot.size(); ++i) out[i] = self.dedx_at(ed[i]);
            return to_array(std::move(out), {static_cast<py::ssize_t>(out.size())});
          },
          py::arg("e_tot"), "dE/dX [GeV/(g/cm^2)] at total energies e_tot [GeV].")
      .def(
          "trace",
          [](const muon::MuonTracer& self,
             py::array_t<double, py::array::c_style | py::array::forcecast> x,
             py::array_t<double, py::array::c_style | py::array::forcecast> p,
             py::array_t<long long, py::array::c_style | py::array::forcecast> q,
             py::array_t<double, py::array::c_style | py::array::forcecast> pol,
             py::array_t<double, py::array::c_style | py::array::forcecast> w, double e_min,
             double w_min, double dt_max, long max_steps, double ctl_bend, double ctl_loss,
             double ctl_decay, double ctl_path, bool decay_beta_one, int n_threads,
             const std::string& deposit, double w_thresh, py::object e_edges, py::object d_edges,
             std::array<double, 3> axis) {
            if (x.ndim() != 2 || x.shape(1) != 3 || p.ndim() != 2 || p.shape(1) != 3)
              throw std::invalid_argument("x and p must have shape (N, 3)");
            const py::ssize_t n = x.shape(0);
            std::vector<double> xv(x.data(), x.data() + 3 * n);
            std::vector<double> pv(p.data(), p.data() + 3 * n);
            std::vector<int8_t> qv(n);
            for (py::ssize_t i = 0; i < n; ++i) qv[i] = static_cast<int8_t>(q.data()[i]);
            std::vector<double> polv(pol.data(), pol.data() + pol.size());
            std::vector<double> wv(w.data(), w.data() + w.size());

            muon::TraceOptions opt;
            opt.e_min = e_min;
            opt.w_min = w_min;
            opt.dt_max = dt_max;
            opt.max_steps = max_steps;
            opt.ctl_bend = ctl_bend;
            opt.ctl_loss = ctl_loss;
            opt.ctl_decay = ctl_decay;
            opt.ctl_path = ctl_path;
            opt.decay_beta_one = decay_beta_one;
            opt.n_threads = n_threads;

            muon::DepositConfig dep;
            if (deposit == "none") {
              dep.kind = 'n';
            } else if (deposit == "bank") {
              dep.kind = 'b';
            } else if (deposit == "spectrum") {
              dep.kind = 's';
            } else if (deposit == "angular") {
              dep.kind = 'a';
            } else {
              throw std::invalid_argument("deposit must be none|bank|spectrum|angular");
            }
            dep.w_thresh = w_thresh;
            dep.axis = axis;
            if (!e_edges.is_none()) {
              dep.e_edges = as_vector(
                  e_edges.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
            }
            if (!d_edges.is_none()) {
              dep.d_edges = as_vector(
                  d_edges.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
            }

            muon::MuTraceResult res;
            {
              py::gil_scoped_release release;
              res = self.trace(xv, pv, qv, polv, wv, opt, dep);
            }

            py::dict out;
            out["x"] = to_array(std::move(res.x), {n, 3});
            out["p"] = to_array(std::move(res.p), {n, 3});
            out["w"] = to_array(std::move(res.w), {n});
            out["th2"] = to_array(std::move(res.th2), {n});
            out["fate"] = to_array_i8(std::move(res.fate));
            out["steps"] = res.steps;
            out["deposited"] = res.deposited;
            py::dict fw;
            fw["ground"] = res.fate_w[0];
            fw["stopped"] = res.fate_w[1];
            fw["faded"] = res.fate_w[2];
            fw["escaped"] = res.fate_w[3];
            fw["maxsteps"] = res.fate_w[4];
            out["fate_w"] = fw;
            out["weight_audit"] = res.weight_audit;
            out["n"] = n;
            if (dep.kind == 'b') {
              const py::ssize_t m = static_cast<py::ssize_t>(res.bank_e.size());
              py::dict bank;
              bank["x"] = to_array(std::move(res.bank_x), {m, 3});
              bank["phat"] = to_array(std::move(res.bank_phat), {m, 3});
              bank["e_mu"] = to_array(std::move(res.bank_e), {m});
              bank["q"] = to_array_i8(std::move(res.bank_q));
              bank["pol"] = to_array(std::move(res.bank_pol), {m});
              bank["sig_mcs"] = to_array(std::move(res.bank_sig), {m});
              bank["w"] = to_array(std::move(res.bank_w), {m});
              bank["total_weight"] = res.bank_total_w;
              out["bank"] = bank;
            } else if (dep.kind == 's') {
              const py::ssize_t ne = static_cast<py::ssize_t>(res.spec_w_pos.size());
              py::dict spec;
              spec["w_pos"] = to_array(std::move(res.spec_w_pos), {ne});
              spec["wp_pos"] = to_array(std::move(res.spec_wp_pos), {ne});
              spec["w_neg"] = to_array(std::move(res.spec_w_neg), {ne});
              spec["wp_neg"] = to_array(std::move(res.spec_wp_neg), {ne});
              out["spectrum"] = spec;
            } else if (dep.kind == 'a') {
              const py::ssize_t ne = static_cast<py::ssize_t>(dep.e_edges.size()) - 1;
              const py::ssize_t nd = static_cast<py::ssize_t>(dep.d_edges.size()) - 1;
              py::dict ang;
              ang["w_pos"] = to_array(std::move(res.ang_w_pos), {ne, nd});
              ang["wp_pos"] = to_array(std::move(res.ang_wp_pos), {ne, nd});
              ang["ws2_pos"] = to_array(std::move(res.ang_ws2_pos), {ne, nd});
              ang["w_neg"] = to_array(std::move(res.ang_w_neg), {ne, nd});
              ang["wp_neg"] = to_array(std::move(res.ang_wp_neg), {ne, nd});
              ang["ws2_neg"] = to_array(std::move(res.ang_ws2_neg), {ne, nd});
              ang["overflow"] = res.ang_overflow;
              out["angular"] = ang;
            }
            return out;
          },
          py::arg("x"), py::arg("p"), py::arg("q"), py::arg("pol"), py::arg("w"),
          py::arg("e_min") = 0.11, py::arg("w_min") = 1e-6, py::arg("dt_max") = 0.0,
          py::arg("max_steps") = 100000, py::arg("ctl_bend") = 0.05, py::arg("ctl_loss") = 0.05,
          py::arg("ctl_decay") = 0.05, py::arg("ctl_path") = 2e5, py::arg("decay_beta_one") = false,
          py::arg("n_threads") = 0, py::arg("deposit") = "none", py::arg("w_thresh") = 0.0,
          py::arg("e_edges") = py::none(), py::arg("d_edges") = py::none(),
          py::arg("axis") = std::array<double, 3>{0.0, 0.0, 1.0});

  py::class_<IGRF>(M, "IGRF", py::module_local())
      .def(py::init<const std::string&, const double>())
      .def_property_readonly("sdate", &IGRF::sdate)
      .def_property_readonly("nmax", &IGRF::nmax)
      .def_property_readonly("cartesian_values", &IGRF::cartesian_values)
      .def("values", &IGRF::values);
}
