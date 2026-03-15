#include "igrf.hpp"

using json = nlohmann::json;

/*
 Initialize the IGRF model. The coefficients are imported and are stored
 in the corresponding arrays.

 Parameters
 -----------
 - fname (std::string) :
       The path to the .COF file.
   - sdate (double) :
       The current year, month, and day in decimal days
 */
IGRF::IGRF(const std::string& fname, const double sdate)
    : model_index{0}, nmodel{24}, igdgc{3}, sdate_{sdate}, 
    epoch1_{0.}, epoch2_{0.}, nmain1_{0}, nmain2_{0}, 
    nsv1_{0}, nsv2_{0} {

  if (sdate_ > igrf_const::MAXEPOCH) {
      epoch1_ = igrf_const::MAXEPOCH;
      epoch2_ = igrf_const::MAXEPOCH + 5.;
  }
  else {
  // get epochs in which we want to interpolate / extrapolate
  for (double epoch=1900; epoch<=igrf_const::MAXEPOCH; epoch+=5) {
    // set date to maximum epoch if given date is extrapolated
        if ((sdate_ - epoch) < 2.5) {  // not abs value to get the epoch before the date
        epoch1_ = epoch;
        epoch2_ = epoch + 5.;
        break;
      }
    }
  }

  // get the spherical harmonic coefficients
  getshc(fname);

  // get the spherical harmonic coeffiecients for the specific date
  // and either interpolate or extrapolate if we have to
  if (epoch2_ + 5 > igrf_const::MAXEPOCH)  // if model isnt the last model in the file
  {
    extrapsh(sdate_, 3);
    extrapsh(sdate_ + 1, 4);
  } else {
    interpsh(sdate_, 3);
    interpsh(sdate_ + 1, 4);
  }
}

/****************************************************************************/
/*                                                                          */
/*                           Subroutine getshc                              */
/*                                                                          */
/****************************************************************************/
/*                                                                          */
/*     Reads spherical harmonic coefficients from the specified             */
/*     model into an array.                                                 */
/*                                                                          */
/*     Input:                                                               */
/*           stream     - Logical unit number                               */
/*           iflag      - Flag for SV equal to ) or not equal to 0          */
/*                        for designated read statements                    */
/*           strec      - Starting record number to read from model         */
/*           nmax_of_gh - Maximum degree and order of model                 */
/*                                                                          */
/*     Output:                                                              */
/*           gh1 or 2   - Schmidt quasi-normal internal spherical           */
/*                        harmonic coefficients                             */
/*                                                                          */
/*     FORTRAN                                                              */
/*           Bill Flanagan                                                  */
/*           NOAA CORPS, DESDIS, NGDC, 325 Broadway, Boulder CO.  80301     */
/*                                                                          */
/*     C                                                                    */
/*           C. H. Shaffer                                                  */
/*           Lockheed Missiles and Space Company, Sunnyvale CA              */
/*           August 15, 1988                                                */
/*                                                                          */
/****************************************************************************/
void IGRF::getshc(const std::string& fname) {
  std::ifstream ifs{fname};
  if (!ifs) {
    throw std::runtime_error("Cannot open file!");
  }

  json igrf_map;

  ifs >> igrf_map;  // read contents of json file to json object

  // convert epoch to strings since thats the form for json keys
  std::string epoch1 = std::to_string(epoch1_);
  std::string epoch2 = std::to_string(epoch2_);

  // set variables (nmain, nsv, gh, gh_sv)
  // nmain stores the spherical-harmonic degree; the actual number of
  // Gauss coefficients is nmain*(nmain+2).  interpsh/extrapsh use
  // 1-based indexing into these arrays (indices 1..ncoeffs), so we
  // fill starting at index 1 and map JSON index j → array index j+1.
  nmain1_ = igrf_map[epoch1]["nmain"];
  nsv1_ = igrf_map[epoch1]["nsv"];
  {
    int ncoeffs1 = nmain1_ * (nmain1_ + 2);
    for (int i = 1; i <= ncoeffs1; ++i) {
      gh1_arr[i]   = igrf_map[epoch1]["gh"][i - 1];
      ghsv1_arr[i] = igrf_map[epoch1]["gh_sv"][i - 1];
    }
  }

  // only set epoch2 if epoch2 is smaller than the latest epoch
  if (epoch2_ < igrf_const::MAXEPOCH) {
    nmain2_ = igrf_map[epoch2]["nmain"];
    nsv2_ = igrf_map[epoch2]["nsv"];
    int ncoeffs2 = nmain2_ * (nmain2_ + 2);
    for (int i = 1; i <= ncoeffs2; ++i) {
      gh2_arr[i]   = igrf_map[epoch2]["gh"][i - 1];
      ghsv2_arr[i] = igrf_map[epoch2]["gh_sv"][i - 1];
    }
  }

}  // getshc

/****************************************************************************/
/*                                                                          */
/*                           Subroutine interpsh                            */
/*                                                                          */
/****************************************************************************/
/*                                                                          */
/*     Interpolates linearly, in time, between two spherical harmonic       */
/*     models.                                                              */
/*                                                                          */
/*     Input:                                                               */
/*           date     - date of resulting model (in decimal year)           */
/*           epoch1_     - date of earlier model                               */
/*           nmax1    - maximum degree and order of earlier model           */
/*           gh1      - Schmidt quasi-normal internal spherical             */
/*                      harmonic coefficients of earlier model              */
/*           epoch2_     - date of later model                                 */
/*           nmax2    - maximum degree and order of later model             */
/*           gh2      - Schmidt quasi-normal internal spherical             */
/*                      harmonic coefficients of internal model             */
/*                                                                          */
/*     Output:                                                              */
/*           gha or b - coefficients of resulting model                     */
/*           nmax     - maximum degree and order of resulting model         */
/*                                                                          */
/*     FORTRAN                                                              */
/*           A. Zunde                                                       */
/*           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225     */
/*                                                                          */
/*     C                                                                    */
/*           C. H. Shaffer                                                  */
/*           Lockheed Missiles and Space Company, Sunnyvale CA              */
/*           August 17, 1988                                                */
/*                                                                          */
/****************************************************************************/
void IGRF::interpsh(double date, int gh) {
  // int nmax;
  int k, l;
  int ii;
  double factor;


  factor = (date - epoch1_) / (epoch2_ - epoch1_);
  if (nmain1_ == nmain2_) {
    k = nmain1_ * (nmain1_ + 2);
    nmax_ = nmain1_;
  } else {
    if (nmain1_ > nmain2_) {
      k = nmain2_ * (nmain2_ + 2);
      l = nmain1_ * (nmain1_ + 2);
      switch (gh) {
        case 3:
          for (ii = k + 1; ii <= l; ++ii) {
            gh_arr[ii] = gh1_arr[ii] + factor * (-gh1_arr[ii]);
          }
          break;
        case 4:
          for (ii = k + 1; ii <= l; ++ii) {
            ghsv_arr[ii] = gh1_arr[ii] + factor * (-gh1_arr[ii]);
          }
          break;
        default:
          std::cout << "\nError in subroutine extrapsh" << std::endl;
          break;
      }
      nmax_ = nmain1_;
    } else {
      k = nmain1_ * (nmain1_ + 2);
      l = nmain2_ * (nmain2_ + 2);
      switch (gh) {
        case 3:
          for (ii = k + 1; ii <= l; ++ii) {
            gh_arr[ii] = factor * gh2_arr[ii];
          }
          break;
        case 4:
          for (ii = k + 1; ii <= l; ++ii) {
            ghsv_arr[ii] = factor * gh2_arr[ii];
          }
          break;
        default:
          std::cout << "\nError in subroutine extrapsh" << std::endl;
          break;
      }
      nmax_ = nmain2_;
    }
  }
  switch (gh) {
    case 3:
      for (ii = 1; ii <= k; ++ii) {
        gh_arr[ii] = gh1_arr[ii] + factor * (gh2_arr[ii] - gh1_arr[ii]);
      }
      break;
    case 4:
      for (ii = 1; ii <= k; ++ii) {
        ghsv_arr[ii] = gh1_arr[ii] + factor * (gh2_arr[ii] - gh1_arr[ii]);
      }
      break;
    default:
      std::cout << "\nError in subroutine extrapsh" << std::endl;
      break;
  }

}

/****************************************************************************/
/*                                                                          */
/*                           Subroutine extrapsh                            */
/*                                                                          */
/****************************************************************************/
/*                                                                          */
/*     Extrapolates linearly a spherical harmonic model with a              */
/*     rate-of-change model.                                                */
/*                                                                          */
/*     Input:                                                               */
/*           date     - date of resulting model (in decimal year)           */
/*           epoch1_     - date of base model                                  */
/*           nmain1_    - maximum degree and order of base model              */
/*           gh1      - Schmidt quasi-normal internal spherical             */
/*                      harmonic coefficients of base model                 */
/*           nmax2    - maximum degree and order of rate-of-change model    */
/*           gh2      - Schmidt quasi-normal internal spherical             */
/*                      harmonic coefficients of rate-of-change model       */
/*                                                                          */
/*     Output:                                                              */
/*           gha or b - Schmidt quasi-normal internal spherical             */
/*                    harmonic coefficients                                 */
/*           nmax   - maximum degree and order of resulting model           */
/*                                                                          */
/*     FORTRAN                                                              */
/*           A. Zunde                                                       */
/*           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225     */
/*                                                                          */
/*     C                                                                    */
/*           C. H. Shaffer                                                  */
/*           Lockheed Missiles and Space Company, Sunnyvale CA              */
/*           August 16, 1988                                                */
/*                                                                          */
/****************************************************************************/
void IGRF::extrapsh(double date, int gh) {
  // int nmax;
  int k, l;
  int ii;
  double factor;

  factor = date - epoch1_;
  if (nmain1_ == nsv1_) {
    k = nmain1_ * (nmain1_ + 2);
    nmax_ = nmain1_;
  } else {
    if (nmain1_ > nsv1_) {
      k = nsv1_ * (nsv1_ + 2);
      l = nmain1_ * (nmain1_ + 2);
      switch (gh) {
        case 3:
          for (ii = k + 1; ii <= l; ++ii) {
            gh_arr[ii] = gh1_arr[ii];
          }
          break;
        case 4:
          for (ii = k + 1; ii <= l; ++ii) {
            ghsv_arr[ii] = gh1_arr[ii];
          }
          break;
        default:
          std::cout << "\nError in subroutine extrapsh" << std::endl;
          break;
      }
      nmax_ = nmain1_;
    } else {
      k = nmain1_ * (nmain1_ + 2);
      l = nsv1_ * (nsv1_ + 2);
      switch (gh) {
        case 3:
          for (ii = k + 1; ii <= l; ++ii) {
            gh_arr[ii] = factor * ghsv1_arr[ii];
          }
          break;
        case 4:
          for (ii = k + 1; ii <= l; ++ii) {
            ghsv_arr[ii] = factor * ghsv1_arr[ii];
          }
          break;
        default:
          std::cout << "\nError in subroutine extrapsh" << std::endl;
          break;
      }
      nmax_ = nsv1_;
    }
  }
  switch (gh) {
    case 3:
      for (ii = 1; ii <= k; ++ii) {
        gh_arr[ii] = gh1_arr[ii] + factor * ghsv1_arr[ii];
      }
      break;
    case 4:
      for (ii = 1; ii <= k; ++ii) {
        ghsv_arr[ii] = gh1_arr[ii] + factor * ghsv1_arr[ii];
      }
      break;
    default:
      std::cout << "\nError in subroutine extrapsh" << std::endl;
      break;
  }
}

/****************************************************************************/
/*                                                                          */
/*                           Subroutine shval3                              */
/*                                                                          */
/****************************************************************************/
/*                                                                          */
/*     Calculates field components from spherical harmonic (sh)             */
/*     models.                                                              */
/*                                                                          */
/*     Input:                                                               */
/*           igdgc     - indicates coordinate system used; set equal        */
/*                       to 1 if geodetic, 2 if geocentric                  */
/*           latitude  - north latitude, in degrees                         */
/*           longitude - east longitude, in degrees                         */
/*           elev      - WGS84 altitude above ellipsoid (igdgc=1), or       */
/*                       radial distance from earth's center (igdgc=2)      */
/*           a2,b2     - squares of semi-major and semi-minor axes of       */
/*                       the reference spheroid used for transforming       */
/*                       between geodetic and geocentric coordinates        */
/*                       or components                                      */
/*           nmax      - maximum degree and order of coefficients           */
/*           iext      - external coefficients flag (=0 if none)            */
/*           ext1,2,3  - the three 1st-degree external coefficients         */
/*                       (not used if iext = 0)                             */
/*                                                                          */
/*     Output:                                                              */
/*           x         - northward component                                */
/*           y         - eastward component                                 */
/*           z         - vertically-downward component                      */
/*                                                                          */
/*     based on subroutine 'igrf' by D. R. Barraclough and S. R. C. Malin,  */
/*     report no. 71/1, institute of geological sciences, U.K.              */
/*                                                                          */
/*     FORTRAN                                                              */
/*           Norman W. Peddie                                               */
/*           USGS, MS 964, box 25046 Federal Center, Denver, CO.  80225     */
/*                                                                          */
/*     C                                                                    */
/*           C. H. Shaffer                                                  */
/*           Lockheed Missiles and Space Company, Sunnyvale CA              */
/*           August 17, 1988                                                */
/*                                                                          */
/****************************************************************************/
void IGRF::shval3(int igdgc, double flat, double flon, double elev, int gh,
                  int iext, int ext1, int ext2, int ext3) {
  double earths_radius = 6371.2;
  double dtr = 0.01745329;
  double slat;
  double clat;
  double ratio;
  double aa, bb, cc, dd;
  double sd;
  double cd;
  double r;
  double a2;
  double b2;
  double rr;
  double fm, fn;
  double sl[14];
  double cl[14];
  double p[119];
  double q[119];
  int ii, j, k, l, m, n;
  int npq;
  int ios;
  double argument;
  double power;
  a2 = 40680631.59; /* WGS84 */
  b2 = 40408299.98; /* WGS84 */
  ios = 0;
  r = elev;
  argument = flat * dtr;
  slat = sin(argument);
  if ((90.0 - flat) < 0.001) {
    aa = 89.999; /*  300 ft. from North pole  */
  } else {
    if ((90.0 + flat) < 0.001) {
      aa = -89.999; /*  300 ft. from South pole  */
    } else {
      aa = flat;
    }
  }
  argument = aa * dtr;
  clat = cos(argument);
  argument = flon * dtr;
  sl[1] = sin(argument);
  cl[1] = cos(argument);
  switch (gh) {
    case 3:
      bfield_.x = 0;
      bfield_.y = 0;
      bfield_.z = 0;
      break;
    case 4:
      bfield_temp_.x = 0;
      bfield_temp_.y = 0;
      bfield_temp_.z = 0;
      break;
    default:
      std::cout << "\nError in subroutine shval3" << std::endl;
      break;
  }
  sd = 0.0;
  cd = 1.0;
  l = 1;
  n = 0;
  m = 1;
  npq = (nmax_ * (nmax_ + 3)) / 2;
  if (igdgc == 1) {
    aa = a2 * clat * clat;
    bb = b2 * slat * slat;
    cc = aa + bb;
    argument = cc;
    dd = sqrt(argument);
    argument = elev * (elev + 2.0 * dd) + (a2 * aa + b2 * bb) / cc;
    r = sqrt(argument);
    cd = (elev + dd) / r;
    sd = (a2 - b2) / dd * slat * clat / r;
    aa = slat;
    slat = slat * cd - clat * sd;
    clat = clat * cd + aa * sd;
  }
  ratio = earths_radius / r;
  argument = 3.0;
  aa = sqrt(argument);
  p[1] = 2.0 * slat;
  p[2] = 2.0 * clat;
  p[3] = 4.5 * slat * slat - 1.5;
  p[4] = 3.0 * aa * clat * slat;
  q[1] = -clat;
  q[2] = slat;
  q[3] = -3.0 * clat * slat;
  q[4] = aa * (slat * slat - clat * clat);
  for (k = 1; k <= npq; ++k) {
    if (n < m) {
      m = 0;
      n = n + 1;
      argument = ratio;
      power = n + 2;
      rr = pow(argument, power);
      fn = n;
    }
    fm = m;
    if (k >= 5) {
      if (m == n) {
        argument = (1.0 - 0.5 / fm);
        aa = sqrt(argument);
        j = k - n - 1;
        p[k] = (1.0 + 1.0 / fm) * aa * clat * p[j];
        q[k] = aa * (clat * q[j] + slat / fm * p[j]);
        sl[m] = sl[m - 1] * cl[1] + cl[m - 1] * sl[1];
        cl[m] = cl[m - 1] * cl[1] - sl[m - 1] * sl[1];
      } else {
        argument = fn * fn - fm * fm;
        aa = sqrt(argument);
        argument = ((fn - 1.0) * (fn - 1.0)) - (fm * fm);
        bb = sqrt(argument) / aa;
        cc = (2.0 * fn - 1.0) / aa;
        ii = k - n;
        j = k - 2 * n + 1;
        p[k] = (fn + 1.0) * (cc * slat / fn * p[ii] - bb / (fn - 1.0) * p[j]);
        q[k] = cc * (slat * q[ii] - clat / fn * p[ii]) - bb * q[j];
      }
    }
    switch (gh) {
      case 3:
        aa = rr * gh_arr[l];
        break;
      case 4:
        aa = rr * ghsv_arr[l];
        break;
      default:
        std::cout << "\nError in subroutine shval3" << std::endl;
        break;
    }
    if (m == 0) {
      switch (gh) {
        case 3:
          bfield_.x += aa * q[k];
          bfield_.z -= aa * p[k];
          break;
        case 4:
          bfield_temp_.x += aa * q[k];
          bfield_temp_.z -= aa * p[k];
          break;
        default:
          std::cout << "\nError in subroutine shval3" << std::endl;
          break;
      }
      l = l + 1;
    } else {
      switch (gh) {
        case 3:
          bb = rr * gh_arr[l + 1];
          cc = aa * cl[m] + bb * sl[m];
          bfield_.x += cc * q[k];
          bfield_.z -= cc * p[k];
          if (clat > 0) {
            bfield_.y +=
                (aa * sl[m] - bb * cl[m]) * fm * p[k] / ((fn + 1.0) * clat);
          } else {
            bfield_.y += (aa * sl[m] - bb * cl[m]) * q[k] * slat;
          }
          l = l + 2;
          break;
        case 4:
          bb = rr * ghsv_arr[l + 1];
          cc = aa * cl[m] + bb * sl[m];
          bfield_temp_.x += cc * q[k];
          bfield_temp_.z -= cc * p[k];
          if (clat > 0) {
            bfield_temp_.y +=
                (aa * sl[m] - bb * cl[m]) * fm * p[k] / ((fn + 1.0) * clat);
          } else {
            bfield_temp_.y += (aa * sl[m] - bb * cl[m]) * q[k] * slat;
          }
          l = l + 2;
          break;
        default:
          std::cout << "\nError in subroutine shval3" << std::endl;
          break;
      }
    }
    m = m + 1;
  }
  if (iext != 0) {
    aa = ext2 * cl[1] + ext3 * sl[1];
    switch (gh) {
      case 3:
        bfield_.x -= ext1 * clat + aa * slat;
        bfield_.y += ext2 * sl[1] - ext3 * cl[1];
        bfield_.z += ext1 * slat + aa * clat;
        break;
      case 4:
        bfield_temp_.x -= ext1 * clat + aa * slat;
        bfield_temp_.y += ext2 * sl[1] - ext3 * cl[1];
        bfield_temp_.z += ext1 * slat + aa * clat;
        break;
      default:
        std::cout << "\nError in subroutine shval3" << std::endl;
        break;
    }
  }
  switch (gh) {
    case 3:
      aa = bfield_.x;
      bfield_.x = bfield_.x * cd + bfield_.z * sd;
      bfield_.z = bfield_.z * cd - aa * sd;
      break;
    case 4:
      aa = bfield_temp_.x;
      bfield_temp_.x = bfield_temp_.x * cd + bfield_temp_.z * sd;
      bfield_temp_.z = bfield_temp_.z * cd - aa * sd;
      break;
    default:
      std::cout << "\nError in subroutine shval3" << std::endl;
      break;
  }
}

std::array<double, 3> IGRF::values(const double& r, const double& theta,
                                   const double& phi) {
  // convert to lat, long, since shval3 is in this format as well
  // latitude is the north latitude in degrees (+ is north)
  // longitude is east latitude in degrees (+ is east)
  // elev is radius from Earth's center
  double elev = r * (1e-3);
  double lat = 90. - (theta * constants::RAD_TO_DEG);
  double lng = phi * constants::RAD_TO_DEG;

  int gh = 3;  // gh=3 is for the main field coefficents (4 for sv)

  // get the x, y, z components of the bfield
  // igdgc = 2 for geocentric, which is what we want
  shval3(2, lat, lng, elev, gh);

  // Transform the NED (North-East-Down) Cartesian field components returned by
  // shval3() into geocentric spherical components (Br, Btheta, Bphi).
  //
  // shval3 with igdgc=2 (geocentric) fills:
  //   bfield_.x = northward component  (B_N)
  //   bfield_.y = eastward component   (B_E)
  //   bfield_.z = vertically-downward  (B_D)
  //
  // In geocentric spherical coordinates (r outward, theta = colatitude,
  // phi = longitude eastward):
  //   B_r     = -B_D   (outward is opposite to "down")
  //   B_theta = -B_N   (theta increases southward, opposite to north)
  //   B_phi   =  B_E   (phi increases eastward, same direction)
  std::array<double, 3> values;

  // shval3 fills bfield_ in nanotesla (nT); convert to Tesla to match
  // MagneticField::values() and the SI units expected by TrajectoryTracer.
  constexpr double NT_TO_T = 1e-9;
  values[0] = -bfield_.z * NT_TO_T;   // Br     = -B_down
  values[1] = -bfield_.x * NT_TO_T;   // Btheta = -B_north
  values[2] =  bfield_.y * NT_TO_T;   // Bphi   =  B_east

  return values;
}

// get the declination, inclination, horizontal intensity, and
// total intensity from field values X, Y, Z
// std::array<double, 4> dihf(int gh);
// {
//   int ios;
//   int j;
//   double sn;
//   double h2;
//   double hpx;
//   double argument, argument2;

//   ios = gh;
//   sn = 0.0001;

//   switch (gh) {
//     case 3:
//       for (j = 1; j <= 1; ++j) {
//         h2 = x * x + y * y;
//         argument = h2;
//         h = sqrt(argument); /* calculate horizontal intensity */
//         argument = h2 + z * z;
//         f = sqrt(argument); /* calculate total intensity */
//         if (f < sn) {
//           d = NaN; /* If d and i cannot be determined, */
//           i = NaN; /*       set equal to NaN         */
//         } else {
//           argument = z;
//           argument2 = h;
//           i = atan2(argument, argument2);
//           if (h < sn) {
//             d = NaN;
//           } else {
//             hpx = h + x;
//             if (hpx < sn) {
//               d = PI;
//             } else {
//               argument = y;
//               argument2 = hpx;
//               d = 2.0 * atan2(argument, argument2);
//             }
//           }
//         }
//       }
//       break;
//     case 4:
//       for (j = 1; j <= 1; ++j) {
//         h2 = xtemp * xtemp + ytemp * ytemp;
//         argument = h2;
//         htemp = sqrt(argument);
//         argument = h2 + ztemp * ztemp;
//         ftemp = sqrt(argument);
//         if (ftemp < sn) {
//           dtemp = NaN; /* If d and i cannot be determined, */
//           itemp = NaN; /*       set equal to 999.0         */
//         } else {
//           argument = ztemp;
//           argument2 = htemp;
//           itemp = atan2(argument, argument2);
//           if (htemp < sn) {
//             dtemp = NaN;
//           } else {
//             hpx = htemp + xtemp;
//             if (hpx < sn) {
//               dtemp = PI;
//             } else {
//               argument = ytemp;
//               argument2 = hpx;
//               dtemp = 2.0 * atan2(argument, argument2);
//             }
//           }
//         }
//       }
//       break;
//     default:
//       std::cout << "\nError in subroutine dihf" << std::endl;
//       break;
//   }
//   return (ios);
// }