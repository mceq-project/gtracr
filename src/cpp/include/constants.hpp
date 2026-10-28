// contains relavant constants used within the package

#ifndef __CONSTANTS_HPP_
#define __CONSTANTS_HPP_

/*
Namespace that contains the relavant constants used within
our code.
*/
namespace constants {
constexpr double SPEED_OF_LIGHT = 299792458.;          // speed of light in vacuum
constexpr double ELEMENTARY_CHARGE = 1.602176634e-19;  // elementary charge in coulombs (SI, exact)
constexpr double RE = 6371.2 * (1e3);                  // radius of the earth in meters
constexpr double pi = 3.141592653589793238462643383279502884;  // pi
constexpr double DEG_TO_RAD = pi / (180.);
constexpr double RAD_TO_DEG = 180. / pi;
// 1 GeV/c^2 = 1e9 e / c^2 kg (exact in SI)
constexpr double KG_PER_GEVC2 = 1e9 * ELEMENTARY_CHARGE / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
// 1 GeV/c = 1e9 e / c kg m/s (exact in SI); rigidity R [GV] = p [GeV/c] / |Z|
constexpr double KG_M_S_PER_GEVC = 1e9 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT;
}  // namespace constants

#endif  //__CONSTANTS_HPP_