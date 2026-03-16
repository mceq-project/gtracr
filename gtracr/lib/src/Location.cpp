
#include "Location.hpp"

#include <iostream>
#include <string>

// Constructor
// Default constructor at 0, 0
Location::Location() : ln{"Null Island"}, lat{0.}, lng{0.}, alt{0.} {}

// Constructor with variable altitude
Location::Location(const double& latitude, const double& longitude, const double& altitude)
    : ln{"DEFAULT"}, lat{latitude}, lng{longitude}, alt{altitude} {}

// Constructor with default altitude
Location::Location(const std::string& name, const double& latitude, const double& longitude,
                   const double& altitude = 0.)
    : ln{name}, lat{latitude}, lng{longitude}, alt{altitude} {}

// copy constructor
Location::Location(const Location& loc) : ln{loc.ln}, lat{loc.lat}, lng{loc.lng}, alt{loc.alt} {}

// Copy assignment operator
Location& Location::operator=(const Location& loc) {
  ln = loc.ln;
  lat = loc.lat;
  lng = loc.lng;
  alt = loc.alt;
  return *this;
}

// print contents
void Location::print() {
  std::cout << "Location Name: " << ln << ", "
            << "Latitude: " << lat << ", "
            << "Longitude: " << lng << ", "
            << "Altitude: " << alt << std::endl;
}
