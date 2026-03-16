/*
Location.h

A class of locations around the globe. Used to allow easy access to geodesic
coordinates of locations of interest. Members:
- name : the name of the location
- latitude : the geographical latitude (0 = equator) in decimal degrees
- longitude : the geographical longitude (0 = prime meridian) in decimal degrees
- altitude : the altitude above sea level of the location in km
*/
#ifndef __LOCATION_HPP_
#define __LOCATION_HPP_

class Location {
  // members of Location class
 private:
  std::string ln;  // Location name
  double lat;      // Latitude
  double lng;      // Longitude
  double alt;      // Altitude
 public:
  // Default Constructor
  // The default constructor is set with latitude, longitude, and altitude = 0.
  // The name correlating to this location is called "Null Island", and thus the
  // name in this default constructor is "Null Island".
  Location();
  // Constructor
  // This constructs a Location object of any arbitrary latitude, longitude, and
  // altitude as inputs. The name of the location is set to be "Default Name",
  // for no other purposes other than to provide some input to the name
  // variable.
  Location(const double& latitude, const double& longitude, const double& altitude);
  // Constructor
  // This constructs a Location object of any arbitrary latitude, longitude, and
  // altitude at some known location, provided with the variable name
  Location(const std::string& name, const double& latitude, const double& longitude,
           const double& altitude);  // with location name
  // destructors
  ~Location() {}
  // Copy constructor / operator
  Location(const Location&);
  Location& operator=(const Location&);
  // get the name of the location
  const std::string& name() const { return ln; }
  // get the latitude of the location
  const double& latitude() const { return lat; }
  // get the longitude of the location
  const double& longitude() const { return lng; }
  // get the altitude of the location
  const double& altitude() const { return alt; }
  // print the name and the geodesic coordinates of the location
  void print();
};

#endif  //__LOCATION_HPP_