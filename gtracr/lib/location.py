"""
Library class that allows us to add locations of varying latitude and longitude
"""


class Location:
    """
    A class of locations around the globe. Used to allow easy access to geodesic coordinates of locations of interest.
    Members:
    - name : str
        the name of the location
    - latitude : float
        the geographical latitude (0 = equator) in decimal degrees
    - longitude : float
        the geographical longitude (0 = prime meridian) in decimal degrees
    - altitude : float
        the altitude above sea level of the location in km
    """

    def __init__(self, name, latitude, longitude, altitude=0.0):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

    def __str__(self):
        return f"{self.name} : Latitude : {self.latitude}, Longitude : {self.longitude}, Elevation : {self.altitude}"


if __name__ == "__main__":
    icecube = Location("IceCube", 89.99, -63.453056, 0.0)
    print(icecube)
