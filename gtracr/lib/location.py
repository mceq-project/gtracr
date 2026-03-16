"""
Library class that allows us to add locations of varying latitude and longitude
"""


class Location:
    """
    Geographic detector location.

    Parameters
    ----------
    name : str
        Human-readable location name (e.g. ``"Kamioka"``).
    latitude : float
        Geographic latitude in decimal degrees (positive = north).
    longitude : float
        Geographic longitude in decimal degrees (positive = east).
    altitude : float, optional
        Altitude above sea level in km (default 0).
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
