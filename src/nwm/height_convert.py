import numpy as np
from metpy.units import units

# Constants
G_EQUAT = 9.7803253359 * units.meters / units.seconds**2  # equatorial gravity (m/s^2)
K_SOMIG = 1.931853e-3  # Somagliana's constant
ECC = 0.081819  # eccentricity
DEG2RAD = 0.0174532925  # degrees to radians
A_EARTH = 6378137.0 * units.meters  # semi-major axis (m)
FLATT = 0.003352811  # flattening
GM_RATIO = 0.003449787  # gravitational ratio
G_WMO = 9.80665 * units.meters / units.seconds**2  # reference gravity (m/s^2)


# M.J. Mahoney. 2001. A discuss of various measures of altitudes


def calculate_gravity(lat, h=None):
    """
    Calculate the gravity at given latitudes and optional heights.

    Parameters:
    lat (array-like): Latitudes in degrees.
    h (array-like, optional): Heights in meters. Default is None.

    Returns:
    np.ndarray: Gravity in m/s^2.
    """
    lat = np.asarray(lat)
    sinlat2 = np.sin(lat * DEG2RAD) ** 2
    g = G_EQUAT * (1.0 + K_SOMIG * sinlat2) / np.sqrt(1.0 - (ECC**2) * sinlat2)

    if h is not None:
        h = np.asarray(h)
        R_lat = calculate_R_eff(lat)
        g = g * (R_lat**2) / ((R_lat + h) ** 2)

    return g


def calculate_R_eff(lat):
    """
    Calculate the effective Earth radius at given latitudes.

    Parameters:
    lat (array-like): Latitudes in degrees.

    Returns:
    np.ndarray: Effective Earth radius in meters.
    """
    lat = np.asarray(lat)
    sinlat2 = np.sin(lat * DEG2RAD) ** 2
    R_eff = A_EARTH / (1.0 + FLATT + GM_RATIO - 2.0 * FLATT * sinlat2)
    return R_eff


def _geopotential_to_geometric(latitude, geopotential_height):
    """
    Convert geopotential height to geometric height.

    Parameters:
    lat (array-like): Latitudes in degrees.
    geop (array-like): Geopotential height in meters.

    Returns:
    np.ndarray: Geometric heights in meters.
    """
    # latitude = np.asarray(latitude)
    # geopotential_height = np.asarray(geopotential_height)

    g_lat = calculate_gravity(latitude)
    R_lat = calculate_R_eff(latitude)

    # h = np.where(
    #     geopotential_height > -9999.0,
    #     R_lat * geopotential_height / (g_lat / G_WMO * R_lat - geopotential_height),
    #     -99999000.0
    # )

    return R_lat * geopotential_height / (g_lat / G_WMO * R_lat - geopotential_height)


def geopotential_to_geometric(latitude, geopotential_height):
    """
    Convert geopotential height to geometric height.

    Parameters:
    lat (array-like): Latitudes in degrees.
    geop (array-like): Geopotential height in meters.

    Returns:
    np.ndarray: Geometric heights in meters.
    """
    # latitude = np.asarray(latitude)
    # geopotential_height = np.asarray(geopotential_height)

    g_lat = calculate_gravity(latitude)
    R_lat = calculate_R_eff(latitude)

    # h = np.where(
    #     geopotential_height > -9999.0,
    #     R_lat * geopotential_height / (g_lat / G_WMO * R_lat - geopotential_height),
    #     -99999000.0
    # )

    return R_lat * geopotential_height / (g_lat / G_WMO * R_lat - geopotential_height)


if __name__ == "__main__":
    latitudes = np.array([45.0, 50.0])  # Latitudes in degrees
    geopotential_heights = (
        np.array([5000.0, 6000.0]) * units.meters
    )  # Geopotential heights in meters

    gravity_values = calculate_gravity(latitudes)
    R_lat_values = calculate_R_eff(latitudes)
    geometric_heights = geopotential_to_geometric(latitudes, geopotential_heights)

    print(f"Gravity at latitudes {latitudes}: {gravity_values} m/s^2\n")
    print(f"Effective Earth radius at latitudes {latitudes}: {R_lat_values} m\n")
    print(
        f"Geometric heights for geopotential heights {geopotential_heights} at latitudes {latitudes}: {geometric_heights} m"
    )
