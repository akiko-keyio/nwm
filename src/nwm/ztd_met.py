import numpy as np


def zwd_aske_and_nordius(e, Tm, lamb, gm):
    """
    This function determines the zenith wet delay based on the
    equation 22 by Aske and Nordius (1987).

    Reference:
    Askne and Nordius, Estimation of tropospheric delay for microwaves from
    surface weather datas, Radio Science, Vol 22(3): 379-386, 1987.

    Input parameters:
    e:      water vapor pressure in hPa
    Tm:     mean temperature in Kelvin
    lamb:   water vapor lapse rate (see definition in Askne and Nordius 1987)

    Output parameters:
    zwd:  zenith wet delay in meter

    Example 1:

    e =  10.9621 hPa
    Tm = 273.8720
    lamb = 2.8071

    Output:
    zwd = 0.1176 m

    Johannes Boehm, 3 August 2013
    """

    # coefficient
    k1 = 77.604  # K/hPa
    k2 = 64.79  # K/hPa
    k2p = k2 - k1 * 18.0152 / 28.9644  # K/hPa
    k3 = 377600  # KK/hPa

    # molar mass of dry air in kg/mol
    dMtr = 28.965 * 10**-3
    # universal gas constant in J/K/mol
    R = 8.3143
    # mean gravity in m/s**2
    if gm is None:
        gm = 9.80665

    # specific gas constant for dry consituents
    Rd = R / dMtr  #

    zwd = 1e-6 * (k2p + k3 / Tm) * Rd / (lamb + 1) / gm * e

    return zwd


def zwd_saastamoinen(e, t):
    return 0.0022768 * (1255.0 / t + 0.05) * e


def zhd_saastamoinen(p, lat, alt):
    dlat = np.radians(lat)
    zhd = 0.0022768 * p / (1.0 - 0.00266 * np.cos(2 * dlat) - 0.00028 * alt * 1.0e-03)
    return zhd
