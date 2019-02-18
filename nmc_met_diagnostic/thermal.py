# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Calculate thermal parameters.
"""

import numpy as np
from nmc_met_base import arr, constants, grid

Cp = constants.cp
Cv = constants.cv
Rd = constants.rd
Rv = constants.rv
RvRd = Rv / Rd
g = constants.g0
L = constants.Lv
Lf = constants.Lf
Talt = constants.Talt
Tfrez = constants.Tfrez
To = constants.T0
Po = constants.P0
Pr = constants.Pr
lapsesta = constants.lapsesta
kappa = constants.kappa
epsil = constants.epsil
pi = constants.pi
pid = pi/180
R_earth = constants.Re
omeg_e = constants.omega
eo = constants.eo
missval = -9999
eps = 2.2204e-16


def pottemp(temp, lev, zdim, punit=100., p0=100000.):
    """
    Calculate potential temperature.

    :param temp: array_like, temperature [K].
    :param lev: array_like, level [punit*Pa].
    :param zdim: vertical dimensions.
    :param punit: pressure level punit.
    :param p0: reference pressure.
    :return: ndarray.
    """

    temp = np.asarray(temp)
    ndim = temp.ndim
    p = arr.expand(lev, ndim, axis=zdim)

    out = temp * ((p0/p/punit)**kappa)

    return out


def thetae(thta, temp, qv):
    """
    Compute equivalent potential temperature

    :param thta: Input potential temperature of column (K)
    :param temp: Input temperature (K) at LCL
    :param qv: Input mixing ratio of column (kg kg-1)
    :return: Output equivalent potential temperature (K)
    """

    thout = thta * np.exp((L * qv) / (Cp * temp))
    return thout


def temp_to_theta(temp, pres, p0=100000.):
    """
    Compute potential temperature.

    :param temp: Input temperature (K)
    :param pres: Input pressure (Pa)
    :param p0: reference pressure (Pa)
    :return: potential temperature (K)
    """

    return temp * (p0 / pres) ** 0.286


def theta_to_temp(theta, pres, p0=100000.):
    """
    Compute temperature.

    :param theta: Input potential temperature (K)
    :param pres: Input pressure (Pa)
    :param p0: reference pressure (Pa)
    :return: Output temperature (K)
    """

    return theta * (pres / p0) ** 0.286


def td_to_mixrat(tdew, pres):
    """
    Convert from dew point temperature to water vapor mixing ratio.

    :param tdew: Input dew point temperature (K)
    :param pres:  Input pressure (Pa)
    :return: Output water vapor mixing ratio (kg/kg)
    """

    pres = pres / 100
    mixrat = eo / (pres * RvRd) * np.exp((L / Rv) * ((1 / Tfrez) - (1 / tdew)))
    return mixrat


def mixrat_to_td(qvap, pres):
    """
    Convert from water vapor mixing ratio to dewpoint temperature.

    :param qvap: Input water vapor mixing ratio (kg/kg)
    :param pres: Input pressure (Pa)
    :return: Output dewpoint temperature (K)
    """

    pres = pres / 100.
    evap = qvap * pres * RvRd
    tdew = 1 / ((1 / Tfrez) - (Rv / L) * np.log(evap / eo))
    return tdew


def spechum_to_td(spechum, pres):
    """
    Convert from specific humidity to dewpoint temperature

    :param spechum: Input specific humidity in (kg/kg)
    :param pres: Input pressure (Pa)
    :return: Output dewpoint temperature (K)
    """

    qvap = (spechum / (1 - spechum))
    pres = pres / 100
    evap = qvap * pres * RvRd
    tdew = 1 / ((1 / Tfrez) - (Rv / L) * np.log(evap / eo))
    return tdew


def claus_clap(temp):
    """
    Compute saturation vapor pressure

    :param temp: Input temperature (K)
    :return:  Output satuation vapor pressure (Pa)
    """

    esat = (eo * np.exp((L / Rv) * (1.0 / Tfrez - 1 / temp))) * 100.
    return esat


def claus_clap_ice(temp):
    """
    Compute saturation vapor pressure over ice

    :param temp: Input temperature (K)
    :return: Output satuation vapor pressure of ice (Pa)
    """

    a = 273.16 / temp
    exponent = -9.09718 * (a - 1.) - 3.56654 * np.log10(a) + \
        0.876793 * (1. - 1. / a) + np.log10(6.1071)
    esi = 10 ** exponent
    esi = esi * 100
    return esi


def sat_vap(temp):
    """
    Compute saturation vapor pressure

    :param temp: Input temperature (K)
    :return: Output satuation vapor pressure (Pa)
    """

    [iinds] = np.where(temp < 273.15)
    [linds] = np.where(temp >= 273.15)
    esat = np.zeros_like(temp).astype('f')

    nice = len(iinds)
    nliq = len(linds)

    tempc = temp - 273.15
    if nliq > 1:
        esat[linds] = 6.112 * np.exp(17.67 * tempc[linds] / (
            tempc[linds] + 243.12)) * 100.
    else:
        if nliq > 0:
            esat = 6.112 * np.exp(17.67 * tempc / (tempc + 243.12)) * 100.
    if nice > 1:
        esat[iinds] = 6.112 * np.exp(22.46 * tempc[iinds] / (
            tempc[iinds] + 272.62)) * 100.
    else:
        if nice > 0:
            esat = 6.112 * np.exp(22.46 * tempc / (tempc + 272.62)) * 100.
    return esat


def moist_lapse(ws, temp):
    """
    Compute moist adiabatic lapse rate

    :param ws: Input saturation mixing ratio (kg kg-1)
    :param temp: Input air temperature (K)
    :return: Output moist adiabatic lapse rate
    """

    return (g / Cp) * ((1.0 + L * ws) / (Rd * temp)) / (
        1.0 + (ws * (L ** 2.0) / (Cp * Rv * temp ** 2.0)))


def satur_mix_ratio(es, pres):
    """
    Compute saturation mixing ratio

    :param es: Input saturation vapor pressure (Pa)
    :param pres:  Input air pressure (Pa)
    :return: Output saturation mixing ratio
    """

    ws = 0.622 * (es / (pres - es))
    return ws


def virtual_temp_from_mixr(tempk, mixr):
    """
    Virtual Temperature

    :param tempk: Temperature (K)
    :param mixr: Mixing Ratio (kg/kg)
    :return:  Virtual temperature (K)
    """

    return tempk * (1.0 + 0.6 * mixr)


def latentc(tempk):
    """
    Latent heat of condensation (vapourisation)
    http://en.wikipedia.org/wiki/Latent_heat#Latent_heat_for_condensation_of_water

    :param tempk: Temperature (K)
    :return: L_w (J/kg)
    """

    tempc = tempk - 273.15
    return 1000 * (
        2500.8 - 2.36 * tempc + 0.0016 * tempc ** 2 -
        0.00006 * tempc ** 3)


def gamma_w(tempk, pres, e=None):
    """
    Function to calculate the moist adiabatic lapse rate (deg K/Pa) based
    on the temperature, pressure, and rh of the environment.

    :param tempk: Temperature (K)
    :param pres: Input pressure (Pa)
    :param e: Input saturation vapor pressure (Pa)
    :return: The moist adiabatic lapse rate (Dec K/Pa)
    """

    es = sat_vap(tempk)
    ws = satur_mix_ratio(es, pres)

    if e is None:
        # assume saturated
        e = es

    w = satur_mix_ratio(e, pres)

    tempv = virtual_temp_from_mixr(tempk, w)
    latent = latentc(tempk)

    A = 1.0 + latent * ws / (Rd * tempk)
    B = 1.0 + epsil * latent * latent * ws / (Cp * Rd * tempk * tempk)
    Rho = pres / (Rd * tempv)
    gamma = (A / B) / (Cp * Rho)
    return gamma


def dry_parcel_ascent(startpp, starttk, starttdewk, nsteps=101):
    """
    Lift a parcel dry adiabatically from startp to LCL.

    :param startpp: Pressure of parcel to lift in Pa
    :param starttk: Temperature of parcel at startp in K
    :param starttdewk: Dewpoint temperature of parcel at startp in K
    :param nsteps:
    :return: presdry, tempdry, pressure (Pa) and temperature (K) along
             dry adiabatic ascent of parcel
             tempiso is in K
             T_lcl, P_lcl, Temperature and pressure at LCL
    """

    assert starttdewk <= starttk

    startt = starttk - 273.15
    starttdew = starttdewk - 273.15
    startp = startpp / 100.

    if starttdew == startt:
        return np.array([startp]), np.array([startt]), np.array([starttdew]),

    Pres = np.logspace(np.log10(startp), np.log10(600), nsteps)

    # Lift the dry parcel
    T_dry = (starttk * (Pres / startp) ** (Rd / Cp)) - 273.15

    # Mixing ratio isopleth
    starte = sat_vap(starttdewk)
    startw = satur_mix_ratio(starte, startpp)
    ee = Pres * startw / (.622 + startw)
    T_iso = 243.5 / (17.67 / np.log(ee / 6.112) - 1.0)

    # Solve for the intersection of these lines (LCL).
    # interp requires the x argument (argument 2)
    # to be ascending in order!
    P_lcl = np.interp(0, T_iso - T_dry, Pres)
    T_lcl = np.interp(P_lcl, Pres[::-1], T_dry[::-1])

    presdry = np.linspace(startp, P_lcl)
    tempdry = np.interp(presdry, Pres[::-1], T_dry[::-1])
    tempiso = np.interp(presdry, Pres[::-1], T_iso[::-1])

    return (
        presdry * 100., tempdry + 273.15,
        tempiso + 273.15, T_lcl + 273.15, P_lcl * 100.)


def moist_ascent(startpp, starttk, ptop=100, nsteps=501):
    """
    Lift a parcel moist adiabatically from startp to endp.

    :param startpp: Pressure of parcel to lift in Pa
    :param starttk: Temperature of parcel at startp in K
    :param ptop: Top pressure of parcel to lift in Pa
    :param nsteps:
    :return:
    """

    startp = startpp / 100.  # convert to hPa
    startt = starttk - 273.15  # convert to deg C

    preswet = np.logspace(np.log10(startp), np.log10(ptop), nsteps)

    temp = startt
    tempwet = np.zeros(preswet.shape)
    tempwet[0] = startt
    for ii in range(preswet.shape[0] - 1):
        delp = preswet[ii] - preswet[ii + 1]
        temp = temp - 100. * delp * gamma_w(
            temp + 273.15, (preswet[ii] - delp / 2) * 100.)
        tempwet[ii + 1] = temp

    return preswet * 100., tempwet + 273.15


def stability(temp, lev, zdim, punit=100.):
    """
    P level coordinates stability (Brunt-Vaisala).

    :param temp: array_like, temperature.
    :param lev: array_like, pressure level.
    :param zdim: vertical dimension axis.
    :param punit: pressure unit.
    :return: ndarray.
    """

    temp = np.asarray(temp)
    ndim = temp.ndim
    p = arr.expand(lev, ndim, axis=zdim) * punit
    theta = pottemp(temp, lev, zdim, punit=punit)
    alpha = Rd * temp / p
    N = -alpha * grid.dvardp(np.log(theta), lev, zdim, punit=punit)

    return N


def atmosphere(alt):
    """python-standard-atmosphere
    Python package for creating pressure and temperature profiles of
    the standard atmosphere for use with geophysical models. This
    package will only calcualate good values up to 86km.
    https://github.com/pcase13/python-standard-atmosphere/blob/master/standard.py
    Arguments:
        alt {scalar} -- altitude, hPa

    Returns:
        scalar -- standard-atmosphere
    """

    # Constants
    REARTH = 6369.0  # radius of earth
    GMR = 34.163195  # hydrostatic constant
    NTAB = 8  # number of entries in defining tables

    # Define defining tables
    htab = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852]
    ttab = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
    ptab = [
        1.0, 2.233611E-1, 5.403295E-2, 8.5666784E-3, 1.0945601E-3,
        6.6063531E-4, 3.9046834E-5, 3.68501E-6]
    gtab = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0]

    # Calculate
    h = alt*REARTH/(alt+REARTH)  # convert to geopotential alt
    i = 1
    j = NTAB

    while(j > i+1):
        k = int((i+j)/2)  # integer division
        if(h < htab[k]):
            j = k
        else:
            i = k
    print(i)
    tgrad = gtab[i]
    tbase = ttab[i]
    deltah = h-htab[i]
    tlocal = tbase + tgrad * deltah
    theta = tlocal/ttab[0]

    if(tgrad == 0.0):
        delta = ptab[i] * np.exp(-1*GMR*deltah/tbase)
    else:
        delta = ptab[i] * (tbase/tlocal)**(GMR/tgrad)

    sigma = delta/theta
    return sigma, delta, theta


def get_standard_atmosphere_1d(z):
    NZ = z.shape[0]
    p0 = 1.013250e5
    t0 = 288.15
    p = np.zeros(z.shape)
    t = np.zeros(z.shape)

    for i in np.arange(NZ):
        sigma, delta, theta = atmosphere(z[i]/1000.)  # convert to km
        p[i] = p0 * delta
        t[i] = t0 * theta
    return p, t


def get_standard_atmosphere_2d(z):
    NZ = z.shape[1]
    NY = z.shape[0]
    p0 = 1.013250e5
    t0 = 288.15
    p = np.zeros(z.shape)
    t = np.zeros(z.shape)

    for i in np.arange(NY):
        for j in np.arange(NZ):
            sigma, delta, theta = atmosphere(z[i, j]/1000.)  # convert to km
            p[i, j] = p0 * delta
            t[i, j] = t0 * theta
    return p, t


def get_standard_atmosphere_3d(z):
    NZ = z.shape[2]
    NX = z.shape[0]
    NY = z.shape[1]
    p0 = 1.013250e5
    t0 = 288.15
    p = np.zeros(z.shape)
    t = np.zeros(z.shape)

    for i in np.arange(NX):
        for j in np.arange(NY):
            for k in np.arange(NZ):
                # convert to km
                sigma, delta, theta = atmosphere(z[i, j, k]/1000.)
                p[i, j, k] = p0 * delta
                t[i, j, k] = t0 * theta
    return p, t
