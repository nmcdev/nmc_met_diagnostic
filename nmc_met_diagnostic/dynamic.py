# _*_ coding: utf-8 _*_

# Copyright (c) 2019 NMC Developers.
# Distributed under the terms of the GPL V3 License.

"""
  Compute dynamic physical parameters on lon/lat grid.

  refer
  https://github.com/keltonhalbert/wrftools/blob/master/wrftools/variables/winds.py
  https://bitbucket.org/tmiyachi/pymet
"""

import numpy as np
from nmc_met_base import arr, constants
from nmc_met_base.grid import calc_dx_dy, dvardx, dvardy, d2vardx2, \
                              d2vardy2, dvardp, gradient_sphere, rot
from nmc_met_diagnostic.thermal import pottemp, stability


NA = np.newaxis
a0 = constants.Re
g = constants.g0
PI = constants.pi
d2r = PI/180.
Rd = constants.rd


def avort(uwind, vwind, lon, lat):
    """
    Calculate absolute vorticity.
    refer to
    https://nbviewer.jupyter.org/url/fujita.valpo.edu/~kgoebber/NAM_vorticity.ipynb

    :param uwind: u direction wind.
    :param vwind: v direction wind.
    :param lon: grid longitude.
    :param lat: grid latitude.
    :return: relative vorticity.

    :Example:

    """

    # grid space
    dx, dy = calc_dx_dy(lon, lat)

    # relative vorticity
    dvdx = np.gradient(vwind, dx, axis=1)
    dudy = np.gradient(uwind, dy, axis=0)
    cor = 2 * (7.292 * 10 ** (-5)) * np.sin(np.deg2rad(lat))

    return dvdx - dudy + cor


def absvrt(uwnd, vwnd, lon, lat, xdim, ydim, cyclic=True, sphere=True):
    """
    Calculate absolute vorticity.

    :param uwnd: ndarray, u-component wind.
    :param vwnd: ndarray, v-component wind.
    :param lon: array_like, longitude.
    :param lat: array_like, latitude.
    :param xdim: the longitude dimension index
    :param ydim: the latitude dimension index
    :param cyclic: east-west boundary is cyclic
    :param sphere: sphere coordinate
    :return:
    """

    u, v = np.ma.getdata(uwnd), np.ma.getdata(vwnd)
    mask = np.ma.getmask(uwnd) | np.ma.getmask(vwnd)
    ndim = u.ndim

    vor = rot(u, v, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    f = arr.expand(constants.earth_f(lat), ndim, axis=ydim)
    out = f + vor

    out = np.ma.array(out, mask=mask)
    out = arr.mrollaxis(out, ydim, 0)
    out[0, ...] = np.ma.masked
    out[-1, ...] = np.ma.masked
    out = arr.mrollaxis(out, 0, ydim + 1)

    return out


def ertelpv(uwnd, vwnd, temp, lon, lat, lev, xdim, ydim, zdim,
            cyclic=True, punit=100., sphere=True):
    """
    Calculate Ertel potential vorticity.
    Hoskins, B.J., M.E. McIntyre and A.E. Robertson, 1985:
      On the use and significance of isentropic potential
      vorticity maps, `QJRMS`, 111, 877-946,
    <http://onlinelibrary.wiley.com/doi/10.1002/qj.49711147002/abstract>

    :param uwnd: ndarray, u component wind [m/s].
    :param vwnd: ndarray, v component wind [m/s].
    :param temp: ndarray, temperature [K].
    :param lon: array_like, longitude [degrees].
    :param lat: array_like, latitude [degrees].
    :param lev: array_like, pressure level [punit*Pa].
    :param xdim: west-east axis
    :param ydim: south-north axis
    :param zdim: vertical axis
    :param cyclic: west-east cyclic boundary
    :param punit: pressure level unit
    :param sphere: sphere coordinates.
    :return:
    """

    u, v, t = np.ma.getdata(uwnd), np.ma.getdata(vwnd), np.ma.getdata(temp)
    mask = np.ma.getmask(uwnd) | np.ma.getmask(vwnd) | np.ma.getmask(temp)
    ndim = u.ndim

    # potential temperature
    theta = pottemp(t, lev, zdim, punit=punit)

    # partial derivation
    dthdp = dvardp(theta, lev, zdim, punit=punit)
    dudp = dvardp(u, lev, zdim, punit=punit)
    dvdp = dvardp(v, lev, zdim, punit=punit)

    dthdx = dvardx(theta, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    dthdy = dvardy(theta, lat, ydim, sphere=sphere)

    # absolute vorticity
    vor = rot(u, v, lon, lat, xdim, ydim, cyclic=cyclic, sphere=sphere)
    f = arr.expand(constants.earth_f(lat), ndim, axis=ydim)
    avor = f + vor

    out = -g * (avor*dthdp - (dthdx*dvdp-dthdy*dudp))

    out = np.ma.array(out, mask=mask)
    out = arr.mrollaxis(out, ydim, 0)
    out[0, ...] = np.ma.masked
    out[-1, ...] = np.ma.masked
    out = arr.mrollaxis(out, 0, ydim+1)

    return out


def vertical_vorticity_latlon(u, v, lats, lons, abs_opt=False):
    """
    Calculate the vertical vorticity on a latitude/longitude grid.

    :param u: 2 dimensional u wind arrays, dimensioned by (lats,lons).
    :param v: 2 dimensional v wind arrays, dimensioned by (lats,lons).
    :param lats: latitude vector
    :param lons: longitude vector
    :param abs_opt: True to compute absolute vorticity,
                    False for relative vorticity only
    :return: Two dimensional array of vertical vorticity.
    """

    dudy, dudx = gradient_sphere(u, lats, lons)
    dvdy, dvdx = gradient_sphere(v, lats, lons)

    if abs_opt:
        # 2D latitude array
        glats = np.zeros_like(u).astype('f')
        for jj in range(0, len(lats)):
            glats[jj, :] = lats[jj]

        # Coriolis parameter
        f = 2 * 7.292e-05 * np.sin(np.deg2rad(glats))
    else:
        f = 0.

    vert_vort = dvdx - dudy + f

    return vert_vort


def epv_sphere(theta, u, v, levs, lats, lons):
    """
    Computes the Ertel Potential Vorticity (PV) on a latitude/longitude grid.

    :param theta: 3D potential temperature array on isobaric levels
    :param u: 3D u components of the horizontal wind on isobaric levels
    :param v: 3D v components of the horizontal wind on isobaric levels
    :param levs: 1D pressure vectors
    :param lats: 1D latitude vectors
    :param lons: 1D longitude vectors
    :return: Ertel PV in potential vorticity units (PVU)
    """

    iz, iy, ix = theta.shape

    dthdp, dthdy, dthdx = gradient_sphere(theta, levs, lats, lons)
    dudp, dudy, dudx = gradient_sphere(u, levs, lats, lons)
    dvdp, dvdy, dvdx = gradient_sphere(v, levs, lats, lons)

    abvort = np.zeros_like(theta).astype('f')
    for kk in range(0, iz):
        abvort[kk, :, :] = vertical_vorticity_latlon(
            u[kk, :, :].squeeze(), v[kk, :, :].squeeze(),
            lats, lons, abs_opt=True)

    epv = (-9.81 * (-dvdp * dthdx - dudp * dthdy + abvort * dthdp)) * 1.0e6
    return epv


def tnflux2d(U, V, strm, lon, lat, xdim, ydim, cyclic=True, limit=100):
    """
    Takaya & Nakamura (2001) 计算水平等压面上的波活动度.
    Takaya, K and H. Nakamura, 2001: A formulation of a phase-independent
      wave-activity flux for stationary and migratory quasigeostrophic eddies
      on a zonally varying basic flow, `JAS`, 58, 608-627.
    http://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282001%29058%3C0608%3AAFOAPI%3E2.0.CO%3B2

    :param U: u component wind [m/s].
    :param V: v component wind [m/s].
    :param strm: stream function [m^2/s].
    :param lon: longitude degree.
    :param lat: latitude degree.
    :param xdim: longitude dimension index.
    :param ydim: latitude dimension index.
    :param cyclic: east-west cyclic boundary.
    :param limit:
    :return:
    """

    U, V = np.asarray(U), np.asarray(V)
    ndim = U.ndim

    dstrmdx = dvardx(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    dstrmdy = dvardy(strm, lat, ydim)
    d2strmdx2 = d2vardx2(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    d2strmdy2 = d2vardy2(strm, lat, ydim)
    d2strmdxdy = dvardy(
        dvardx(strm, lon, lat, xdim, ydim, cyclic=cyclic),
        lat, ydim)

    tnx = U * (dstrmdx ** 2 - strm * d2strmdx2) + \
        V * (dstrmdx * dstrmdy - strm * d2strmdxdy)
    tny = U * (dstrmdx * dstrmdy - strm * d2strmdxdy) + \
        V * (dstrmdy ** 2 - strm * d2strmdy2)

    tnx = 0.5 * tnx / np.abs(U + 1j * V)
    tny = 0.5 * tny / np.abs(U + 1j * V)

    tnxy = np.sqrt(tnx ** 2 + tny ** 2)
    tnx = np.ma.asarray(tnx)
    tny = np.ma.asarray(tny)
    tnx[tnxy > limit] = np.ma.masked
    tny[tnxy > limit] = np.ma.masked
    tnx[U < 0] = np.ma.masked
    tny[U < 0] = np.ma.masked

    return tnx, tny


def tnflux3d(U, V, T, strm, lon, lat, lev, xdim, ydim, zdim,
             cyclic=True, limit=100, punit=100.):
    """
    Takaya & Nakamura (2001) 计算等压面上的波活动度.
    Takaya, K and H. Nakamura, 2001: A formulation of a phase-independent
      wave-activity flux for stationary and migratory quasigeostrophic eddies
      on a zonally varying basic flow, `JAS`, 58, 608-627.
    http://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282001%29058%3C0608%3AAFOAPI%3E2.0.CO%3B2

    :param U: u component wind [m/s].
    :param V: v component wind [m/s].
    :param T: climate temperature [K].
    :param strm: stream function bias [m^2/s]
    :param lon: longitude degree.
    :param lat: latitude degree.
    :param lev: level pressure.
    :param xdim: longitude dimension index.
    :param ydim: latitude dimension index.
    :param zdim: level dimension index.
    :param cyclic: east-west cyclic boundary.
    :param limit:
    :param punit: level pressure unit.
    :return: east-west, south-north, vertical component.
    """

    U, V, T = np.asarray(U), np.asarray(V), np.asarray(T)
    ndim = U.ndim
    S = stability(T, lev, zdim, punit=punit)
    f = arr.expand(constants.earth_f(lat), ndim, axis=ydim)

    dstrmdx = dvardx(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    dstrmdy = dvardy(strm, lat, ydim)
    dstrmdp = dvardp(strm, lev, zdim, punit=punit)
    d2strmdx2 = d2vardx2(strm, lon, lat, xdim, ydim, cyclic=cyclic)
    d2strmdy2 = d2vardy2(strm, lat, ydim)
    d2strmdxdy = dvardy(dstrmdx, lat, ydim)
    d2strmdxdp = dvardx(dstrmdp, lon, lat, xdim, ydim, cyclic=True)
    d2strmdydp = dvardy(dstrmdp, lat, ydim)

    tnx = U * (dstrmdx ** 2 - strm * d2strmdx2) + \
        V * (dstrmdx * dstrmdy - strm * d2strmdxdy)
    tny = U * (dstrmdx * dstrmdy - strm * d2strmdxdy) + \
        V * (dstrmdy ** 2 - strm * d2strmdy2)
    tnz = f ** 2 / S ** 2 * (
        U * (dstrmdx * dstrmdp - strm * d2strmdxdp) -
        V * (dstrmdy * dstrmdp - strm * d2strmdydp))

    tnx = 0.5 * tnx / np.abs(U + 1j * V)
    tny = 0.5 * tny / np.abs(U + 1j * V)

    tnxy = np.sqrt(tnx ** 2 + tny ** 2)
    tnx = np.ma.asarray(tnx)
    tny = np.ma.asarray(tny)
    tnz = np.ma.asarray(tnz)
    tnx[(U < 0) | (tnxy > limit)] = np.ma.masked
    tny[(U < 0) | (tnxy > limit)] = np.ma.masked
    tnz[(U < 0) | (tnxy > limit)] = np.ma.masked

    return tnx, tny, tnz


def w_to_omega(w, pres, tempk):
    """
    Compute vertical velocity on isobaric surfaces

    :param w: Input vertical velocity (m s-1)
    :param pres: Input half level pressures (full field) in Pa
    :param tempk: Input temperature (K)
    :return:
    """

    omeg = -((pres * g) / (Rd * tempk)) * w
    return omeg
