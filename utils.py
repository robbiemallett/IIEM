import numpy as np
import math
from cmath import sqrt
import itertools
from mpmath import besselk, besselj
from scipy.integrate import quad

def roughness_spectrum(sp, xx, wvnb, sig, L, Ts):

    wn = np.zeros(Ts)

    # -- math.exponential correl func

    if sp.lower() == 'exponential':
        for n in np.arange(1,Ts+1):

            wn[n-1] = L ** 2 / n ** 2 * (1 + (wvnb * L / n) ** 2) ** (-1.5)

        rss = sig / L

    # -- gaussian correl func

    if sp.lower() == 'gaussian':
        for n in np.arange(1,Ts+1):
            wn[n-1] = L  ** 2 / (2 * n) * math.exp(-(wvnb * L)  ** 2 / (4 * n))

        rss = sqrt(2) * sig / L

    # -- x - power correl func
    if sp.lower() == 'power_spec':

        for n in np.arange(1,Ts+1):
            if wvnb == 0:
                wn[n-1] = L ** 2 / (3 * n - 2)
        else:
            wn[n-1] = L ** 2 * (wvnb * L) ** (-1 + xx * n) * besselk(1 - xx * n, wvnb * L) \
                     / (2 ** (xx * n - 1) * math.gamma(xx * n))


    # -- x - math.exponential correl func

    if sp.lower() == 'exponential_spec':
        for n in np.arange(1,Ts+1):
            tmp = quad(lambda z: x_exponential_spectrum(z, wvnb, L, n, xx), 0, 9) # integrate over z

            wn[n-1] = L ** 2 / n ** (2 / xx) * tmp

        rss = sig / L

    return(wn, rss)


def Fppupdn_is_calculations(ud, is_, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs):

    if is_ == 1:

        Gqi = ud * kz
        Gqti = ud * k * sqrt(er - s  ** 2)
        qi = ud * kz

        c11 = k * cfs * (ksz - qi)
        c21 = cs * (cfs * (k  ** 2 * s * cf * (ss * cfs - s * cf) + Gqi * (k * css - qi))
                     + k  ** 2 * cf * s * ss * sfs  ** 2)
        c31 = k * s * (s * cf * cfs * (k * css - qi) - Gqi * (cfs * (ss * cfs - s * cf) + ss * sfs ** 2))
        c41 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c51 = Gqi * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))

        c12 = k * cfs * (ksz - qi)
        c22 = cs * (cfs * (k  ** 2 * s * cf * (ss * cfs - s * cf) + Gqti * (k * css - qi))
                     + k  ** 2 * cf * s * ss * sfs ** 2)
        c32 = k * s * (s * cf * cfs * (k * css - qi) - Gqti * (cfs * (ss * cfs - s * cf) - ss * sfs ** 2))
        c42 = k * cs * (cfs * css * (k * css - qi) + k * ss * (ss * cfs - s * cf))
        c52 = Gqti * (cfs * css * (qi - k * css) - k * ss * (ss * cfs - s * cf))


    elif is_ == 2:

        Gqs = ud * ksz
        Gqts = ud * k * sqrt(er - ss  ** 2)
        qs = ud * ksz

        c11 = k * cfs * (kz + qs)
        c21 = Gqs * (cfs * (cs * (k * cs + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c31 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c41 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c51 = -css * (k  ** 2 * ss * (ss * cfs - s * cf) + Gqs * cfs * (kz + qs))

        c12 = k * cfs * (kz + qs)
        c22 = Gqts * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c32 = k * ss * (k * cs * (ss * cfs - s * cf) + s * (kz + qs))
        c42 = k * css * (cfs * (cs * (kz + qs) - k * s * (ss * cfs - s * cf)) - k * s * ss * sfs  ** 2)
        c52 = -css * (k  ** 2 * ss * (ss * cfs - s * cf) + Gqts * cfs * (kz + qs))


    q = kz
    qt = k * sqrt(er - s  ** 2)

    vv = (1 + Rvi) * (-(1 - Rvi) * c11 / q + (1 + Rvi) * c12 / qt) + \
        (1 - Rvi) * ((1 - Rvi) * c21 / q - (1 + Rvi) * c22 / qt) + \
        (1 + Rvi) * ((1 - Rvi) * c31 / q - (1 + Rvi) * c32 / er / qt) + \
        (1 - Rvi) * ((1 + Rvi) * c41 / q - er * (1 - Rvi) * c42 / qt) + \
        (1 + Rvi) * ((1 + Rvi) * c51 / q - (1 - Rvi) * c52 / qt)

    hh = (1 + Rhi) * ((1 - Rhi) * c11 / q - er * (1 + Rhi) * c12 / qt) - \
        (1 - Rhi) * ((1 - Rhi) * c21 / q - (1 + Rhi) * c22 / qt) - \
        (1 + Rhi) * ((1 - Rhi) * c31 / q - (1 + Rhi) * c32 / qt) - \
        (1 - Rhi) * ((1 + Rhi) * c41 / q - (1 - Rhi) * c42 / qt) - \
        (1 + Rhi) * ((1 + Rhi) * c51 / q - (1 - Rhi) * c52 / qt)

    return(vv, hh)

def Rav_integration(Zx, Zy, cs, s, er, s2, sigx, sigy):

    A = cs + Zx * s
    B = er * (1 + Zx  ** 2 + Zy  ** 2)
    CC = s2 - 2 * Zx * s * cs + Zx  ** 2 * cs  ** 2 + Zy  ** 2

    Rv = (er * A - sqrt(B - CC)) / (er * A + sqrt(B - CC))

    pd = math.exp(-Zx ** 2 / (2 * sigx  ** 2) - Zy  ** 2 / (2 * sigy  ** 2))
    Rav = Rv * pd

    return Rav

def Rah_integration(Zx, Zy, cs, s, er, s2, sigx, sigy):

    A = cs + Zx * s
    B = er * (1 + Zx  ** 2 + Zy  ** 2)
    CC = s2 - 2 * Zx * s * cs + Zx  ** 2 * cs  ** 2 + Zy  ** 2

    Rh = (A - sqrt(B - CC)) / (A + sqrt(B - CC))

    pd = math.exp(-Zx  ** 2 / (2 * sigx  ** 2) - Zy  ** 2 / (2 * sigy  ** 2))
    Rah = Rh * pd

    return Rah


def x_exponential_spectrum(z,wvnb,L,n,xx):

    tmp = math.exp(-abs(z) ** xx) * besselj(0, z*wvnb*L/(n**(1/xx)))*z

    return(tmp)


def spectrm2(sp, xx, kl2, L, rx, ry, s, np_, nr):

    wm = np.zeros((np_,nr))

    if sp.lower() == 'exponential':  # exponential
        for n in np.arange(1,np_+1):
            wm[n-1,:] = n* kl2/(n**2 + kl2 *((rx+s)**2+ry**2))**1.5


    if sp.lower() == 'gaussian':  #  gaussian
        for n in np.arange(1,np_+1):
            wm[n-1,:] = 0.5 * kl2/n* math.exp(-kl2*((rx+s)**2 + ry**2)/(4*n))


    if sp.lower()== 'power_spec': # x-power
        for n in np.arange(1,np_+1):
            wm[n-1,:] = kl2/(2.**(xx*n-1)*math.gamma(xx*n))* ( ( (rx+s)**2.
              + ry**2)*L)**(xx*n-1)* besselk(-xx*n+1, L*((rx+s)**2 + ry**2))

    return wm


def spectrm1(sp, xx, kl2, L, rx, ry, s, np_, nr):

    wn = np.zeros((np_,nr))

    if sp.lower() == 'exponential':  # exponential
        for n in np.arange(1,np_+1):
            wn[n-1, :] = n* kl2/(n**2 + kl2 *((rx-s)**2+ry**2))**1.5


    if sp.lower() == 'gaussian':  #  gaussian
        for n in np.arange(1,np_+1):
            wn[n-1,:] = 0.5 * kl2/n* math.exp(-kl2*((rx-s)**2 + ry**2)/(4*n))

    if sp.lower() == 'power_spec':  # x-power
        for n in np.arange(1,np_+1):
            wn[n-1,:] = kl2/(2.**(xx*n-1)*math.gamma(xx*n))* ( ( (rx-s)**2.
              + ry**2)*L)**(xx*n-1)* besselk(-xx*n+1, L*((rx-s)**2 + ry**2))

    return wn


def xpol_integralfunc_vec(r, phi, sp, xx, ks2, cs,s, kl2, L, er, rss, rvh, n_spec, factorials):

    cs2 = cs **2

    r2 = r ** 2

    if type(r) == float:
        nr = 1
    else:
        nr = len(r)


    sf = math.sin(phi)
    csf = math.cos(phi)
    rx = r * csf
    ry = r * sf

    #-- calculation of the field coefficients
    rp = 1 + rvh
    rm = 1 - rvh

    q = np.sqrt(1.0001 - r2)
    qt = np.sqrt(er - r2)

    a = rp /q
    b = rm /q
    c = rp /qt
    d = rm /qt

    #--calculate cross-pol coefficient
    B3 = rx * ry /cs
    fvh1 = (b-c)*(1- 3*rvh) - (b - c/er) * rp
    fvh2 = (a-d)*(1+ 3*rvh) - (a - d*er) * rm
    Fvh = ( abs( (fvh1 + fvh2) *B3)) **2


    #-- calculate shadowing func for multiple scattering

    au = (q /r /1.414 /rss)



    fsh = (0.2821/au) *math.exp(-au **2) -0.5 *(1- math.erf(au))
    sha = 1./(1 + fsh)

    #-- calculate expressions for the surface spectra
    wn = spectrm1(sp, xx, kl2, L, rx, ry, s, n_spec, nr)
    wm = spectrm2(sp, xx, kl2, L, rx, ry, s, n_spec, nr)


    #--compute VH scattering coefficient
    acc = math.exp(-2* ks2 *cs2) /(16 * math.pi)

    vhmnsum = 0
    my_range = np.arange(1,n_spec+1)
    for (n, m) in itertools.product(my_range, my_range):
        vhmnsum = vhmnsum + wn[n-1,:]*wm[m-1,:] * (ks2*cs2) **(n+m) /factorials[n]/factorials[m]

    VH = 4 * acc * Fvh * vhmnsum * r

    y = VH * sha

    y =y * 1.e5 # rescale so dblquad() works better.

    return(y)