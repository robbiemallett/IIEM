import math
import numpy as np
from cmath import sqrt
from scipy.integrate import dblquad
import scipy
import mpmath
from utils import roughness_spectrum, xpol_integralfunc_vec
import utils
from scipy.special import erfc

def backscatter(frequency,
                sig,
                L,
                thi, # Theta incident
                er, # Dielectric constant
                sp,
                xx,
                block_crosspol=False):

    ths = thi

    phs = 180

    (sigma_0_vv, sigma_0_hh) = I2EM_Bistat_model(frequency,
                                                 sig,
                                                 L,
                                                 thi,
                                                 ths,
                                                 phs,
                                                 er.conjugate(), # Dielectric constant
                                                 sp,
                                                 xx)

    auto = 0 # Or 1

    if block_crosspol:
        sigma_0_hv = np.nan
    else:
        sigma_0_hv = IEMX_model(frequency,
                                sig,
                                L,
                                thi,
                                er.conjugate(),
                                sp,
                                xx,
                                auto)

    return(sigma_0_vv,
           sigma_0_hh,
           sigma_0_hv)

def I2EM_Bistat_model(frequency,
                      sigma_h,
                      CL,
                      theta_i,
                      theta_scat,
                      phi_scat,
                      er, # Dielectric constant
                      sp,
                      xx):

    error = 1.0e8

    sigma_h = sigma_h * 100 # change from m to cm scale
    CL = CL * 100

    mu_r = 1 # relative permeability

    k = 2 * math.pi * frequency / 30 # wavenumber in free space.\
    # Speed of light is in cm / sec
    theta = theta_i * math.pi / 180 # transform to radian
    phi = 0
    thetas = theta_scat * math.pi / 180
    phis = phi_scat * math.pi / 180

    ks = k * sigma_h # roughness parameter
    kl = k * CL

    ks2 = ks * ks

    cs = math.cos(theta + 0.01)
    s = math.sin(theta + 0.01)


    sf = math.sin(phi)
    cf = math.cos(phi)

    ss = math.sin(thetas)
    css = math.cos(thetas)

    cfs = math.cos(phis)
    sfs = math.sin(phis)

    s2 = s ** 2

    kx = k * s * cf
    ky = k * s * sf
    kz = k * cs

    ksx = k * ss * cfs
    ksy = k * ss * sfs
    ksz = k * css

    # -- reflection coefficients
    rt = sqrt(er - s2)
    Rvi = (er * cs - rt) / (er * cs + rt)
    Rhi = (cs - rt) / (cs + rt)


    wvnb = k * math.sqrt((ss * cfs - s * cf) ** 2 + (ss * sfs - s * sf) ** 2)

    Ts = 1

    while error > 1.0e-8:
        Ts = Ts + 1
        error = (ks2 * (cs + css) ** 2) ** Ts / math.factorial(Ts)


    # ---------------- calculating roughness spectrum - ----------

    (wn, rss) = roughness_spectrum(sp,
                                   xx,
                                   wvnb,
                                   sigma_h,
                                   CL,
                                   Ts)


    # ----------- compute R - transition - -----------

    Rv0 = (sqrt(er) - 1) / (sqrt(er) + 1)
    Rh0 = -Rv0


    Ft = 8 * Rv0 ** 2 * ss * (cs + sqrt(er - s2)) / (cs * sqrt(er - s2))
    a1 = 0
    b1 = 0


    for n in np.arange(1,Ts+1):
        a0 = (ks * cs) ** (2 * n) / math.factorial(n)
        a1 = a1 + a0 * wn[n-1]
        b1 = b1 + a0 * (abs(Ft / 2 + 2 ** (n + 1) * Rv0 / cs * math.exp(-(ks * cs) ** 2))) ** 2 *wn[n-1]

    St = 0.25 * (abs(Ft)) ** 2 * a1 / b1

    St0 = 1 / (abs(1 + 8 * Rv0 / (cs * Ft))) ** 2

    Tf = 1 - St / St0


    # ----------- compute average reflection coefficients - -----------
    # -- these coefficients account for slope effects, especially near the
    # brewster angle.They are not important if the slope is small.

    sigx = 1.1 * sigma_h / CL
    sigy = sigx
    xxx = 3 * sigx

    # print( cs, s, er, s2, sigx, sigy)

    #######################

    def complex_dblquad(func, a, b, c, d):

        def real_func(x,y):
            return scipy.real(func(x,y))

        def imag_func(x,y):
            return scipy.imag(func(x,y))

        real_integral = dblquad(real_func, a, b, c, d)
        imag_integral = dblquad(imag_func, a, b, c, d)

        return (real_integral[0] + 1j * imag_integral[0])

    #####################

    Rav = complex_dblquad(lambda Zx,Zy : utils.Rav_integration(Zx, Zy, cs, s, er, s2, sigx, sigy),
                            -xxx, xxx, -xxx, xxx ) # Integrate over Zx, Zy

    Rah = complex_dblquad(lambda Zx,Zy : utils.Rah_integration(Zx, Zy, cs, s, er, s2, sigx, sigy),
                            -xxx, xxx, -xxx, xxx ) # Integrate over Zx, Zy


    Rav = Rav / (2 * math.pi * sigx * sigy)
    Rah = Rah / (2 * math.pi * sigx * sigy)



    # -- select proper reflection coefficients

    if (theta_i == theta_scat) & (phi_scat == 180): # i.e.operating in backscatter mode
        Rvt = Rvi + (Rv0 - Rvi) * Tf
        Rht = Rhi + (Rh0 - Rhi) * Tf

    else: # in this case, it is the bistatic configuration and average R is used
        # Rvt = Rav + (Rv0 - Rav) * Tf
        # Rht = Rah + (Rh0 - Rah) * Tf
        Rvt = Rav
        Rht = Rah

    fvv = 2 * Rvt * (s * ss - (1 + cs * css) * cfs) / (cs + css)
    fhh = -2 * Rht * (s * ss - (1 + cs * css) * cfs) / (cs + css)


    # ------- Calculate the Fppup(dn) i(s) coefficients - ---
    (Fvvupi, Fhhupi) = utils.Fppupdn_is_calculations(+1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    (Fvvups, Fhhups) = utils.Fppupdn_is_calculations(+1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    (Fvvdni, Fhhdni) = utils.Fppupdn_is_calculations(-1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
    (Fvvdns, Fhhdns) = utils.Fppupdn_is_calculations(-1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)

    qi = k * cs
    qs = k * css

    # ----- calculating Ivv and Ihh - ---

    fvv = fvv.conjugate()
    fhh = fhh.conjugate()

    Fvvupi = Fvvupi.conjugate()
    Fvvups = Fvvups.conjugate()
    Fvvdni = Fvvdni.conjugate()
    Fvvdns = Fvvdns.conjugate()

    Fhhupi = Fhhupi.conjugate()
    Fhhups = Fhhups.conjugate()
    Fhhdni = Fhhdni.conjugate()
    Fhhdns = Fhhdns.conjugate()


    Ivv = np.zeros(Ts, dtype=np.complex_)
    Ihh = Ivv.copy()

    for n in np.arange(1,Ts+1):

        Ivv[n-1] = (kz + ksz) ** n * fvv *math.exp(-sigma_h ** 2 * kz * ksz) + \
        0.25 * (Fvvupi * (ksz - qi) ** (n - 1) *math.exp(-sigma_h ** 2 * (qi ** 2 - qi * (ksz - kz))) +
        Fvvdni* (ksz+qi) ** (n - 1) *math.exp(-sigma_h ** 2 * (qi ** 2 + qi * (ksz - kz))) +
        Fvvups * (kz + qs) ** (n - 1) *math.exp(-sigma_h ** 2 * (qs ** 2 - qs * (ksz - kz))) +
        Fvvdns * (kz - qs) ** (n - 1) *math.exp(-sigma_h ** 2 * (qs ** 2 + qs * (ksz - kz))))


        Ihh[n-1] = (kz + ksz) ** n * fhh *math.exp(-sigma_h ** 2 * kz * ksz) + \
        0.25 * (Fhhupi * (ksz - qi) ** (n - 1) *math.exp(-sigma_h ** 2 * (qi ** 2 - qi * (ksz - kz))) +
        Fhhdni* (ksz+qi) ** (n - 1) *math.exp(-sigma_h ** 2 * (qi ** 2 + qi * (ksz - kz))) +
        Fhhups * (kz + qs) ** (n - 1) *math.exp(-sigma_h ** 2 * (qs ** 2 - qs * (ksz - kz))) +
        Fhhdns * (kz - qs) ** (n - 1) *math.exp(-sigma_h ** 2 * (qs ** 2 + qs * (ksz - kz))))


    # -- Shadowing function calculations

    if (theta_i == theta_scat) & (phi_scat == 180): # i.e.working in backscatter mode
        ct = mpmath.cot(theta)
        cts = mpmath.cot(thetas)
        rslp = rss
        ctorslp = float((ct / sqrt(2) / rslp).real)
        ctsorslp = float((cts / sqrt(2) / rslp).real)

        shadf = 0.5 * (math.exp(-ctorslp ** 2) / sqrt(math.pi) / ctorslp - erfc(ctorslp))
        shadfs = 0.5 * (math.exp(-ctsorslp ** 2) / sqrt(math.pi) / ctsorslp - erfc(ctsorslp))
        ShdwS = 1 / (1 + shadf + shadfs)
    else:
        ShdwS = 1


    # ------- calculate the values of sigma_note - -------------

    sigmavv = 0
    sigmahh = 0

    for n in np.arange(1,Ts+1):

        a0 = wn[n-1] / math.factorial(n) * sigma_h ** (2 * n)


        sigmavv = sigmavv + abs(Ivv[n-1]) ** 2 * a0
        sigmahh = sigmahh + abs(Ihh[n-1]) ** 2 * a0



    sigmavv = sigmavv * ShdwS * k ** 2 / 2 * math.exp(-sigma_h ** 2 * (kz ** 2 + ksz ** 2))
    sigmahh = sigmahh * ShdwS * k ** 2 / 2 * math.exp(-sigma_h ** 2 * (kz ** 2 + ksz ** 2))

    ssv = 10 * math.log10(sigmavv.real)
    ssh = 10 * math.log10(sigmahh.real)

    sigma_0_vv = ssv
    sigma_0_hh = ssh

    return sigma_0_vv, sigma_0_hh

def IEMX_model(fr, sig, L, theta_d, er, sp, xx, auto):
    er = er.conjugate()
    sig = sig * 100 # change to cm scale
    L = L * 100 # change to cm scale

    # - fr: frequency in GHz
    # - sig: rms height of surface in cm
    # - L: correlation length of surface in cm
    # - theta_d: incidence angle in degrees
    # - er: relative permittivity
    # - sp: type of surface correlation function

    error = 1.0e8

    k = 2 * math.pi * fr / 30 # wavenumber in free space.Speed of light is in cm / sec
    theta = theta_d * math.pi / 180 # transform to radian


    ks = k * sig # roughness parameter
    kl = k * L

    ks2 = ks * ks
    kl2 = kl ** 2

    cs = math.cos(theta)
    s = math.sin(theta + 0.001)

    s2 = s ** 2

    # -- calculation of reflection coefficints
    rt = sqrt(er - s2)


    rv = (er * cs - rt) / (er * cs + rt)


    rh = (cs - rt) / (cs + rt)

    rvh = (rv - rh) / 2

    # print(rt, rv, rh, rvh)
    # exit()


    # -- rms slope values
    sig_l = sig / L
    if sp.lower() == 'exponential': # -- exponential correl func
        rss = sig_l


    if sp.lower() == 'gaussian': # -- Gaussian correl func
        rss = sig_l * sqrt(2)


    if sp.lower() == 'power_spec': # -- 1.5 - power spectra correl func
        rss = sig_l * sqrt(2 * xx)

    # --- Selecting number of spectral components of the surface roughness
    if auto == 0:
        n_spec = 15 # numberofterms to include in the surface roughness spectra


    if auto == 1:
        n_spec = 1
        while error > 1.0e-8:
            n_spec = n_spec + 1
            error = (ks2 * (2 * cs) ** 2) ** n_spec / math.factorial(n_spec)


    # -- calculating shadow consideration in single scat(Smith, 1967)

    ct = mpmath.cot(theta + 0.001)
    farg = (ct / sqrt(2) / rss).real


    gamma = 0.5 * (math.exp(-float(farg.real) ** 2) / 1.772 / float(farg.real) - erfc(float(farg.real)))

    Shdw = 1 / (1 + gamma)

    # -- calculating multiple scattering contribution
    # ------ a double integration function

    factorials = {}
    for number in np.arange(1,n_spec+1):
        factorials[number] = math.factorial(number)

    svh = dblquad(lambda phi, r : xpol_integralfunc_vec(r, phi, sp, xx, ks2, cs, s,
                                                            kl2, L, er, rss, rvh, n_spec, factorials),
                          0.1, 1, lambda x : 0, lambda x : math.pi)[0]




    svh = svh * 1.e-5 # # un - scale after rescalingin the integrand function.
    sigvh = 10 * math.log10(svh * Shdw)

    return(sigvh)