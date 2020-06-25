import math
import numpy as np
from utils import Fppupdn_is_calculations
from math import sqrt
from scipy.integrate import dblquad
import scipy
import mpmath
from utils import roughness_spectrum, xpol_integralfunc_vec
import utils
from scipy.special import erfc

def Multiscale_I2EM_Backscatter(fr,sig1, L1, sig2, L2, sig3, L3, thi, er):

        ths = thi

        phs = 180

        error = 1.0e8

        sig1 = sig1 * 100 # change from m to cm scale
        L1 = L1 * 100

        sig2 = sig2 * 100 # change from m to cm scale
        L2 = L2 * 100

        sig3 = sig3 * 100 # change from m to cm scale
        L3 = L3 * 100

        sig12 = sig1 ** 2
        sig22 = sig2 ** 2
        sig32 = sig3 ** 2

        sigs2 = sig12 + sig22 + sig32

        mu_r = 1 # relative permeability

        # Speed of light is in cm / sec

        theta = thi *math.pi / 180 # transform to radian

        k = 2 *math.pi * fr / 30 # wavenumber in free space.

        phi = 0
        thetas = ths *math.pi / 180
        phis = phs *math.pi / 180

        ks = k * math.sqrt(sigs2) # roughness parameter
        kl1 = k * L1

        ks2 = ks * ks

        cs = math.cos(theta + 0.01)
        s = math.sin(theta + 0.01)

        sf = math.sin(phi)
        cf = math.cos(phi)

        ss = math.sin(thetas)
        css = math.cos(thetas)

        cfs = math.cos(phis)
        sfs = math.sin(phis)

        s2 = s * s
        # sq = sqrt(er - s2)

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

        wvnb = k * sqrt((ss * cfs - s * cf) ** 2 + (ss * sfs - s * sf) ** 2)

        Ts = 1


        while error > 1.0e-8:
                Ts = Ts + 1

                error = (ks2.real * (cs + css) ** 2) ** Ts / math.factorial(Ts)

        # ---------------- calculating roughness spectrum - ----------

        wn = utils.multi_roughness_spectrum(sig12, sig22, sig32, sigs2, L1, L2, L3, k, s, Ts)

        rss = sqrt(2) / sigs2 * (sig12 * sig1 / L1 + sig22 * sig2 / L2 + sig32 * sig3 / L3)

        # ----------- compute R - transition - -----------

        Rv0 = (sqrt(er) - 1) / (sqrt(er) + 1)
        Rh0 = -Rv0

        Ft = 8 * Rv0 ** 2 * ss * (cs + sqrt(er - s2)) / (cs * sqrt(er - s2))
        a1 = 0
        b1 = 0
        for n in np.arange(1,Ts + 1):
                a0 = (ks.real * cs) ** (2 * n) / math.factorial(n)
                a1 = a1 + a0 * wn[n-1]

                b1 = b1 + a0 * (abs(Ft / 2 + 2 ** (n + 1) * Rv0 / cs * math.exp(-(ks * cs) ** 2))) ** 2 *wn[n-1]

        St = 0.25 * (abs(Ft)) ** 2 * a1 / b1

        St0 = 1 / (abs(1 + 8 * Rv0 / (cs * Ft))) ** 2

        Tf = 1 - St / St0

        # # ----------- compute average reflection coefficients - -----------
        # # -- these coefficients account for slope effects
        # # especially near the  brewster angle.
        # # They are not important if the slope is small.
        #
        # sigx = 1.1 * sig / L
        # sigy = sigx
        # xxx = 3 * sigx
        #
        # Rav = dblquad( @ (Zx, Zy)
        # Rav_integration(Zx, Zy, cs, s, er, s2, sigx, sigy), -xxx, xxx, -xxx, xxx )
        #
        # Rah = dblquad( @ (Zx, Zy)
        # Rah_integration(Zx, Zy, cs, s, er, s2, sigx, sigy), -xxx, xxx, -xxx, xxx )
        #
        # Rav = Rav / (2 *math.pi * sigx * sigy)
        # Rah = Rah / (2 *math.pi * sigx * sigy)
        #

        # -- select proper reflection coefficients

        # if thi == ths & & phs == 180, # i.e.operating in backscatter mode
        Rvt = Rvi + (Rv0 - Rvi) * Tf
        Rht = Rhi + (Rh0 - Rhi) * Tf

        # else # in this case, it is the bistatic configuration and average R is used
        # # Rvt = Rav + (Rv0 - Rav) * Tf
        # # Rht = Rah + (Rh0 - Rah) * Tf
        # Rvt = Rav
        # Rht = Rah
        # end

        fvv = 2 * Rvt * (s * ss - (1 + cs * css) * cfs) / (cs + css)
        fhh = -2 * Rht * (s * ss - (1 + cs * css) * cfs) / (cs + css)

        # ------- Calculate the Fppup(dn) i(s) coefficients - ---

        (Fvvupi, Fhhupi) = Fppupdn_is_calculations(+1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
        (Fvvups, Fhhups) = Fppupdn_is_calculations(+1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
        (Fvvdni, Fhhdni) = Fppupdn_is_calculations(-1, 1, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)
        (Fvvdns, Fhhdns) = Fppupdn_is_calculations(-1, 2, Rvi, Rhi, er, k, kz, ksz, s, cs, ss, css, cf, cfs, sfs)

        qi = k * cs
        qs = k * css

        # ----- calculating Ivv and Ihh - ---

        Ivv = np.zeros(Ts, dtype=np.complex_)
        Ihh = Ivv.copy()


        for n in np.arange(1, Ts + 1):
                Ivv[n-1] = (kz + ksz) ** n * fvv * math.exp(-sigs2 * kz * ksz) + \
                0.25 * (Fvvupi * (ksz - qi) ** (n - 1) * math.exp(-sigs2 * (qi ** 2 - qi * (ksz - kz))) +
                        Fvvdni * (ksz+qi) ** (n - 1) * math.exp(-sigs2 * (qi ** 2 + qi * (ksz - kz))) +
                Fvvups * (kz + qs) ** (n - 1) * math.exp(-sigs2 * (qs ** 2 - qs * (ksz - kz))) +
                Fvvdns * (kz - qs) ** (n - 1) * math.exp(-sigs2 * (qs ** 2 + qs * (ksz - kz))))


                Ihh[n-1] = (kz + ksz) ** n * fhh * math.exp(-sigs2 * kz * ksz) + \
                0.25 * (Fhhupi * (ksz - qi) ** (n - 1) * math.exp(-sigs2 * (qi ** 2 - qi * (ksz - kz))) +
                        Fhhdni * (ksz+qi) ** (n - 1) * math.exp(-sigs2 * (qi ** 2 + qi * (ksz - kz))) +
                Fhhups * (kz + qs) ** (n - 1) * math.exp(-sigs2 * (qs ** 2 - qs * (ksz - kz))) +
                Fhhdns * (kz - qs) ** (n - 1) * math.exp(-sigs2 * (qs ** 2 + qs * (ksz - kz))))


        # -- Shadowing function calculations

        # if thi == ths & & phs == 180 # i.e.working in backscatter mode
        ct = mpmath.cot(theta)
        cts = mpmath.cot(thetas)
        rslp = rss
        ctorslp = float(ct / math.sqrt(2) / rslp)
        ctsorslp = float(cts / math.sqrt(2) / rslp)

        shadf = 0.5 * (math.exp(-ctorslp ** 2) / sqrt(math.pi) / ctorslp - erfc(ctorslp))
        shadfs = 0.5 * (math.exp(-ctsorslp ** 2) / sqrt(math.pi) / ctsorslp - erfc(ctsorslp))
        ShdwS = 1 / (1 + shadf + shadfs)


        # ------- calculate the values of sigma_note - -------------

        sigmavv = 0
        sigmahh = 0

        for n in np.arange(1, Ts + 1):

                a0 = wn[n-1] / math.factorial(n) * sigs2 ** (n)

                sigmavv = sigmavv + abs(Ivv[n-1]) ** 2 * a0
                sigmahh = sigmahh + abs(Ihh[n-1]) ** 2 * a0


        sigmavv = sigmavv * ShdwS * k ** 2 / 2 * math.exp(-sigs2 * (kz ** 2 + ksz ** 2))
        sigmahh = sigmahh * ShdwS * k ** 2 / 2 * math.exp(-sigs2 * (kz ** 2 + ksz ** 2))

        if (abs(sigmavv.imag) + abs(sigmahh.imag)) > 0:
                raise Exception('backscatter coeff must be real')

        ssv = 10 * math.log10(sigmavv.real)
        ssh = 10 * math.log10(sigmahh.real)

        sigma_0_vv = ssv
        sigma_0_hh = ssh

        return(sigma_0_vv, sigma_0_hh)

if __name__ == "__main__":

        sig1 = 0.4 / 100 # rms height(m)
        L1 = 0.07 # correlation length(m)

        sig2 = 0.25 / 100 # rms height(m)
        L2 = 0.03 # correlation length(m)

        sig3 = 0.13 / 100 # rms height(m)
        L3 = 0.015 # correlation length(m)
        er = 9 # dielectric constant

        fr = 4 # frequency(GHz)

        thi = 30 # incidence angle

        # -- using the I2EM Multiscalecode
        sigma_0_vv, sigma_0_hh = Multiscale_I2EM_Backscatter(fr, sig1, L1, sig2, L2,
                                                               sig3, L3, thi, er)

        print(sigma_0_vv,
              sigma_0_hh)