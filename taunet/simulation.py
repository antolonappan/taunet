import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import camb
import healpy as hp
import pysm3
import pysm3.units as u
from taunet.ncm import NoiseModel, NoiseModelDiag
from taunet import DATADIR
import os
import pickle as pl
import taunet.database as db


try:
    cosbeam = hp.read_cl(os.path.join(DATADIR,'beam_440T_coswinP_pixwin16.fits'))[1]
except FileNotFoundError:
    cosbeam = hp.read_cl(os.path.join(DATADIR,'beam_coswin_ns16.fits'))[0]

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1.0 / cl[np.where(cl > 0)]
    return ret


class CMBspectra:
    """
    Theoretical CMB power spectra

    Parameters:
    -----------

    H0 : float, optional (default=67.32)
        Hubble constant
    ombh2 : float, optional (default=0.02237)
        Physical baryon density
    omch2 : float, optional (default=0.1201)
        Physical cold dark matter density
    ns : float, optional (default=0.9651)
        Scalar spectral index
    mnu : float, optional (default=0.06)
        Sum of neutrino masses
    tau : float, optional (default=0.06)
        Optical depth
    
    """

    def __init__(
        self, H0=67.32, ombh2=0.02237, omch2=0.1201, ns=0.9651, mnu=0.06, tau=0.06,
        ignore_db=False
    ) -> None:
        pars = camb.CAMBparams()
        self.tau = tau
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, tau=tau)
        pars.InitPower.set_params(ns=ns, r=0)
        pars.set_for_lmax(284, lens_potential_accuracy=0)
        self.results = camb.get_results(pars)
        self.db = db.SpectrumDB()
        if ignore_db:
            self.powers = self.results.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=True)
        else:
            self.powers = self.powers()
        self.EE = self.powers[:, 1]
        self.ell = np.arange(len(self.EE))

    
    def powers(self):
        if self.db.check_tau_exist(self.tau):
            return self.db.get_spectra(self.tau)
        else:
            powers = self.results.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=True)
            self.db.insert_spectra(self.tau, powers)
            del powers
            return self.db.get_spectra(self.tau)


    def save_power(self, libdir, retfile=False):
        fname = os.path.join(
            libdir, f"lensed_scalar_cls_{str(self.tau).replace('.','p')}.dat"
        )
        if not os.path.isfile(fname):
            np.savetxt(fname, self.powers)
        if retfile:
            return fname


class CMBmap:
    """
    Simulations class to generate CMB maps

    Parameters
    ----------

    libdir : str
        Library directory to save the maps
    tau : float
        Optical depth
    """

    def __init__(self, tau: float,ignore_db=False):
        self.tau = tau
        print("tau = {}".format(tau))
        self.NSIDE = 16
        self.lmax = 3 * self.NSIDE - 1
        self.ignore_db = ignore_db
        self.db = None if ignore_db else db.MapDB()

    @property
    def EE(self):
        return CMBspectra(tau=self.tau,ignore_db=self.ignore_db).EE
        

    def QU(self, idx=None):

        def __QU__():
            alm = hp.synalm(self.EE, lmax=100, new=True)
            hp.almxfl(alm, cosbeam, inplace=True)
            dummy = np.zeros_like(alm)
            TQU = hp.alm2map([dummy, alm, dummy], self.NSIDE)
            QU = TQU[1:].copy()
            del (alm, dummy, TQU)
            return QU
        
        if idx is None:
            seed = 261092
        else:
            seed = 261092 + idx
        
        np.random.seed(seed)

        if self.ignore_db:
            return __QU__()
        else:
            if self.db.check_seed_exist(self.tau, seed):
                return self.db.get_map(seed, self.tau)
            else:
                qu = __QU__()
                self.db.insert_map(seed, self.tau, qu)
                return self.db.get_map(seed, self.tau)


    def Emode(self, idx=None):
        QU = self.QU(idx=idx)
        return hp.map2alm_spin(QU, 2, lmax=self.lmax)[0]

class FGMap:
    """
    SkySimulation class to generate foreground maps

    Parameters
    ----------
    libdir : str
        Library directory to save the maps
    model : list, optional (default=["d1","s1"])
        Foreground model to use
    """

    def __init__(self, model=["d1", "s1"],ignore_db=False):
        self.model = model
        self.NSIDE = 16
        self.lmax = 3 * self.NSIDE - 1
        self.ignore_db = ignore_db
        self.db = None if ignore_db else db.ForegroundDB()

    def __QU__(self, band):
        sky = pysm3.Sky(nside=self.NSIDE, preset_strings=self.model)
        maps = sky.get_emission(band * u.GHz)
        maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band * u.GHz))
        alms = hp.map2alm(maps.value, lmax=self.lmax)
        hp.almxfl(alms[1], cosbeam, inplace=True)
        hp.almxfl(alms[2], cosbeam, inplace=True)
        TQU = hp.alm2map(alms, self.NSIDE)
        QU = TQU[1:].copy()
        del maps
        return QU
    
    def QU(self,band):
        if self.ignore_db:
            return self.__QU__(band)
        else:
            if self.db.check_model_exist(self.model, band):
                return self.db.get_map(self.model, band)
            else:
                qu = self.__QU__(band)
                self.db.insert_map(self.model, band, qu)
                return self.db.get_map(self.model, band)


    def Emode(self, band, mask=None,):
        QU = self.QU(band)
        if mask is not None:
            QU = QU * mask
        return hp.map2alm_spin(QU, 2, lmax=self.lmax)[0]

class SkySimulation:
    """
    SkySimulation class to generate CMB and foreground maps

    Parameters
    ----------
    tau : float
        Optical depth
    add_noise : bool, optional (default=True)
        Add noise to the maps
    add_fg : bool, optional (default=True)
        Add foregrounds to the maps
    fg : list, optional (default=["s0","d0"])
        Foreground model to use
    ssim : int, optional (default=0)
        Starting index for the simulations
    noise_g : bool, optional (default=False)
        Use Gaussian noise
    noise_diag : bool, optional (default=False)
        Use diagonal noise
    noise_method : str, optional (default="sroll")
        Noise method to use
    fullsky : bool, optional (default=False)
        Use full-sky maps
    """

    def __init__(
        self,
        tau: float,
        add_noise: bool = True,
        add_fg: bool = True,
        fg: list = ["s1", "d1"],
        ssim: int = 0,
        noise_g: bool = False,
        noise_diag: bool = False,
        noise_method: str = "sroll",
        fullsky: bool = False,
    ):

        self.CMB = CMBmap(tau)
        self.FG = FGMap(fg)
        self.noise = NoiseModel(diag=noise_diag,method=noise_method)
        self.ssim = ssim
        self.nside = self.CMB.NSIDE
        self.add_fg = add_fg
        self.add_noise = add_noise
        self.noise_g = noise_g
        self.noise_diag = noise_diag
        if fullsky:
            self.mask = np.ones(hp.nside2npix(self.nside))
        else:
            self.mask = self.noise.polmask("ring")
        self.fullsky = fullsky
    
    def QU(self, band, idx=None, unit="uK", order="ring",):
        idx=idx + self.ssim
        QU = self.CMB.QU(idx=idx,)
        if not self.noise_g:
            noise = self.noise.noisemap(band,idx=idx,order="ring",unit="uK")
        else:
            noise = NoiseModelDiag(self.nside).noisemap()
        if self.add_fg:
            fg = self.FG(band).QU(band)
            QU += fg
        if self.add_noise:
            QU += noise
        
        if unit == "uK":
            pass
        elif unit == "K":
            QU *= 1e-6
        else:
            raise ValueError("unit must be uK or K")
        QU = QU * self.mask
        if order == "ring":
            pass
        elif order == "nested":
            QU = hp.reorder(QU, r2n=True)
        else:
            raise ValueError("order must be ring or nested")
        return QU

    def Emode(self, band, idx=None,):
        QU = self.QU(band, idx=idx)
        return hp.map2alm_spin(QU, 2, lmax=self.CMB.lmax)[0]



if __name__ == '__main__':
    import argparse
    



