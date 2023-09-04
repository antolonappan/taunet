import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import camb
import healpy as hp
import pysm3
import pysm3.units as u


class CMBspectra:

    def __init__(self, H0=67.32,ombh2=0.02237,omch2=0.1201,ns=0.9651,mnu=0.06,tau=0.06) -> None:
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu,tau=tau)
        pars.InitPower.set_params(ns=ns,r=0)
        pars.set_for_lmax(100, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        powers = results.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=True)
        self.EE = powers[:,1]
    
    def plot(self):
        plt.loglog(self.EE)
        plt.show()


class CMBmap:

    def __init__(self,nsim,tau):
        self.nsim = nsim
        self.tau = tau
        self.EE = CMBspectra(tau=tau).EE
        self.NSIDE = 16
        self.lmax = 3*self.NSIDE-1

    def synthetic_beam(self,lmax=60):
        bl = np.zeros(lmax + 1)
        bl[:41] = 1.0
        bl[41:49] = (1 + np.cos(np.arange(41, 49) - 40) * (np.pi / 8)) / 2.0
        return bl
    
    def alm(self,idx=None):
        if idx is None:
            seed = 261092
        else:
            raise NotImplementedError
        np.random.seed(seed)
        Elm = hp.synalm(self.EE, lmax=100, new=True)
        hp.almxfl(Elm, self.synthetic_beam(lmax=100), inplace=True)
        return Elm
    
    def QU(self,idx=None):
        alm = self.alm(idx=idx)
        dummy = np.zeros_like(alm)
        TQU = hp.alm2map([dummy,alm,dummy], self.NSIDE)
        QU = TQU[1:].copy()
        del (alm,dummy,TQU)
        return QU
       
        

class FGround:

    def __init__(self,model=["d1","s1"]):
        self.NSIDE = 16
        self.lmax = 3*self.NSIDE-1
        self.sky = pysm3.Sky(nside=self.NSIDE, preset_strings=model)

    def QU(self,band):
        maps = self.sky.get_emission(band * u.GHz)
        maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
        QU = maps[1:].copy()
        del maps
        return QU
    

    



        
