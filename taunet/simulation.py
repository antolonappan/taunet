import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import camb
import healpy as hp
import pysm3
import pysm3.units as u
from taunet.Noise.ncm import NoiseModel
import os
import pickle as pl


class CMBspectra:

    def __init__(self,H0=67.32,ombh2=0.02237,omch2=0.1201,ns=0.9651,mnu=0.06,tau=0.06) -> None:
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

    def __init__(self,libdir,nsim,tau):
        self.libdir = os.path.join(libdir,"CMB")
        os.makedirs(self.libdir,exist_ok=True)
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
            seed = 261092 + idx
        np.random.seed(seed)
        Elm = hp.synalm(self.EE, lmax=100, new=True)
        hp.almxfl(Elm, self.synthetic_beam(lmax=100), inplace=True)
        return Elm
    
    def QU(self,idx=None):
        fname = os.path.join(self.libdir,"QU.pkl" if idx is None else f"QU_{idx:04d}.pkl")
        if os.path.isfile(fname):
            QU = pl.load(open(fname,"rb"))
        else:
            alm = self.alm(idx=idx)
            dummy = np.zeros_like(alm)
            TQU = hp.alm2map([dummy,alm,dummy], self.NSIDE)
            QU = TQU[1:].copy()
            del (alm,dummy,TQU)
            pl.dump(QU,open(fname,"wb"))

        return QU
    
    def Emode(self,idx=None):
        QU = self.QU(idx=idx)
        return hp.map2alm_spin(QU,2,lmax=self.lmax)[0]
       
        

class FGMap:

    def __init__(self,libdir,model=["d1","s1"]):
        self.libdir = os.path.join(libdir,"FG","".join(model))
        os.makedirs(self.libdir,exist_ok=True)
        self.model = model
        self.NSIDE = 16
        self.lmax = 3*self.NSIDE-1

    def QU(self,band,seed=None):
        seed = 261092 if seed is None else seed
        fname = os.path.join(self.libdir,f"QU_{seed}_{band}.pkl")
        if os.path.isfile(fname):
            QU = pl.load(open(fname,"rb"))
        else:
            sky = pysm3.Sky(nside=self.NSIDE, preset_strings=self.model)
            maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
            QU = maps[1:].copy()
            pl.dump(QU,open(fname,"wb"))
            del maps
        return QU.value
    
    def Emode(self,band,seed=None):
        QU = self.QU(band,seed=seed)
        return hp.map2alm_spin(QU,2,lmax=self.lmax)[0]
    

class SkySimulation:

    def __init__(self,libdir,nsim,tau,fg,cmb_const=True,fg_const=True):
        if cmb_const:
            self.qu_cmb = CMBmap(libdir,nsim,tau).QU()
        if fg_const:
            self.qu_fg_100 = FGMap(libdir,fg).QU(100)
            self.qu_fg_143 = FGMap(libdir,fg).QU(143)
        
        self.CMB = CMBmap(libdir,nsim,tau)
        self.FG = FGMap(libdir,fg)
        self.noise = NoiseModel()
        self.nsim = nsim
        self.cmb_const = cmb_const
        self.fg_const = fg_const

    
    def QU(self,band,idx=None):
        if self.cmb_const:
            cmb = self.qu_cmb
        else:
            cmb = self.CMB.QU(idx=idx)
        if self.fg_const:
            if band == 100:
                fg = self.qu_fg_100
            elif band == 143:
                fg = self.qu_fg_143
            else:
                raise ValueError("Band should be 100 or 143")
        else:
            fg = self.FG.QU(band)
        
        noise = self.noise.noisemap(band)
        QU = cmb + fg + noise
        return QU*self.noise.polmask
    
    def Emode(self,band,idx=None):
        QU = self.QU(band,idx=idx)
        return hp.map2alm_spin(QU,2,lmax=self.CMB.lmax)[0]