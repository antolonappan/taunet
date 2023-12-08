import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import camb
import healpy as hp
import pysm3
import pysm3.units as u
from taunet.ncm import NoiseModel
import os
import pickle as pl

cosbeam = hp.read_cl('/marconi/home/userexternal/aidicher/luca/lowell-likelihood-analysis/ancillary/beam_coswin_ns16.fits')[0]

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

class CMBspectra:

    def __init__(self,H0=67.32,ombh2=0.02237,omch2=0.1201,ns=0.9651,mnu=0.06,tau=0.06) -> None:
        pars = camb.CAMBparams()
        self.tau = tau
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu,tau=tau)
        pars.InitPower.set_params(ns=ns,r=0)
        pars.set_for_lmax(284, lens_potential_accuracy=0)
        self.results = camb.get_results(pars)
        self.powers = self.results.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=True)
        self.EE = self.powers[:,1]
        self.ell = np.arange(len(self.EE))
    
    def tofile(self,libdir):
        fname = os.path.join(libdir,f"lensed_scalar_cls_{str(self.tau).replace('.','p')}.dat")
        powers = self.results.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=False)
        powers = powers[2:,:]
        lmax = len(powers)
        l = np.arange(2,lmax+2)
        powers = np.column_stack((l.reshape(-1),powers))
        np.savetxt(fname,powers)
        

    
    def plot(self):
        plt.loglog(self.EE)
        plt.show()
    
    def save_power(self,libdir):
        fname = os.path.join(libdir,f"lensed_scalar_cls_{str(self.tau).replace('.','p')}.dat")
        np.savetxt(fname,self.powers)


class CMBmap:

    def __init__(self,libdir,nsim,tau):
        self.libdir = os.path.join(libdir,"CMB")
        os.makedirs(self.libdir,exist_ok=True)
        self.nsim = nsim
        self.tau = tau
        self.EE = CMBspectra(tau=tau).EE
        self.NSIDE = 16
        self.lmax = 3*self.NSIDE-1
    
    def alm(self,idx=None):
        if idx is None:
            seed = 261092
        else:
            seed = 261092 + idx
        np.random.seed(seed)
        Elm = hp.synalm(self.EE, lmax=100, new=True)
        return Elm
    
    def QU(self,idx=None,beam=False):
        fname = os.path.join(self.libdir,f"QU_b{int(beam)}.pkl" if idx is None else f"QU_b{int(beam)}_{idx:04d}.pkl")
        if os.path.isfile(fname):
            QU = pl.load(open(fname,"rb"))
        else:
            alm = self.alm(idx=idx)
            if beam:
                hp.almxfl(alm,cosbeam,inplace=True)
            dummy = np.zeros_like(alm)
            TQU = hp.alm2map([dummy,alm,dummy], self.NSIDE)
            QU = TQU[1:].copy()
            del (alm,dummy,TQU)
            pl.dump(QU,open(fname,"wb"))

        return QU
    
    def Emode(self,idx=None,beam=False):
        QU = self.QU(idx=idx,beam=beam)
        return hp.map2alm_spin(QU,2,lmax=self.lmax)[0]
       
        

class FGMap:

    def __init__(self,libdir,model=["d1","s1"]):
        self.libdir = os.path.join(libdir,"FG","".join(model))
        os.makedirs(self.libdir,exist_ok=True)
        self.model = model
        self.NSIDE = 16
        self.lmax = 3*self.NSIDE-1

    def QU(self,band,seed=None,beam=False):
        seed = 261092 if seed is None else seed
        fname = os.path.join(self.libdir,f"QU_b{int(beam)}_{seed}_{band}.pkl")
        if os.path.isfile(fname):
            QU = pl.load(open(fname,"rb"))
        else:
            sky = pysm3.Sky(nside=self.NSIDE, preset_strings=self.model)
            maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
            alms = hp.map2alm(maps.value, lmax=self.lmax)
            if beam:
                hp.almxfl(alms[1],cosbeam,inplace=True)
                hp.almxfl(alms[2],cosbeam,inplace=True)
            TQU = hp.alm2map(alms, self.NSIDE)
            QU = TQU[1:].copy()
            pl.dump(QU,open(fname,"wb"))
            del maps
        return QU
    
    def Emode(self,band,seed=None,mask=None,beam=False):
        QU = self.QU(band,seed=seed,beam=beam)
        if mask is not None:
            QU = QU*mask
        return hp.map2alm_spin(QU,2,lmax=self.lmax)[0]
    

class SkySimulation:

    def __init__(self,libdir,nsim,tau,fg,cmb_const=True,fg_const=True):
        if cmb_const:
            self.qu_cmb = CMBmap(libdir,nsim,tau).QU()
        if fg_const:
            self.qu_fg_30 = FGMap(libdir,fg).QU(30)
            self.qu_fg_100 = FGMap(libdir,fg).QU(100)
            self.qu_fg_143 = FGMap(libdir,fg).QU(143)
            self.qu_fg_353 = FGMap(libdir,fg).QU(353)
        
        self.CMB = CMBmap(libdir,nsim,tau)
        self.FG = FGMap(libdir,fg)
        self.noise = NoiseModel()
        self.nsim = nsim
        self.cmb_const = cmb_const
        self.fg_const = fg_const



    
    def apply_beam(self,QU):
        beam = self.get_beam()
        TQU = [np.zeros_like(QU[0]),QU[0],QU[1]] 
        alms = hp.map2alm(TQU, lmax=self.CMB.lmax)
        hp.almxfl(alms[1], beam, inplace=True)
        hp.almxfl(alms[2], beam, inplace=True)
        return hp.alm2map(alms, self.CMB.NSIDE, verbose=False)[1:]

    def QU(self,band,idx=None,unit='uK',order='ring',beam=True,deconvolve=False,):
        if self.cmb_const:
            cmb = self.qu_cmb
        else:
            cmb = self.CMB.QU(idx=idx,beam=True)
        if self.fg_const:
            if band == 30:
                fg = self.qu_fg_30
            elif band == 100:
                fg = self.qu_fg_100
            elif band == 143:
                fg = self.qu_fg_143
            elif band == 353:
                fg = self.qu_fg_353
            else:
                raise ValueError("Band should be 100 or 143")
        else:
            fg = self.FG.QU(band,beam=True)
        
        noise = self.noise.noisemap(band,'ring','uK',deconvolve=deconvolve)
        QU = cmb + fg + noise
        #QU = self.apply_beam(QU)*self.noise.polmask('ring')
        if unit=='uK':
            pass
        elif unit=='K':
            QU *= 1e-6
        else:
            raise ValueError('unit must be uK or K')
        QU = QU * self.noise.polmask('ring')
        if order=='ring':
            pass
        elif order=='nested':
            QU = hp.reorder(QU,r2n=True)
        else:
            raise ValueError('order must be ring or nested')
        return QU 

    
    def Emode(self,band,idx=None,beam=True,deconvolve=False):
        QU = self.QU(band,idx=idx,beam=beam,deconvolve=deconvolve)
        return hp.map2alm_spin(QU,2,lmax=self.CMB.lmax)[0]