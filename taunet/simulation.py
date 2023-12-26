import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import camb
import healpy as hp
import pysm3
import pysm3.units as u
from taunet.ncm import NoiseModel,NoiseModelGaussian
from taunet import DATADIR
import os
import pickle as pl
import subprocess

cosbeam = hp.read_cl(os.path.join(DATADIR,'beam_coswin_ns16.fits'))[0]

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
    
    def tofile(self,libdir,retfile=False):
        fname = os.path.join(libdir,f"lensed_scalar_cls_{str(self.tau).replace('.','p')}.dat")
        powers = self.results.get_lensed_scalar_cls(CMB_unit='muK', raw_cl=False)
        powers = powers[2:,:]
        lmax = len(powers)
        l = np.arange(2,lmax+2)
        powers = np.column_stack((l.reshape(-1),powers))
        np.savetxt(fname,powers)
        if retfile:
            return fname
        

    
    def plot(self):
        plt.loglog(self.EE)
        plt.show()
    
    def save_power(self,libdir,retfile=False):
        fname = os.path.join(libdir,f"lensed_scalar_cls_{str(self.tau).replace('.','p')}.dat")
        if not os.path.isfile(fname):
            np.savetxt(fname,self.powers)
        if retfile:
            return fname


class CMBmap:

    def __init__(self,libdir,tau,nsim=None):
        self.libdir = os.path.join(libdir,"CMB")
        self.specdir = os.path.join(libdir,"SPECTRA")
        os.makedirs(self.libdir,exist_ok=True)
        os.makedirs(self.specdir,exist_ok=True)
        self.nsim = nsim
        self.tau = tau
        self.NSIDE = 16
        self.lmax = 3*self.NSIDE-1
    
    @property
    def EE(self):
        fname = os.path.join(self.specdir,f'spectra_tau_{self.tau}.pkl')
        if os.path.isfile(fname):
            return pl.load(open(fname,'rb'))
        else:
            ee = CMBspectra(tau=self.tau).EE
            pl.dump(ee,open(fname,'wb'))
            return ee
    
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

    def __init__(self,libdir,nsim,tau,fg,cmb_const=True,fg_const=True,noise_g=False):
        if cmb_const:
            self.qu_cmb = CMBmap(libdir,nsim,tau).QU(beam=True)
        if fg_const:
            self.qu_fg_23 = FGMap(libdir,fg).QU(23,beam=True)
            self.qu_fg_30 = FGMap(libdir,fg).QU(30,beam=True)
            self.qu_fg_100 = FGMap(libdir,fg).QU(100,beam=True)
            self.qu_fg_143 = FGMap(libdir,fg).QU(143,beam=True)
            self.qu_fg_353 = FGMap(libdir,fg).QU(353,beam=True)
        
        self.CMB = CMBmap(libdir,nsim,tau)
        self.FG = FGMap(libdir,fg)
        self.noise = NoiseModel()
        self.nsim = nsim
        self.nside = self.CMB.NSIDE
        self.cmb_const = cmb_const
        self.fg_const = fg_const

        self.noise_g = noise_g

    
    def apply_beam(self,QU):
        beam = self.get_beam()
        TQU = [np.zeros_like(QU[0]),QU[0],QU[1]] 
        alms = hp.map2alm(TQU, lmax=self.CMB.lmax)
        hp.almxfl(alms[1], beam, inplace=True)
        hp.almxfl(alms[2], beam, inplace=True)
        return hp.alm2map(alms, self.CMB.NSIDE, verbose=False)[1:]

    def QU(self,band,idx=None,unit='uK',order='ring',beam=True,deconvolve=False,nlevp=None):
        if self.cmb_const:
            cmb = self.qu_cmb
        else:
            cmb = self.CMB.QU(idx=idx,beam=True)
        if self.fg_const:
            if band == 23:
                fg = self.qu_fg_23
            elif band == 30:
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
        
        if not self.noise_g:
            noise = self.noise.noisemap(band,'ring','uK',deconvolve=deconvolve)
        else:
            assert nlevp is not None, "nlevp must be specified"
            noise = NoiseModelGaussian(self.nside,nlevp).noisemaps('uK')[1:]
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
    

class MakeSims:
    def __init__(self,out_dir,fg=['s0','d0'],nside=16,noise_g=False,tau=0.06,nsim=100,ssim=0):
        out_dir = out_dir
        os.makedirs(out_dir,exist_ok=True)
        simdir = os.path.join(out_dir,'SIMULATIONS_'+''.join(fg) + f'N_{int(noise_g)}')
        os.makedirs(simdir,exist_ok=True)
        spectra_dir = os.path.join(out_dir,'SPECTRA')
        os.makedirs(spectra_dir,exist_ok=True)

        self.simdir = simdir

        spectra = CMBspectra(tau=tau)
        spefname = spectra.save_power(spectra_dir,retfile=True)
        print("Saved power spectra")

        self.spectrafile = spefname

        assert len(fg) == 2, "fg must be a list of length 2"
        fg = fg

        self.fg = fg

        noise_g = noise_g

        self.noise_g = noise_g

        tau = tau
        nsim = nsim

        self.nsim  = nsim
        self.ssim = ssim
    
        noise_levels = {'23': 61,
                        '100': 55,
                        '143': 54,
                        '353': 60,}
        self.tau = tau
        
        sky = SkySimulation(out_dir,nsim,tau,fg,cmb_const=False,noise_g=noise_g)

        self.sky = sky

        for f in [23,100,143,353]:
            for i in tqdm(range(nsim),desc=f"Generating {f} GHz maps",unit='sim'):
                fname = os.path.join(simdir,f'sky_{f}_{i:06d}.fits')
                if not os.path.isfile(fname):
                    Q, U = sky.QU(f,idx=i,unit='K',order='nested',nlevp=noise_levels[str(f)])
                    hp.write_map(fname,[Q*0,Q,U],nest=True,dtype=np.float64)
        
        clean_dir = os.path.join(out_dir,'CLEAN_'+''.join(fg) + f'N_{int(noise_g)}')
        os.makedirs(clean_dir,exist_ok=True)

        self.clean_dir = clean_dir



        if noise_g:
            ncm_dir = os.path.join(out_dir,'NCMG')
            self.ncm_dir = ncm_dir
            os.makedirs(ncm_dir,exist_ok=True)
            print("Generating noise covariance matrices: NoiseModelGaussian")
            for f in [23,100,143,353]:
                fname = os.path.join(ncm_dir,'ncm_{}.bin'.format(f))
                if os.path.isfile(fname):
                    continue 
                ncm = NoiseModelGaussian(16,noise_levels[str(f)])
                cov = ncm.ncm('K')
                cov.tofile(fname)
                print("Saved {}".format(fname))
        else:
            ncm_dir = os.path.join(out_dir,'NCM')
            self.ncm_dir = ncm_dir
            os.makedirs(ncm_dir,exist_ok=True)
            print("Generating noise covariance matrices: NoiseModel")
            ncm = NoiseModel()
            for f in [100,143,353]:
                fname = os.path.join(ncm_dir,'ncm_{}.bin'.format(f))
                if os.path.isfile(fname):
                    continue
                cov = ncm.get_full_ncm(f,unit='K',pad_temp=True,reshape=True,order='ring')
                cov.tofile(fname)
                print("Saved {}".format(fname))
        
        fmask = os.path.join(out_dir,'mask.fits')
        self.maskpath = fmask
        if not os.path.isfile(fmask):
            print("Generating mask")
            ncm = NoiseModel()
            mask = ncm.polmask('nested')
            hp.write_map(fmask,[mask,mask,mask],nest=True)
            print("Saved {}".format(fmask))


    def make_params(self,band,dire='./',ret=False):
        assert band in [100,143], "Band must be 143 or 100"

        fname = os.path.join(dire,f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}.ini")

        if band == 100:
            bega=0.005
            enda=0.015
            begb=0.015
            endb=0.022
        elif band == 143:
            bega=0.000
            enda=0.015
            begb=0.038
            endb=0.041

        params = f"""maskfile={self.maskpath}
ncvmfilesync=/marconi/home/userexternal/aidicher/luca/wmap/wmap_K_coswin_ns16_9yr_v5_covmat.bin
ncvmfiledust={os.path.join(self.ncm_dir,'ncm_353.bin')}
ncvmfile={os.path.join(self.ncm_dir,f'ncm_{band}.bin')}
            
root_datasync={os.path.join(self.simdir,'sky_23_')}
root_datadust={os.path.join(self.simdir,'sky_353_')}
root_data={os.path.join(self.simdir,f'sky_{band}_')}

root_out_cleaned_map={os.path.join(self.clean_dir,f'cleaned_{band}_')}
file_scalings={os.path.join(self.clean_dir,'scalings.txt')}

fiducialfile={self.spectrafile}

ordering=1
nside= 16
lmax=64
calibration= 1e6
do_signal_covmat=T
calibration_S_cov=1.0

use_beam_file=T
beam_file=/marconi/home/userexternal/aidicher/luca/lowell-likelihood-analysis/ancillary/beam_coswin_ns16.fits
regularization_noise_S_covmat=0.020
do_likelihood_marginalization=F

na=320
nb=320

bega={bega}
enda={enda}
begb={begb}
endb={endb}
use_complete_covmat=T
minimize_chi2=T
output_clean_dataset=T


add_polarization_white_noise=F
output_covariance=F
template_marginalization=F
ssim={self.ssim}
nsim={self.nsim}

zerofill = 6
suffix_map = .fits 
"""
        f = open(fname, "wt")
        f.write(params)
        f.close()
        if ret:
            return f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}.ini"
    
    def job_file(self,band,dire='./',ret=False):
        assert band in [100,143], "Band must be 143 or 100"
        pname = self.make_params(band,dire=dire,ret=True)
        fname = os.path.join(dire,f"slurm_{''.join(self.fg)}_N{int(self.noise_g)}_{band}.sh")

        if not os.path.isfile(fname):
            slurm = f"""#!/bin/bash -l

#SBATCH -p skl_usr_dbg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -t 00:30:00
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -A INF23_litebird
#SBATCH --export=ALL
#SBATCH --mem=182000
#SBATCH --mail-type=ALL

source ~/.bash_profile
cd /marconi/home/userexternal/aidicher/workspace/taunet/taunet/template_fitting
export OMP_NUM_THREADS=2

srun grid_compsep_mpi.x {pname}
"""
            f = open(fname, "wt")
            f.write(slurm)
            f.close()

        if ret:
            return fname
    
    def submit_job(self,band,dire='./'):
        fname = self.job_file(band,dire=dire,ret=True)
        os.system(f"sbatch {fname}")

    
    def anal_cleaned(self,nsim=20):
        cmb = self.sky.CMB
        Earr = []
        for i in tqdm(range(nsim)):
            CMB_QU = cmb.QU(idx=i,beam=True)
            cmb_alm = hp.map2alm_spin(CMB_QU,spin=2)
            ee = hp.alm2cl(cmb_alm[0])
            Earr.append(ee)
        Earr = np.array(Earr)
        Emean = np.mean(Earr,axis=0)
        Estd = np.std(Earr,axis=0)
        cl = []
        for i in tqdm(range(nsim)):
            fname1 = os.path.join(self.clean_dir,f'cleaned_100_{i:06d}.fits')
            fname2 = os.path.join(self.clean_dir,f'cleaned_143_{i:06d}.fits')
            QU1 = hp.read_map(fname1,field=(1,2))
            QU2 = hp.read_map(fname2,field=(1,2))
            E1,_ = hp.map2alm_spin(QU1,spin=2)
            E2,_ = hp.map2alm_spin(QU2,spin=2)
            ee = hp.alm2cl(E1,E2)
            cl.append(ee)
        cl = np.array(cl)
        clmean = np.mean(cl,axis=0)
        clstd = np.std(cl,axis=0)
        return Emean,Estd,clmean,clstd
    
    def plot_cleaned(self,nsim=20):
        Emean,Estd,clmean,clstd = self.anal_cleaned(nsim=nsim)
        ell = np.arange(len(Emean))
        spectra = CMBspectra(tau=self.tau)
        plt.figure()
        plt.loglog(ell,spectra.EE[ell],label='Signal')
        plt.errorbar(ell,Emean,yerr=Estd,fmt='o',label='CMB')
        plt.errorbar(ell,clmean*1e12/0.54,yerr=clstd*1e12/0.54,fmt='o',label='Cleaned')
        plt.legend()
    
    




    

