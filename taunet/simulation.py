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

class MakeSims:
    """
    Make simulations class to generate CMB and foreground maps


    Parameters
    ----------
    out_dir : str
        Output directory to save the maps
    fg : list, optional (default=["s0","d0"])
        Foreground model to use
    nside : int, optional (default=16)
        Healpix resolution
    noise_g : bool, optional (default=False)
        Use Gaussian noise
    noise_diag : bool, optional (default=False)
        Use diagonal noise
    noise_method : str, optional (default="roger")
        Noise method to use
    tau : float, optional (default=0.06)
        Optical depth
    nsim : int, optional (default=100)
        Number of simulations
    ssim : int, optional (default=0)
        Starting index for the simulations
    fullsky : bool, optional (default=False)
        Use full-sky maps
    
    """
    def __init__(
        self,
        out_dir,
        fg=["s0", "d0"],
        nside=16,
        noise_g=False,
        noise_diag=False,
        noise_method="roger",
        tau=0.06,
        nsim=100,
        ssim=0,
        fullsky=False,
    ):
        out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        if fullsky:
            simdir = os.path.join(
                out_dir, "SIMULATIONS_" + "".join(fg) + f"N_{int(noise_g)}_fullsky{'' if noise_method=='roger' else noise_method}"
            )
        else:
            simdir = os.path.join(
                out_dir, "SIMULATIONS_" + "".join(fg) + f"N_{int(noise_g)}{'' if noise_method=='roger' else noise_method}"
            )
        if noise_diag:
            simdir += "_diag"

        self.noise_diag = noise_diag
        os.makedirs(simdir, exist_ok=True)
        spectra_dir = os.path.join(out_dir, "SPECTRA")
        os.makedirs(spectra_dir, exist_ok=True)

        self.simdir = simdir

        spectra = CMBspectra(tau=tau)
        spefname = spectra.save_power(spectra_dir, retfile=True)
        print("Saved power spectra")

        self.spectrafile = spefname

        assert len(fg) == 2, "fg must be a list of length 2"
        fg = fg
        self.fg = fg
        noise_g = noise_g
        self.noise_g = noise_g
        nsim = nsim
        self.nsim = nsim
        self.ssim = ssim
        self.tau = tau
        self.noise_method = noise_method

        sky = SkySimulation(
            out_dir,
            tau,
            fg=fg,
            noise_g=noise_g,
            noise_diag=noise_diag,
            noise_method=noise_method,
            nsim=nsim,
            fullsky=fullsky,
        )

        self.sky = sky

        for f in [23, 100, 143, 353]:
            for i in tqdm(range(nsim), desc=f"Generating {f} GHz maps", unit="sim"):
                fname = os.path.join(simdir, f"sky_{f}_{i+self.ssim:06d}.fits")
                if not os.path.isfile(fname):
                    Q, U = sky.QU(
                        f,
                        idx=i + self.ssim,
                        unit="uK" if (fullsky or noise_g) else "K",
                        order="nested",
                    )
                    hp.write_map(fname, [Q * 0, Q, U], nest=True, dtype=np.float64)

        if fullsky:
            clean_dir = os.path.join(
                out_dir, "CLEAN_" + "".join(fg) + f"N_{int(noise_g)}_fullsky{'' if noise_method=='roger' else noise_method}"
            )
        else:
            clean_dir = os.path.join(
                out_dir, "CLEAN_" + "".join(fg) + f"N_{int(noise_g)}{'' if noise_method=='roger' else noise_method}"
            )

        if noise_diag:
            clean_dir += "_diag"

        os.makedirs(clean_dir, exist_ok=True)
        self.clean_dir = clean_dir

        if noise_g:
            ncm_dir = os.path.join(out_dir, "NCMG")
            self.ncm_dir = ncm_dir
            os.makedirs(ncm_dir, exist_ok=True)
            print("Generating noise covariance matrices: NoiseModelGaussian")
            for f in [23, 100, 143, 353]:
                fname = os.path.join(ncm_dir, "ncm_{}.bin".format(f))
                if os.path.isfile(fname):
                    continue
                ncm = NoiseModelDiag(16)
                cov = ncm.ncm("uK")
                cov.tofile(fname)
                print("Saved {}".format(fname))
        else:
            if noise_diag:
                _ncm_dir_ = f"NCMD{'' if noise_method=='roger' else noise_method}" 
            else:
                _ncm_dir_ = f"NCM{'' if noise_method=='roger' else noise_method}"
            ncm_dir = os.path.join(out_dir, _ncm_dir_)
            self.ncm_dir = ncm_dir
            os.makedirs(ncm_dir, exist_ok=True)
            print(f"Generating noise covariance matrices: NoiseModel{' Diag' if noise_diag else ''}")
            ncm = NoiseModel(self.noise_diag)
            freqs = [23, 100, 143, 353] if noise_diag else [100, 143, 353]
            for f in freqs:
                fname = os.path.join(ncm_dir, "ncm_{}.bin".format(f))
                if os.path.isfile(fname):
                    continue
                cov = ncm.get_full_ncm(
                    f, unit="K", pad_temp=True, reshape=True, order="ring"
                )
                cov.tofile(fname)
                print("Saved {}".format(fname))

        if fullsky:
            fmask = os.path.join(out_dir, "mask_fullsky.fits")
        else:
            fmask = os.path.join(out_dir, "mask.fits")
        self.maskpath = fmask
        if not os.path.isfile(fmask):
            print("Generating mask")
            if fullsky:
                mask = np.ones(hp.nside2npix(16))
                hp.write_map(fmask, [mask, mask, mask], nest=True)
            else:
                ncm = NoiseModel()
                mask = ncm.polmask("nested")
                hp.write_map(fmask, [mask, mask, mask], nest=True)
            print("Saved {}".format(fmask))

        self.fullsky = fullsky

    def make_params(self, band, dire="./", ret=False):
        assert band in [100, 143], "Band must be 143 or 100"

        if self.fullsky:
            fname = os.path.join(
                dire,
                f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}_fullsky{'' if not self.noise_diag else '_Diag'}{'' if self.noise_method=='roger' else self.noise_method}.ini",
            )
        else:
            fname = os.path.join(
                dire, f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}{'' if not self.noise_diag else '_Diag'}{'' if self.noise_method=='roger' else self.noise_method}.ini"
            )

        if self.noise_g or self.fullsky:
            nab = 80
            calib = 1e0
        else:
            calib = 1e6
            nab = 80#320

        if self.noise_g or self.noise_diag:
            ncvmfilesync = os.path.join(self.ncm_dir, "ncm_23.bin")
        else:
            ncvmfilesync = "/marconi/home/userexternal/aidicher/luca/wmap/wmap_K_coswin_ns16_9yr_v5_covmat.bin"

        if band == 100:
            bega = 0.010
            enda = 0.020
            begb = 0.016
            endb = 0.028
            # bega=0.005
            # enda=0.015
            # begb=0.015
            # endb=0.022
        elif band == 143:
            bega = 0.000
            enda = 0.010
            begb = 0.035
            endb = 0.045
            # bega=0.000
            # enda=0.015
            # begb=0.038
            # endb=0.041

        params = f"""maskfile={self.maskpath}
ncvmfilesync={ncvmfilesync}
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
calibration= {calib}
do_signal_covmat=T
calibration_S_cov=1.0

use_beam_file=T
beam_file=/marconi/home/userexternal/aidicher/luca/lowell-likelihood-analysis/ancillary/beam_440T_coswinP_pixwin16.fits
regularization_noise_S_covmat=0.0
do_likelihood_marginalization=F

na={nab}
nb={nab}

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
            if self.fullsky:
                return (
                    f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}_fullsky{'' if not self.noise_diag else '_Diag'}{'' if self.noise_method=='roger' else self.noise_method}.ini"
                )
            else:
                return f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}{'' if not self.noise_diag else '_Diag'}{'' if self.noise_method=='roger' else self.noise_method}.ini"

    def job_file(self, band, dire="./", ret=False):
        assert band in [100, 143], "Band must be 143 or 100"
        pname = self.make_params(band, dire=dire, ret=True)
        if self.fullsky:
            fname = os.path.join(
                dire, f"slurm_{''.join(self.fg)}_N{int(self.noise_g)}_{band}_fullsky{'' if not self.noise_diag else '_Diag'}{'' if self.noise_method=='roger' else self.noise_method}.sh"
            )
        else:
            fname = os.path.join(
                dire, f"slurm_{''.join(self.fg)}_N{int(self.noise_g)}_{band}{'' if not self.noise_diag else '_Diag'}{'' if self.noise_method=='roger' else self.noise_method}.sh"
            )

        if self.fullsky:
            q = "skl_usr_dbg"
            ti = "00:30:00"
        else:
            q = "skl_usr_dbg"
            ti = "00:30:00"

        if not os.path.isfile(fname):
            slurm = f"""#!/bin/bash -l

#SBATCH -p {q}
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH -t {ti}
#SBATCH -J test
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH -A INF24_litebird
#SBATCH --export=ALL
#SBATCH --mem=182000
#SBATCH --mail-type=ALL

source ~/.bash_profile
cd /marconi/home/userexternal/aidicher/workspace/taunet/taunet/template_fitting
export OMP_NUM_THREADS=1

srun grid_compsep_mpi.x {pname}
"""
            f = open(fname, "wt")
            f.write(slurm)
            f.close()

        if ret:
            return fname

    def submit_job(self, band, dire="./"):
        fname = self.job_file(band, dire=dire, ret=True)
        os.system(f"sbatch {fname}")

    def anl_cleaned(self, nsim=20, ret_full=False, ret_cmb=True):
        cmb = self.sky.CMB
        if ret_cmb:
            Earr = []
            for i in tqdm(range(nsim)):
                CMB_QU = cmb.QU(idx=i, beam=True)
                cmb_alm = hp.map2alm_spin(CMB_QU, spin=2)
                ee = hp.alm2cl(cmb_alm[0])
                Earr.append(ee)
            Earr = np.array(Earr)
            Emean = np.mean(Earr, axis=0)
            Estd = np.std(Earr, axis=0)
        cl = []
        for i in tqdm(range(nsim)):
            fname1 = os.path.join(self.clean_dir, f"cleaned_100_{i:06d}.fits")
            fname2 = os.path.join(self.clean_dir, f"cleaned_143_{i:06d}.fits")
            QU1 = hp.read_map(fname1, field=(1, 2))
            QU2 = hp.read_map(fname2, field=(1, 2))
            E1, _ = hp.map2alm_spin(QU1, spin=2)
            E2, _ = hp.map2alm_spin(QU2, spin=2)
            ee = hp.alm2cl(E1, E2)
            cl.append(ee)
        cl = np.array(cl)
        clmean = np.mean(cl, axis=0)
        clstd = np.std(cl, axis=0)
        if ret_full:
            if ret_cmb:
                return Earr, cl
            else:
                return cl
        else:
            return Emean, Estd, clmean, clstd

    def plot_cleaned(self, nsim=20, unit="K"):
        if unit == "uK":
            fac = 1
        elif unit == "K":
            fac = 1e12
        else:
            raise ValueError("unit must be uK or K")

        Emean, Estd, clmean, clstd = self.anl_cleaned(nsim=nsim)
        ell = np.arange(len(Emean))
        spectra = CMBspectra(tau=self.tau)
        plt.figure()
        plt.loglog(ell, spectra.EE[ell], label="Signal")
        plt.errorbar(ell, Emean, yerr=Estd, fmt="o", label="CMB")
        if self.fullsky:
            plt.errorbar(ell, clmean, yerr=clstd, fmt="o", label="Cleaned")
        else:
            plt.errorbar(
                ell,
                clmean * fac / 0.54,
                yerr=clstd * fac / 0.54,
                fmt="o",
                label="Cleaned",
            )
        plt.legend()



