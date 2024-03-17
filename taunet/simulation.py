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
import subprocess

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
        self, H0=67.32, ombh2=0.02237, omch2=0.1201, ns=0.9651, mnu=0.06, tau=0.06
    ) -> None:
        pars = camb.CAMBparams()
        self.tau = tau
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, tau=tau)
        pars.InitPower.set_params(ns=ns, r=0)
        pars.set_for_lmax(284, lens_potential_accuracy=0)
        self.results = camb.get_results(pars)
        self.powers = self.results.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=True)
        self.EE = self.powers[:, 1]
        self.ell = np.arange(len(self.EE))

    def tofile(self, libdir, retfile=False):
        fname = os.path.join(
            libdir, f"lensed_scalar_cls_{str(self.tau).replace('.','p')}.dat"
        )
        powers = self.results.get_lensed_scalar_cls(CMB_unit="muK", raw_cl=False)
        powers = powers[2:, :]
        lmax = len(powers)
        l = np.arange(2, lmax + 2)
        powers = np.column_stack((l.reshape(-1), powers))
        np.savetxt(fname, powers)
        if retfile:
            return fname

    def plot(self):
        plt.loglog(self.EE)
        plt.show()

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

    def __init__(self, libdir, tau, nsim=None):
        self.libdir = os.path.join(libdir, "CMB")
        self.specdir = os.path.join(libdir, "SPECTRA")
        os.makedirs(self.libdir, exist_ok=True)
        os.makedirs(self.specdir, exist_ok=True)
        self.nsim = nsim
        self.tau = tau
        print("tau = {}".format(tau))
        self.NSIDE = 16
        self.lmax = 3 * self.NSIDE - 1

    @property
    def EE(self):
        fname = os.path.join(self.specdir, f"spectra_tau_{self.tau}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname, "rb"))
        else:
            ee = CMBspectra(tau=self.tau).EE
            pl.dump(ee, open(fname, "wb"))
            return ee

    def alm(self, idx=None):
        if idx is None:
            seed = 261092
        else:
            seed = 261092 + idx
        np.random.seed(seed)
        Elm = hp.synalm(self.EE, lmax=100, new=True)
        return Elm

    def QU(self, idx=None, beam=False):
        alm = self.alm(idx=idx)
        if beam:
            hp.almxfl(alm, cosbeam, inplace=True)
        dummy = np.zeros_like(alm)
        TQU = hp.alm2map([dummy, alm, dummy], self.NSIDE)
        QU = TQU[1:].copy()
        del (alm, dummy, TQU)

        return QU

    def Emode(self, idx=None, beam=False):
        QU = self.QU(idx=idx, beam=beam)
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

    def __init__(self, libdir, model=["d1", "s1"]):
        self.libdir = os.path.join(libdir, "FG", "".join(model))
        os.makedirs(self.libdir, exist_ok=True)
        self.model = model
        self.NSIDE = 16
        self.lmax = 3 * self.NSIDE - 1

    def QU(self, band, seed=None, beam=True):
        seed = 261092 if seed is None else seed
        fname = os.path.join(self.libdir, f"QU_b{int(beam)}_{seed}_{band}.pkl")
        if os.path.isfile(fname):
            QU = pl.load(open(fname, "rb"))
        else:
            sky = pysm3.Sky(nside=self.NSIDE, preset_strings=self.model)
            maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band * u.GHz))
            alms = hp.map2alm(maps.value, lmax=self.lmax)
            if beam:
                hp.almxfl(alms[1], cosbeam, inplace=True)
                hp.almxfl(alms[2], cosbeam, inplace=True)
            TQU = hp.alm2map(alms, self.NSIDE)
            QU = TQU[1:].copy()
            pl.dump(QU, open(fname, "wb"))
            del maps
        return QU

    def Emode(self, band, seed=None, mask=None, beam=False):
        QU = self.QU(band, seed=seed, beam=beam)
        if mask is not None:
            QU = QU * mask
        return hp.map2alm_spin(QU, 2, lmax=self.lmax)[0]


class SkySimulation:
    """
    SkySimulation class to generate CMB and foreground maps

    Parameters
    ----------
    libdir : str
        Library directory to save the maps
    tau : float
        Optical depth
    add_noise : bool, optional (default=True)
        Add noise to the maps
    add_fg : bool, optional (default=True)
        Add foregrounds to the maps
    fg : list, optional (default=["s0","d0"])
        Foreground model to use
    nsim : int, optional (default=None)
        Number of simulations
    ssim : int, optional (default=0)
        Starting index for the simulations
    fg_const : bool, optional (default=True)
        Use constant foregrounds
    noise_g : bool, optional (default=False)
        Use Gaussian noise
    noise_diag : bool, optional (default=False)
        Use diagonal noise
    fullsky : bool, optional (default=False)
        Use full-sky maps
    """

    def __init__(
        self,
        libdir: str,
        tau: float,
        add_noise: bool = True,
        add_fg: bool = True,
        fg: list = ["s0", "d0"],
        nsim=None,
        ssim: int = 0,
        fg_const: bool = True,
        noise_g: bool = False,
        noise_diag: bool = False,
        fullsky: bool = False,
    ):

        if fg_const:
            self.qu_fg_23 = FGMap(libdir, fg).QU(23) if add_fg else None
            self.qu_fg_30 = FGMap(libdir, fg).QU(30) if add_fg else None
            self.qu_fg_100 = FGMap(libdir, fg).QU(100) if add_fg else None
            self.qu_fg_143 = FGMap(libdir, fg).QU(143) if add_fg else None
            self.qu_fg_353 = FGMap(libdir, fg).QU(353) if add_fg else None

        self.CMB = CMBmap(libdir, tau, nsim)
        self.FG = FGMap(libdir, fg)
        self.noise = NoiseModel(noise_diag)
        self.nsim = nsim
        self.ssim = ssim
        self.nside = self.CMB.NSIDE
        self.fg_const = fg_const
        self.add_fg = add_fg
        self.add_noise = add_noise
        self.noise_g = noise_g
        self.noise_diag = noise_diag
        if fullsky:
            self.mask = np.ones(hp.nside2npix(self.nside))
        else:
            self.mask = self.noise.polmask("ring")

        self.fullsky = fullsky

    def apply_beam(self, QU):
        beam = self.get_beam()
        TQU = [np.zeros_like(QU[0]), QU[0], QU[1]]
        alms = hp.map2alm(TQU, lmax=self.CMB.lmax)
        hp.almxfl(alms[1], beam, inplace=True)
        hp.almxfl(alms[2], beam, inplace=True)
        return hp.alm2map(alms, self.CMB.NSIDE, verbose=False)[1:]

    def QU(self, band, idx=None, unit="uK", order="ring", beam=True, deconvolve=False):
        idx=idx + self.ssim
        cmb = self.CMB.QU(idx=idx, beam=True)
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
            fg = self.FG.QU(band, beam=True)

        if not self.noise_g:
            noise = self.noise.noisemap(band,idx=idx,order="ring",unit="uK",deconvolve=deconvolve)
        else:
            noise = NoiseModelDiag(self.nside).noisemap()

        QU = cmb
        if self.add_fg:
            QU += fg
        if self.add_noise:
            QU += noise
        # QU = self.apply_beam(QU)*self.noise.polmask('ring')
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

    def Emode(self, band, idx=None, beam=True, deconvolve=False):
        QU = self.QU(band, idx=idx, beam=beam, deconvolve=deconvolve)
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
        tau=0.06,
        nsim=100,
        ssim=0,
        fullsky=False,
    ):
        out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        if fullsky:
            simdir = os.path.join(
                out_dir, "SIMULATIONS_" + "".join(fg) + f"N_{int(noise_g)}_fullsky"
            )
        else:
            simdir = os.path.join(
                out_dir, "SIMULATIONS_" + "".join(fg) + f"N_{int(noise_g)}"
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

        sky = SkySimulation(
            out_dir,
            tau,
            fg=fg,
            noise_g=noise_g,
            noise_diag=noise_diag,
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
                out_dir, "CLEAN_" + "".join(fg) + f"N_{int(noise_g)}_fullsky"
            )
        else:
            clean_dir = os.path.join(
                out_dir, "CLEAN_" + "".join(fg) + f"N_{int(noise_g)}"
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
            ncm_dir = os.path.join(out_dir, "NCMD" if noise_diag else "NCM")
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
                f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}_fullsky{'' if not self.noise_diag else '_Diag'}.ini",
            )
        else:
            fname = os.path.join(
                dire, f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}{'' if not self.noise_diag else '_Diag'}.ini"
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
                    f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}_fullsky{'' if not self.noise_diag else '_Diag'}.ini"
                )
            else:
                return f"params_{''.join(self.fg)}_N{int(self.noise_g)}_{band}{'' if not self.noise_diag else '_Diag'}.ini"

    def job_file(self, band, dire="./", ret=False):
        assert band in [100, 143], "Band must be 143 or 100"
        pname = self.make_params(band, dire=dire, ret=True)
        if self.fullsky:
            fname = os.path.join(
                dire, f"slurm_{''.join(self.fg)}_N{int(self.noise_g)}_{band}_fullsky{'' if not self.noise_diag else '_Diag'}.sh"
            )
        else:
            fname = os.path.join(
                dire, f"slurm_{''.join(self.fg)}_N{int(self.noise_g)}_{band}{'' if not self.noise_diag else '_Diag'}.sh"
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
