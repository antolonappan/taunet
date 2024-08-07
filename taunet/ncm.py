import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from taunet import DATADIR, DBDIR
import pickle as pkl
from numba import njit,f8
from warnings import warn
from taunet import mpi



def cho2map(cho):
    pix = cho.shape[0]
    noisem = np.random.normal(0,1,pix)
    noisemap = np.dot(cho, noisem)
    return noisemap
    
    
try:
    cosbeam = hp.read_cl(os.path.join(DATADIR,'beam_440T_coswinP_pixwin16.fits'))[1]
except FileNotFoundError:
    cosbeam = hp.read_cl(os.path.join(DATADIR,'beam_coswin_ns16.fits'))[0]

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

class NoiseModel:

    def __init__(self,diag=False,method='sroll'):
        assert method in ['sroll','ffp8'], 'method must be one of sroll, ffp8 '
        self.nside = 16
        self.npix = 12*self.nside**2
        self.dtype = np.float32
        self.diag = diag

        self.basedir = os.path.join(DBDIR,f"Noise{'_diag' if diag else ''}")

        # directory
        self.__cfd__ = DATADIR
        self.__cholesky__ = os.path.join(self.__cfd__,f"cholesky_{method}{'' if not diag else '_diag'}")
        if mpi.rank == 0:
            os.makedirs(self.__cholesky__,exist_ok=True)
            os.makedirs(self.basedir,exist_ok=True)
            for f in [23,100,143,353]:
                os.makedirs(os.path.join(self.basedir,f"{f}"),exist_ok=True)
        mpi.barrier()
        # mask
        self.__tmask__ = os.path.join(self.__cfd__,'tmaskfrom0p70.dat')
        self.__qumask__ = os.path.join(self.__cfd__,'mask_pol_nside16.dat')
        self.fsky = np.average(self.polmask('ring'))
        self.method = method


        #NCM
        ncms_sroll = {
           100 : os.path.join(self.__cfd__,'noise_SROLL20_100psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
           143 : os.path.join(self.__cfd__,'noise_SROLL20_143psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
           353 : os.path.join(self.__cfd__,'noise_SROLL20_353psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
        }
        ncms_ffp8 = {
            100 : os.path.join(self.__cfd__,'dx11_ncm_100_combined_smoothed_nside0016.dat'),
            143 : os.path.join(self.__cfd__,'dx11_ncm_143_combined_smoothed_nside0016.dat'),
            353 : os.path.join(self.__cfd__,'dx11_ncm_353_combined_smoothed_nside0016.dat'),
        }


        if method == 'sroll':
            self.ncms = ncms_sroll
        elif method == 'ffp8':
            self.ncms = ncms_ffp8
 
        self.ncms[23] = os.path.join(self.__cfd__,'wmap_K_coswin_ns16_9yr_v5_covmat.bin')
        self.cholesky_dict = {}

    
    
    def inpvec(self,fname, double_prec=True):
        if double_prec:
            dat=np.fromfile(fname,dtype=np.float64, sep='', offset=4)
            dat=dat.astype('float32')         
        else:
            dat = np.fromfile(fname,dtype=self.dtype).astype(self.dtype)
        return dat
    
    def polmask(self,order='ring'):
        mask = self.inpvec(self.__qumask__, double_prec=False)
        if order=='ring':
            pass
        elif order=='nested':
            mask = hp.reorder(mask,r2n=True)
        else:
            raise ValueError('order must be ring or nested')
        return mask
    
   
    def inpcovmat(self,fname, double_prec=True):
        if double_prec==True:
            dat=np.fromfile(fname,dtype=np.float64, sep='', offset=4)
            dat=dat.astype('float32') 
        elif double_prec==False:
            #dat=np.fromfile(fname,dtype=np.float32, sep='')
            dat=np.fromfile(fname,dtype=self.dtype).astype(self.dtype)
        n=int(np.sqrt(dat.shape))
        return np.reshape(dat,(n,n)).copy()
    
    @staticmethod
    def corr(mat):
        mat2  = mat.copy()
        sdiag = 1./np.sqrt(np.diag(mat))
        mat2[:,:]=mat[:,:]*sdiag[:]*sdiag[:]
        return mat2

    @staticmethod
    def unmask(x, mask):
        y=np.zeros(len(mask))
        y[mask==1] = x
        return y

    @staticmethod
    def unmask_matrix(matrix, mask):
        unmasked_mat = np.zeros((len(mask),len(mask)))
        idx       = np.where(mask>0.)[0]
        x_idx, y_idx = np.meshgrid(idx, idx)
        unmasked_mat[x_idx, y_idx]  = matrix[:,:]
        return unmasked_mat

    @staticmethod
    def mask_matrix(matrix, mask):
        matrixold = matrix.copy()
        ncovold   = matrix.shape[0]
        ncov      = np.sum(mask>0)
        matrix    = np.zeros((ncov,ncov))
        idx       = np.where(mask>0)[0]
        x_idx, y_idx = np.meshgrid(idx, idx)
        matrix[:,:]  = matrixold[x_idx, y_idx]
        return matrix
    
    def get_prefac(self, which):
        fac_reg = 1.e-2 #UNCLEAR
        fac_degrade_HFI = ((2048)/(128))**2
        fac_ncm_HFI     = ((4.*np.pi)/(12.*2048.**2))**2*1.e12
        fac_degrade_LFI = ((1024)/(128))**2
        fac_ncm_LFI     = ((4.*np.pi)/(12.*1024.**2))**2*1.e12
        fac_K = 1.e12 #UNCLEAR
        if which == 'HFI':
            return fac_degrade_HFI*fac_ncm_HFI
        elif which == 'LFI':
            return fac_degrade_LFI*fac_ncm_LFI
    
    def __offdiag_to_zeros__(self,mat: np.ndarray) -> np.ndarray:
        return np.diag(np.diag(mat))
    
    def get_ncm_generic(self,freq,unit='uK'):
        ncm = np.fromfile(self.ncms[freq])
        ncm = ncm.reshape(3*self.npix,3*self.npix)
        ncmqu = ncm[self.npix:,self.npix:]
        lambdaI = np.eye(2*self.npix) * 4e-16
        ncmqu = ncmqu + lambdaI
        if unit == 'uK':
            ncmqu *= 1e12
        del (ncm,lambdaI)
        if self.diag:
            return self.__offdiag_to_zeros__(ncmqu)
        else:
            return ncmqu
        
    def get_ncm_sroll(self,freq,unit='uK'):
        if freq not in self.ncms.keys():
            raise ValueError('Freq must be one of {}'.format(self.ncms.keys()))
        
        if freq==23:
            return self.get_ncm_generic(freq,unit=unit)
        
        ncm_dir = self.ncms[freq]
        fac = 1
        if unit=='uK':
            pass
        elif unit=='K':
            fac = 1e-12
        else:
            raise ValueError('unit must be uK or K')
        if self.diag:
            return self.__offdiag_to_zeros__(self.inpcovmat(ncm_dir,double_prec=False))*fac
        else:
            return np.float64(self.inpcovmat(ncm_dir,double_prec=False)) *fac
        
    def get_ncm(self,*args,**kwargs):
        if self.method == 'sroll':
            return self.get_ncm_sroll(*args,**kwargs)
        elif (self.method == 'ffp8'):
            return self.get_ncm_generic(*args,**kwargs)
        
    def get_full_ncm_generic(self,freq,unit='uK',pad_temp=False,reshape=False,order='ring'):
        ncm_pol = self.get_ncm_generic(freq,unit=unit)
        if order=='ring':
            pass
        elif order=='nested':
            re_idx = hp.nest2ring(self.nside,np.arange(hp.nside2npix(self.nside)))
            re_idx_full = np.concatenate([re_idx,re_idx+hp.nside2npix(self.nside)])
            ncm_pol = ncm_pol[:,re_idx_full][re_idx_full,:]
        else:
            raise ValueError('order must be ring or nested')
            
        del ncm
        if pad_temp:
            NCM = np.zeros((3*self.npix,3*self.npix))
            NCM[self.npix:,self.npix:] = ncm_pol
        else:
            NCM = ncm_pol
        
        if reshape:
            return NCM.reshape(-1)
        else:
            return NCM
    
    def get_full_ncm_sroll(self,freq,unit='uK',pad_temp=False,reshape=False,order='ring'):
        ncm = self.get_ncm_sroll(freq,unit=unit)
        
        if (freq == 23) and self.diag:
            ncm_pol = ncm
        else:
            ncm_pol =  self.unmask_matrix(ncm,np.concatenate([self.polmask('ring'),self.polmask('ring')]))
        if order=='ring':
            pass
        elif order=='nested':
            re_idx = hp.nest2ring(self.nside,np.arange(hp.nside2npix(self.nside)))
            re_idx_full = np.concatenate([re_idx,re_idx+hp.nside2npix(self.nside)])
            ncm_pol = ncm_pol[:,re_idx_full][re_idx_full,:]
        else:
            raise ValueError('order must be ring or nested')
            
        del ncm
        if pad_temp:
            NCM = np.zeros((3*self.npix,3*self.npix))
            NCM[self.npix:,self.npix:] = ncm_pol
        else:
            NCM = ncm_pol
        
        if reshape:
            return NCM.reshape(-1)
        else:
            return NCM
        
    def get_full_ncm(self,*args,**kwargs):
        if self.method == 'sroll':
            return self.get_full_ncm_sroll(*args,**kwargs)
        elif (self.method == 'ffp8'):
            return self.get_full_ncm_generic(*args,**kwargs)
        
    def noisemap_sroll(self,freq,idx=None,order='nested',unit='K'):
        fname = os.path.join(self.__cholesky__,f'cholesky{freq}_{order}_{unit}.pkl')
        if freq == 23:
            return self.noisemap_generic(freq,idx=idx,order=order,unit=unit)
        
        if os.path.exists(fname):
            if fname not in self.cholesky_dict.keys():
                self.cholesky_dict[fname] = pkl.load(open(fname,'rb'))
            cho = self.cholesky_dict[fname]
        else:
            ncm = self.get_ncm(freq,unit=unit)
            cho = np.linalg.cholesky(ncm)
            pkl.dump(cho,open(fname,'wb'))
        noisemap = cho2map(cho)
        polmask=self.inpvec(self.__qumask__, double_prec=False)
        pl = int(sum(polmask))

        QU =  np.array([self.unmask(noisemap[:pl],polmask),self.unmask(noisemap[pl:],polmask)])
        QU = QU * self.polmask('ring')
        if order=='ring':
            return QU
        elif order=='nested':
            QU = hp.reorder(QU,r2n=True)
        else:
            raise ValueError('order must be ring or nested')
        return QU
    
    def noisemap_generic(self,freq,idx=None,order='nested',unit='K'):
        fname = os.path.join(self.__cholesky__,f'cholesky{freq}_{order}_{unit}.pkl')
        if os.path.exists(fname):
            if fname not in self.cholesky_dict.keys():
                self.cholesky_dict[fname] = pkl.load(open(fname,'rb'))
            cho = self.cholesky_dict[fname]
        else:
            ncm = self.get_ncm_generic(freq,unit=unit)
            cho = np.linalg.cholesky(ncm)
            pkl.dump(cho,open(fname,'wb'))
        noisemap = cho2map(cho)
        pix = cho.shape[0]
        QU =  np.array([noisemap[:pix//2]*self.polmask('nested'),
                        noisemap[pix//2:]*self.polmask('nested')])
        if order=='ring':
            QU = hp.reorder(QU,n2r=True)
        return QU
    

        
    def noisemap(self,freq,idx=None,order='nested',unit='K'):
        seed = 91094 + (0 if idx is None else idx) + int(freq)
        fname = os.path.join(self.basedir,f"{freq}",f"noisemap_{self.method}_{order}_{unit}_{idx:06d}.pkl")
        if os.path.exists(fname):
            QU = pkl.load(open(fname,'rb'))
            return QU
        else:
            np.random.seed(seed)
            if self.method == 'sroll':
                QU = self.noisemap_sroll(freq,idx,order,unit)
            elif (self.method == 'ffp8'):
                QU = self.noisemap_generic(freq,idx,order,unit)
            
            pkl.dump(QU,open(fname,'wb'))
            return QU
            

    def Emode(self,freq,idx,unit='uK'):
        Q,U = self.noisemap(freq,idx,order='ring',unit=unit)
        return hp.map2alm_spin([Q,U],2,lmax=3*self.nside-1)[0]
        

class NoiseModelDiag:

    def __init__(self,nside=16):
        warn(f'Only for testing purpose, will be deprecated soon', DeprecationWarning, stacklevel=2)
        self.nside = nside
        self.npix = hp.nside2npix(self.nside)
    
    def ncm(self,unit='uK'):
        ncm = np.eye(3*self.npix)
        if unit == 'uK':
            fac = 1.
        elif unit == 'K':
            fac = 1e-12
        else:
            raise ValueError('unit not recognized')
        
  
        sigma = 0.01
        return ncm * sigma * fac
    
    def noisemap(self,unit='uK'):
        sigma_pix = 0.01
        return np.array([np.random.normal(0,sigma_pix,self.npix),np.random.normal(0,sigma_pix,self.npix)])
    
    def Emode(self,idx=None,unit='uK'):
        Q,U = self.noisemaps(unit)
        return hp.map2alm_spin([Q,U],2,lmax=3*self.nside-1)[0]

# def swap_diagonal(matrixa, new_order_indices):
#     matrix = matrixa.copy()
#     n = len(matrix)
#     original_diagonal = np.diag(matrix)
#     new_diagonal = original_diagonal[new_order_indices]
#     np.fill_diagonal(matrix, new_diagonal)
#     for i, new_index in enumerate(new_order_indices):
#         if i != new_index:
#             matrix[i, new_index], matrix[new_index, i] = matrix[new_index, new_index], matrix[new_index, i]
#     return matrix