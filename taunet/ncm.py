import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from taunet import DATADIR, SROLL2, FFP8
import pickle as pkl
from numba import njit,f8
from warnings import warn


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
    #Roger et.al

    def __init__(self,diag=False,method='roger'):
        assert method in ['roger','ffp8','sroll2'], 'method must be one of roger, ffp8, sroll2'
        self.nside = 16
        self.npix = 12*self.nside**2
        self.dtype = np.float32
        self.diag = diag

        # directory
        self.__cfd__ = DATADIR
        self.__cholesky__ = os.path.join(self.__cfd__,f"cholesky_{method}{'' if not diag else '_diag'}")
        os.makedirs(self.__cholesky__,exist_ok=True)
        # mask
        self.__tmask__ = os.path.join(self.__cfd__,'tmaskfrom0p70.dat')
        self.__qumask__ = os.path.join(self.__cfd__,'mask_pol_nside16.dat')
        self.fsky = np.average(self.polmask('ring'))
        self.method = method


        #NCM
        ncms_roger = {
           100 : os.path.join(self.__cfd__,'noise_SROLL20_100psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
           143 : os.path.join(self.__cfd__,'noise_SROLL20_143psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
           353 : os.path.join(self.__cfd__,'noise_SROLL20_353psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
        }
        ncms_ffp8 = {
            100 : os.path.join(FFP8,'dx11_ncm_100_combined_smoothed_nside0016.dat'),
            143 : os.path.join(FFP8,'dx11_ncm_143_combined_smoothed_nside0016.dat'),
            353 : os.path.join(FFP8,'dx11_ncm_353_combined_smoothed_nside0016.dat'),
        }
        ncms_sroll2 = {
            100: os.path.join(SROLL2,'map_sroll2_100psb_coswin_ns16_full.fits'),
            143: os.path.join(SROLL2,'map_sroll2_143psb_coswin_ns16_full.fits'),
            353: os.path.join(SROLL2,'map_sroll2_353psb_coswin_ns16_full.fits'),
        }


        if method == 'roger':
            self.ncms = ncms_roger
        elif method == 'ffp8':
            self.ncms = ncms_ffp8
        elif method == 'sroll2':
            self.ncms = ncms_sroll2
 
        self.ncms[23] = os.path.join(self.__cfd__,'wmap_K_coswin_ns16_9yr_v5_covmat.bin')

    
    
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
    
    def __get_ncm23_roger__(self,unit='K'):
        if unit!='K':
            raise ValueError('unit must be K')
        ncm = np.fromfile(self.ncms[23])
        ncm = ncm.reshape(3*self.npix,3*self.npix)
        ncmqu = ncm[self.npix:,self.npix:]
        lambdaI = np.eye(2*self.npix) * 4e-16
        ncmqu = ncmqu + lambdaI
        del (ncm,lambdaI)
        return ncmqu
    
    def get_ncm_roger(self,freq,unit='uK'):
        if freq not in self.ncms.keys():
            raise ValueError('Freq must be one of {}'.format(self.ncms.keys()))
        
        if freq==23:
            if self.diag:
                return self.__offdiag_to_zeros__(self.__get_ncm23_roger__(unit=unit))
            else:
                return self.__get_ncm23_roger__(unit=unit)
        
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
    
    def get_full_ncm_roger(self,freq,unit='uK',pad_temp=False,reshape=False,order='ring'):
        ncm = self.get_ncm_roger(freq,unit=unit)
        
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
    
    def __noisemap23_roger__(self,idx=None,order='nested',unit='K'):
        fname = os.path.join(self.__cholesky__,f'cholesky23_{order}_{unit}.pkl')
        if self.diag:
            ncm = self.__offdiag_to_zeros__(self.__get_ncm23_roger__(unit='K'))
        else:
            ncm = self.__get_ncm23_roger__(unit='K')
        if os.path.exists(fname):
            cho = pkl.load(open(fname,'rb'))
        else:
            cho = np.linalg.cholesky(ncm)
            pkl.dump(cho,open(fname,'wb'))
        
        seed = 152 + (0 if idx is None else idx) + 23
        np.random.seed(seed)
        noisemap = cho2map(cho)
        pix = cho.shape[0]
        del cho
        #ncm_cho = np.linalg.cholesky(ncm)
        #pix = ncm_cho.shape[0]
        #noisem = np.random.normal(0,1,pix)
        #noisemap = np.dot(ncm_cho, noisem)
        QU =  np.array([noisemap[:pix//2]*self.polmask('nested'),
                        noisemap[pix//2:]*self.polmask('nested')])
        if order=='ring':
            QU[0] = hp.reorder(QU[0],n2r=True)
            QU[1] = hp.reorder(QU[1],n2r=True)
        
        if unit=='uK':
            QU *=1e6
        return QU
    
    def noisemap_roger(self,freq,idx=None,order='ring',unit='uK'):

    
        fname = os.path.join(self.__cholesky__,f'cholesky{freq}_{order}_{unit}.pkl')
        if freq == 23:
            return self.__noisemap23_roger__(idx=idx,order=order,unit=unit)
        
        ncm = self.get_ncm(freq,unit=unit)
        if os.path.exists(fname):
            cho = pkl.load(open(fname,'rb'))
        else:
            cho = np.linalg.cholesky(ncm)
            pkl.dump(cho,open(fname,'wb'))
        seed = 152 + (0 if idx is None else idx) + freq
        np.random.seed(seed)
        noisemap = cho2map(cho)
        #noisem = np.random.normal(0,1,pix)
        #noisemap = np.dot(ncm_cho, noisem)

        polmask=self.inpvec(self.__qumask__, double_prec=False)
        pl = int(sum(polmask))

        QU =  np.array([self.unmask(noisemap[:pl],polmask),self.unmask(noisemap[pl:],polmask)])
        
        QU = QU * self.polmask('ring')

        if order=='ring':
            pass
        elif order=='nested':
            QU = hp.reorder(QU,r2n=True)
        else:
            raise ValueError('order must be ring or nested')
        return QU
    
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
    def noisemap_generic(self,freq,idx=None,order='ring',unit='uK'):
        ncm = self.get_ncm_generic(freq,unit=unit)
        fname = os.path.join(self.__cholesky__,f'cholesky{freq}_{order}_{unit}.pkl')
        if os.path.exists(fname):
            cho = pkl.load(open(fname,'rb'))
        else:
            cho = np.linalg.cholesky(ncm)
            pkl.dump(cho,open(fname,'wb'))
        seed = 152 + (0 if idx is None else idx) + freq
        np.random.seed(seed)
        noisemap = cho2map(cho)
        pix = cho.shape[0]
        QU =  np.array([noisemap[:pix//2]*self.polmask('nested'),
                        noisemap[pix//2:]*self.polmask('nested')])
        if order=='ring':
            QU[0] = hp.reorder(QU[0],n2r=True)
            QU[1] = hp.reorder(QU[1],n2r=True)
        
        # if unit=='uK':
        #     QU *=1e6
        return QU
    
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


    def get_ncm(self,*args,**kwargs):
        if self.method == 'roger':
            return self.get_ncm_roger(*args,**kwargs)
        elif (self.method == 'ffp8') or (self.method == 'sroll2'):
            return self.get_ncm_generic(*args,**kwargs)
    
    def get_full_ncm(self,*args,**kwargs):
        if self.method == 'roger':
            return self.get_full_ncm_roger(*args,**kwargs)
        elif (self.method == 'ffp8') or (self.method == 'sroll2'):
            return self.get_full_ncm_generic(*args,**kwargs)
        
    def noisemap(self,*args,**kwargs):
        if self.method == 'roger':
            return self.noisemap_roger(*args,**kwargs)
        elif (self.method == 'ffp8') or (self.method == 'sroll2'):
            return self.noisemap_generic(*args,**kwargs)
        

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