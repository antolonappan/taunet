import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
from taunet import DATADIR
import pickle as pkl

cosbeam = hp.read_cl(os.path.join(DATADIR,'beam_coswin_ns16.fits'))[0]

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

class NoiseModel:
    #Roger et.al

    def __init__(self):
        self.nside = 16
        self.npix = 12*self.nside**2
        self.dtype = np.float32

        # directory
        self.__cfd__ = DATADIR
        self.__cholesky__ = os.path.join(self.__cfd__,'cholesky')
        os.makedirs(self.__cholesky__,exist_ok=True)
        # mask
        self.__tmask__ = os.path.join(self.__cfd__,'tmaskfrom0p70.dat')
        self.__qumask__ = os.path.join(self.__cfd__,'mask_pol_nside16.dat')
        self.fsky = np.average(self.polmask('ring'))


        #NCM
        self.ncms = {
            23 : os.path.join(self.__cfd__,'wmap_K_coswin_ns16_9yr_v5_covmat.bin'),
            30 : os.path.join(self.__cfd__,'noise_FFP10_30full_EB_lmax4_pixwin_200sims_smoothmean_AC.dat'),
            40 : os.path.join(self.__cfd__,'noise_FFP10_44full_EB_lmax4_pixwin_200sims_smoothmean_AC.dat'),
            70 : os.path.join(self.__cfd__,'noise_FFP10_70full_EB_lmax4_pixwin_200sims_smoothmean_AC.dat'),
           100 : os.path.join(self.__cfd__,'noise_SROLL20_100psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
           143 : os.path.join(self.__cfd__,'noise_SROLL20_143psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
           217 : os.path.join(self.__cfd__,'noise_SROLL20_217psb_full_EB_lmax4_pixwin_200sims_smoothmean_AC_suboffset_new.dat'),
           353 : os.path.join(self.__cfd__,'noise_SROLL20_353psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'),
        }
    
    
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
    
    def __get_ncm23__(self,unit='K'):
        if unit!='K':
            raise ValueError('unit must be K')
        ncm = np.fromfile(self.ncms[23])
        ncm = ncm.reshape(3*self.npix,3*self.npix)
        ncmqu = ncm[self.npix:,self.npix:]
        lambdaI = np.eye(2*self.npix) * 4e-16
        ncmqu = ncmqu + lambdaI
        del (ncm,lambdaI)
        return ncmqu
    
    def get_ncm(self,freq,unit='uK'):
        if freq not in self.ncms.keys():
            raise ValueError('Freq must be one of {}'.format(self.ncms.keys()))
        
        if freq==23:
            return self.__get_ncm23__(unit=unit)
        
        ncm_dir = self.ncms[freq]
        fac = 1
        if unit=='uK':
            pass
        elif unit=='K':
            fac = 1e-12
        else:
            raise ValueError('unit must be uK or K')
        return np.float64(self.inpcovmat(ncm_dir,double_prec=False)) *fac
    
    def get_full_ncm(self,freq,unit='uK',pad_temp=False,reshape=False,order='ring'):
        ncm = self.get_ncm(freq,unit=unit)
        
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
    
    def __noisemap23__(self,order='nested',unit='K'):
        fname = os.path.join(self.__cholesky__,'cholesky_23_nested.pkl')
        if os.path.exists(fname):
            ncm_cho = pkl.load(open(fname,'rb'))
        else:
            ncm = self.__get_ncm23__(unit=unit)
            ncm_cho = np.linalg.cholesky(ncm)
            pkl.dump(ncm_cho,open(fname,'wb'))
        pix = ncm_cho.shape[0]
        noisem = np.random.normal(0,1,pix)
        noisemap = np.dot(ncm_cho, noisem)
        QU =  np.array([noisemap[:pix//2]*self.polmask('nested'),
                        noisemap[pix//2:]*self.polmask('nested')])
        if order=='ring':
            QU[0] = hp.reorder(QU[0],n2r=True)
            QU[1] = hp.reorder(QU[1],n2r=True)
        
        if unit=='uK':
            QU *=1e6
        return QU
    
    def noisemap(self,freq,order='ring',unit='uK',deconvolve=False):
        if freq == 23:
            return self.__noisemap23__(order=order,unit=unit)
        
        fname = os.path.join(self.__cholesky__,'cholesky_{}_{}.pkl'.format(freq,order))
        if os.path.exists(fname):
            ncm_cho = pkl.load(open(fname,'rb'))
        else:
            ncm = self.get_ncm(freq,unit=unit)
            ncm_cho = np.linalg.cholesky(ncm)
            pkl.dump(ncm_cho,open(fname,'wb'))
        

        polmask=self.inpvec(self.__qumask__, double_prec=False)
        pl = int(sum(polmask))

        pix = ncm_cho.shape[0]
        noisem = np.random.normal(0,1,pix)
        noisemap = np.dot(ncm_cho, noisem)
        QU =  np.array([self.unmask(noisemap[:pl],polmask),self.unmask(noisemap[pl:],polmask)])

        
        if deconvolve:
            alm = hp.map2alm([QU[0]*0,QU[0],QU[1]],lmax=3*self.nside-1)
            hp.almxfl(alm[1],cli(cosbeam),inplace=True)
            hp.almxfl(alm[2],cli(cosbeam),inplace=True)
            QU = hp.alm2map(alm,self.nside,verbose=False)[1:]
        
        QU = QU * self.polmask('ring')

        if order=='ring':
            pass
        elif order=='nested':
            QU = hp.reorder(QU,r2n=True)
        else:
            raise ValueError('order must be ring or nested')
        return QU
    
    def Emode(self,freq,unit='uK',deconvolve=False):
        Q,U = self.noisemap(freq,'ring',unit=unit,deconvolve=deconvolve)
        return hp.map2alm_spin([Q,U],2,lmax=3*self.nside-1)[0]
        
class NoiseModelGaussian:

    def __init__(self,nside,nlevp):
        self.libdir = os.path.join(DATADIR,'choleskyGaussian')
        os.makedirs(self.libdir,exist_ok=True)
        self.nside = nside
        self.nlevp = nlevp
        self.nlevt = nlevp * np.sqrt(2.)
        self.npix = hp.nside2npix(self.nside)
    
    def ncm(self,unit='uK'):
        ncm = np.eye(3*self.npix)
        if unit == 'uK':
            fac = 1.
        elif unit == 'K':
            fac = 1e-12
        else:
            raise ValueError('unit not recognized')
        
        pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.nside)) * (180. * 60. / np.pi) ** 2
        sigma_t = np.sqrt(self.nlevt ** 2 / pix_amin2)
        sigma_p = np.sqrt(self.nlevp ** 2 / pix_amin2)
        ncm[:self.npix,:self.npix] *= (sigma_t** 2)
        ncm[self.npix:,self.npix:] *= (sigma_p** 2)
        return ncm * fac
    
    def noisemaps(self,unit='uK'):
        fname = os.path.join(self.libdir,f"cholesky_{str(self.nlevt).replace('.','p')}_{str(self.nlevp).replace('.','p')}_{unit}.pkl")
        if os.path.isfile(fname):
            cho = pkl.load(open(fname,'rb'))
        else:
            ncm = self.ncm(unit)
            cho = np.linalg.cholesky(ncm)
            pkl.dump(cho,open(fname,'wb'))
        noisemaps = np.dot(cho,np.random.normal(0.,1.,cho.shape[0]))
        return noisemaps[:self.npix],noisemaps[self.npix:2*self.npix],noisemaps[2*self.npix:]
    
    def Emode(self,unit='uK'):
        T,Q,U = self.noisemaps(unit)
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