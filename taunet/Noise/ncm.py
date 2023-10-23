import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os



class NoiseModel:
    #Roger et.al

    def __init__(self):
        self.nside = 16
        self.npix = 12*self.nside**2
        self.dtype = np.float32

        # directory
        self.__cfd__ = os.path.dirname(os.path.realpath(__file__))
        # mask
        self.__tmask__ = os.path.join(self.__cfd__,'tmaskfrom0p70.dat')
        self.__qumask__ = os.path.join(self.__cfd__,'mask_pol_nside16.dat')


        #NCM
        self.ncms = {
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
    
    def get_ncm(self,freq):
        if freq not in self.ncms.keys():
            raise ValueError('Freq must be one of {}'.format(self.ncms.keys()))
        ncm_dir = self.ncms[freq]
        return np.float64(self.inpcovmat(ncm_dir,double_prec=False))
    
    def get_full_ncm(self,freq,pad_temp=False,reshape=False,save=None,order='ring'):
        ncm = self.get_ncm(freq)
        
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

    def noisemap(self,freq,order='ring'):
        polmask=self.inpvec(self.__qumask__, double_prec=False)
        pl = int(sum(polmask))

        ncm = self.get_ncm(freq)
        ncm_cho = np.linalg.cholesky(ncm)
        pix = ncm.shape[0]
        noisem = np.random.normal(0,1,pix)
        noisemap = np.dot(ncm_cho, noisem)
        QU =  self.unmask(noisemap[:pl],polmask),self.unmask(noisemap[pl:],polmask)
        if order=='ring':
            pass
        elif order=='nested':
            QU = hp.reorder(QU,r2n=True)
        else:
            raise ValueError('order must be ring or nested')
        return QU
    
    def Emode(self,freq):
        Q,U = self.noisemap(freq,'ring')
        return hp.map2alm_spin([Q,U],2,lmax=3*self.nside-1)[0]
        