# Copyright (C) 2023 Roger de Belsunce
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

#define type
dtype = np.float32

nside = 16
npix = 12*nside**2
# directories

basedir = '/Users/rdb/Desktop/research/cmb_lowl_tau/'

tmaskname      = basedir + 'tmaskfrom0p70.dat'
qumaskname     = basedir + 'mask_pol_nside16.dat'

# NR foreground code
split_LFI=''
split_HFI='full'
ncm_LFI     = '/Users/rdb/Desktop/research/cmb_lowl_tau/'
ncm_30_dir = ncm_LFI+'noise_FFP10_30full{}_EB_lmax4_pixwin_200sims_smoothmean_AC.dat'.format(split_LFI)
ncm_44_dir = ncm_LFI+'noise_FFP10_44full{}_EB_lmax4_pixwin_200sims_smoothmean_AC.dat'.format(split_LFI)
ncm_70_dir = ncm_LFI+'noise_FFP10_70full{}_EB_lmax4_pixwin_200sims_smoothmean_AC.dat'.format(split_LFI)
ncm_HFI     = '/Users/rdb/Desktop/research/cmb_lowl_tau/'
ncm_100_dir = ncm_HFI+'noise_SROLL20_100psb_{}_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'.format(split_HFI)
ncm_143_dir = ncm_HFI+'noise_SROLL20_143psb_{}_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'.format(split_HFI)
ncm_217_dir = ncm_HFI+'noise_SROLL20_217psb_{}_EB_lmax4_pixwin_200sims_smoothmean_AC_suboffset_new.dat'.format(split_HFI)
ncm_353_dir = ncm_HFI+'noise_SROLL20_353psb_{}_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat'.format(split_HFI)


def inpvec(fname, double_prec=True):
    if double_prec:
        dat=np.fromfile(fname,dtype=np.float64, sep='', offset=4)
        dat=dat.astype('float32')         
    else:
        dat = np.fromfile(fname,dtype=dtype).astype(dtype)
    return dat

def inpcovmat(fname, double_prec=True):
    if double_prec==True:
        dat=np.fromfile(fname,dtype=np.float64, sep='', offset=4)
        dat=dat.astype('float32') 
    elif double_prec==False:
        #dat=np.fromfile(fname,dtype=np.float32, sep='')
        dat=np.fromfile(fname,dtype=dtype).astype(dtype)
    n=int(np.sqrt(dat.shape))
    return np.reshape(dat,(n,n)).copy()

def corr(mat):
    mat2  = mat.copy()
    sdiag = 1./np.sqrt(np.diag(mat))
    mat2[:,:]=mat[:,:]*sdiag[:]*sdiag[:]
    return mat2

def unmask(x, mask):
    y=np.zeros(len(mask))
    y[mask==1] = x
    return y

def unmask_matrix(matrix, mask):
    unmasked_mat = np.zeros((len(mask),len(mask)))
    idx       = np.where(mask>0.)[0]
    x_idx, y_idx = np.meshgrid(idx, idx)
    unmasked_mat[x_idx, y_idx]  = matrix[:,:]
    return unmasked_mat

def mask_matrix(matrix, mask):
    matrixold = matrix.copy()
    ncovold   = matrix.shape[0]
    ncov      = np.sum(mask>0)
    matrix    = np.zeros((ncov,ncov))
    idx       = np.where(mask>0)[0]
    x_idx, y_idx = np.meshgrid(idx, idx)
    matrix[:,:]  = matrixold[x_idx, y_idx]
    return matrix

############################################################
# load NCMs
############################################################
# regularising noise
fac_reg = 1.e-2


fac_degrade_HFI = ((2048)/(128))**2
fac_ncm_HFI     = ((4.*np.pi)/(12.*2048.**2))**2*1.e12
fac_degrade_LFI = ((1024)/(128))**2
fac_ncm_LFI     = ((4.*np.pi)/(12.*1024.**2))**2*1.e12
fac_K           = 1.e12
print('NCM LFI pre factor: ', fac_degrade_LFI*fac_ncm_LFI, flush=True)
print('NCM HFI pre factor: ', fac_degrade_HFI*fac_ncm_HFI, flush=True)
print('---CAREFUL PREFACTOR FOR NCM_30, NCM_44, NCM_70, NCM_217, NCM_353 ---', flush=True)


#read NCMs
ncm_30  = np.float64(inpcovmat(ncm_30_dir,double_prec=False))   #fac correct
ncm_44  = np.float64(inpcovmat(ncm_44_dir,double_prec=False))   #fac correct
ncm_70  = np.float64(inpcovmat(ncm_70_dir,double_prec=False))   #fac correct
ncm_100 = np.float64(inpcovmat(ncm_100_dir,double_prec=False))  #fac correct
ncm_143 = np.float64(inpcovmat(ncm_143_dir,double_prec=False))  #fac correct
ncm_217 = np.float64(inpcovmat(ncm_217_dir,double_prec=False))  #fac correct
ncm_353 = np.float64(inpcovmat(ncm_353_dir,double_prec=False))  #fac correct

n_11=ncm_100
n_22=ncm_143

print(n_11.shape)
print(n_22.shape)

polmask=inpvec(qumaskname, double_prec=False)
qumask=np.concatenate((polmask,polmask))
pl=int(sum(polmask))

print(pl)

unmask_n11 = unmask_matrix(n_11, qumask)
unmask_n22 = unmask_matrix(n_22, qumask)

hp.mollview(np.diag(unmask_n11)[:npix]);plt.show()
hp.mollview(np.diag(unmask_n22)[:npix]);plt.show()

# generate Gaussian realizations of NCMs
#perform cholesky decomposition of matrices
l_11 = np.linalg.cholesky(n_11)

pix = l_11.shape[0]
totpix = 12*nside*nside
mu = 0.
sig = 1.
noise_mean_11 = np.zeros(pix)
tmp_11 = np.zeros((3,totpix))
map_tot_11= np.zeros((3,totpix))
map_ncm_11= np.zeros((3,totpix))
mean_tot_11= np.zeros((3,totpix))

#noise_tau_r{0..1.0}_0.060_16R_pixwin_{1..5000}.fits
tau_integer = 1
tau_value_min = 0.4
delta_tau = 0.1
stepsize_tau =0.05

sim_max = 5
counter_seed=3215000 + int(tau_integer*sim_max)


for tau_value in np.arange(tau_value_min, tau_value_min+delta_tau+stepsize_tau,stepsize_tau): #counter_seed=3215000
    print('tau={:1.3f}'.format(tau_value), flush=True)
    counter=0
    while counter < sim_max:
        fname_11     = 'Noise_simulation_tau_{:1.3f}_16R_pixwin_{}.fits'.format(tau_value,counter) # save noise sims
        fname_tot_11 = 'Signal_noise_sim_NCM_tau_{:1.3f}_{}.fits'.format(tau_value,counter)  # signal + noise

        # read CMB realisation 
        # ANTO - COMMENT IN THE FOLLOWING LINES WITH YOUR REALIZATIONS
        sname_11     = 'CMB_signal_sim_tau_tau_{:1.3f}_16R_pixwin_{}.fits'.format(tau_value, counter) # generate CAMB CMB maps
        #map   = hp.read_map(sname_11, field=(0,1,2))
        # DANGEROUS ANTO
        map = np.zeros((3,npix))
        # DANGEROUS ANTO    



        np.random.seed(counter_seed)
        noise_11 = np.random.normal(mu, sig,pix)
        noise_map_11 = np.dot(l_11, noise_11)
        
        map_ncm_11[1,:] = unmask(map[1,polmask==1],polmask)
        map_ncm_11[2,:] = unmask(map[2,polmask==1],polmask)
        hp.write_map(fname_11, map_ncm_11,  overwrite=True)
        map_tot_11[1,:] = map_ncm_11[1,:]+unmask(noise_map_11[:pl],polmask)
        map_tot_11[2,:] = map_ncm_11[2,:]+unmask(noise_map_11[pl:],polmask)
        hp.write_map(fname_tot_11, map_tot_11,  overwrite=True)
        counter+=1
        counter_seed+=1
    print(fname_11, flush=True)
    print(fname_tot_11, flush=True)
    print(sname_11, flush=True)
    print(counter, flush=True)
    print(counter_seed, flush=True)
    print('done: no. sims={:5d}, r={:1.2f}'.format(counter, tau_value), flush=True)
    print('------------------------------------------', flush=True)
print('done', flush=True)

