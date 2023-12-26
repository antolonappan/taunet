import os
import hashlib
import requests
from tqdm import tqdm

__path__ = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(__path__,'..','Data')
os.makedirs(DATADIR, exist_ok=True)


__LINK__ = 'https://figshare.com/ndownloader/files/'

__FILES__ =  {'noise_FFP10_30full_EB_lmax4_pixwin_200sims_smoothmean_AC.dat':[43491375,'f16b81e701fc781c57398b28fa2f39ad'],
              'noise_SROLL20_100psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat':[43491423,'1a994dbce1e986ca2986ebccf40af873'],
              'noise_SROLL20_143psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat':[43491426,'b344cd02dcc83d8cd89bccc65c3c2229'],
              'noise_SROLL20_353psb_full_EB_lmax4_pixwin_400sims_smoothmean_AC_suboffset_new.dat':[43491432,'d2ee127baa8b041c54e0e557aaa0ef0e'],
              'tmaskfrom0p70.dat':[43491480,'6523f7c30cfb0e95fa0d9a7e210b2964'],
              'wmap_K_coswin_ns16_9yr_v5_covmat.bin':[43491483,'7f271b328e5edfc63f1dc2d822200bd7'],
              'beam_coswin_ns16.fits':[43491582,'c5543348abad26a88c1a9a574212934e'],
              'mask_pol_nside16.dat':[43491585,'28db044800168128d355c8275675c72d'],
              'beam_coswin_ns16.fits':[43715016,'c5543348abad26a88c1a9a574212934e'],
            }

def md5(fname):
    """Compute md5 checksum of the specified file."""
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, filename):
    """Download a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

# Checking and downloading files
for file, (file_id, md5_hash) in __FILES__.items():
    file_path = os.path.join(DATADIR, file)
    
    if not os.path.exists(file_path):
        print(f"File {file} not found in Data. Downloading from cloud.")
        download_url = f"{__LINK__}{file_id}"
        download_file(download_url, file_path)
        print("Download complete. Checking MD5 checksum...")
        
        if md5(file_path) == md5_hash:
            print("Checksum passed.")
        else:
            print("Checksum failed. File might be corrupted.")