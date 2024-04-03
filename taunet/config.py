import os
import warnings


if 'NERSC_HOST' in os.environ.keys():
    print(f'TauNet: Running on {os.environ["NERSC_HOST"]}')
    DATADIR = os.path.join(os.environ['PSCRATCH'],'TauNet','Data')
    FFP8 = ''
    warnings.warn('FFP8 not available on NERSC')
else:
    print(f'TauNet: Running on {os.environ["HOSTNAME"]}')
    DATADIR = os.path.join(os.environ['WORK'],'anto','TauNet','Data')
    FFP8 = '/marconi_work/INF24_litebird/lpagano0/4anto/ffp8_covmats'

os.makedirs(DATADIR, exist_ok=True)
DB_LOCAL = False
DB_DIR = './'
