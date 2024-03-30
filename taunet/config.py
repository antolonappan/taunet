import os

DATADIR = os.path.join(os.environ['WORK'],'anto','TauNet','Data')
os.makedirs(DATADIR, exist_ok=True)
FFP8 = '/marconi_work/INF24_litebird/lpagano0/4anto/ffp8_covmats'
DB_TESTING = False
DB_DIR = './'