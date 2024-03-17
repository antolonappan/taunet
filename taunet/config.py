import os

#__path__ = os.path.dirname(os.path.realpath(__file__))
#DATADIR = os.path.join(__path__,'..','Data')

DATADIR = os.path.join(os.environ['WORK'],'anto','TauNet','Data')
os.makedirs(DATADIR, exist_ok=True)

SROLL2 = '/marconi_work/INF24_litebird/lpagano0/4anto/sroll2'
FFP8 = '/marconi_work/INF24_litebird/lpagano0/4anto/ffp8_covmats'