import os

if 'NERSC_HOST' in os.environ.keys():
    print(f'TauNet: Running on {os.environ["NERSC_HOST"]}')
    DATADIR = os.path.join(os.environ['PSCRATCH'],'TauNet','Data')
    DBDIR = os.path.join(os.environ['PSCRATCH'],'TauNet','DB')
else:
    print(f'TauNet: Running on {os.environ["HOSTNAME"]}')
    DATADIR = os.path.join(os.environ['WORK'],'anto','TauNet','Data')
    DBDIR = os.path.join(os.environ['WORK'],'anto','TauNet','DB')

os.makedirs(DATADIR, exist_ok=True)
os.makedirs(DBDIR, exist_ok=True)
