from taunet import DB_LOCAL

if DB_LOCAL:
    from .local_database import *
else:
    from .remote_database import *
