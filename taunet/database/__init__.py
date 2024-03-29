from taunet import DB_TESTING

if DB_TESTING:
    from .local_database import *
else:
    from .remote_database import *
