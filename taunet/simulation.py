import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import camb


class MapObject:
    def __init__(self) -> None:
        self.cmb = None
        self.fg = None
        self.noise = None
        self.tau = None
    
    @property
    def total(self):
        assert self.cmb is not None
        assert self.fg is not None
        assert self.noise is not None
        return self.cmb + self.fg + self.noise

class CMBgen:

    def __init__(self,nsim,nside,tau):
        self.nsim = nsim
        self.nside = nside
