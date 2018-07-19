import numpy as np

class Cell(object):
    def __init__(self, _idx, _pos, _vels=None, _forces=None, _vol=None):
        self.idx = _idx
        self.pos = _pos
        self.vels = _vels
        self.forces = _forces
        self.volume = _vol


    def CoM(self):
        return np.mean(self.pos, axis=0)

    def Cells(self):
        return self.cells

    def __str__(self):
        return "Cell{} CoM = {}, volume={}".format(self.idx, self.CoM(), self.volume)
