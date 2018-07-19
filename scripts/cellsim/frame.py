from cell import Cell

import numpy as np
class Frame(object):

    def __init__(self, _idx, _numCells=0):
        self.idx = _idx
        self.numCells = _numCells
        self.cells = []
        self.posStack = np.empty((3,1))

    def AddCell(self, _pos, _vel, _force, _vol):
        self.cells.append(Cell(self.numCells, _pos, _vel, _force, _vol))
        self.numCells += 1

    def _ReStack(self):
        self.pos

    def Positions(self):
        l = []
        for cell in self.cells:
            l.append(cell.pos)

        posStack = np.vstack(l)

        return posStack

    def CoM(self):
        return np.mean(self.Positions(), axis=0)

    def __str__(self):
        return "Frame {} containing {} cells.".format(self.idx, self.numCells)
