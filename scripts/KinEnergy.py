#!/usr/bin/env python3
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.style
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt


import numpy as np
import celldiv as cd
import argparse
from tqdm import tqdm
import pandas as pd

parser = argparse.ArgumentParser()

#parser.add_argument("traj", type=str)

parser.add_argument("-k", "--skip",
                    help="Trajectory frame skip rate",
                    type = int,
                    default = 1)

parser.add_argument("-s", "--step",
                    help="Trajectory frame skip rate",
                    type = int,
                    default = 1)

#parser.add_argument('dt', type=float)

parser.add_argument('csv')


args=parser.parse_args()

# t = 0
# dt = args.skip*args.dt
# with cd.TrajHandle(args.traj) as th:
#     t = dt*(int(th.maxFrames/args.skip))
#     print(dt, t)
#     keList = [0 for i in range(int(th.maxFrames/args.skip))]
#     f1 = np.vstack(th.ReadFrame(inc=args.skip))
#     f1 -= np.mean(f1, axis=0)
#     for i in tqdm(range(int(th.maxFrames/args.skip))):
#         frame = np.vstack(th.ReadFrame(inc=args.skip))
#         frame -= np.mean(frame, axis=0)
#         Vs = (frame[:f1.shape[0]] - f1)/dt
#         keList[i] = 0.5*np.sum(np.linalg.norm(Vs, axis=1)**2)
#         f1 = np.copy(frame)


# plt.plot(keList, lw=1.5, label="calculated")


d = pd.read_csv(args.csv, usecols=["step", "F", "V"]).as_matrix()
keList2 = []
FList=[]
VList =[]
for s in tqdm(range(0, int(d[:,0].max()), args.step*args.skip)):
    V = d[d[:, 0] == s, 1]
    F = d[d[:, 0] == s, 2]
    keList2.append(np.sum(0.5*V**2))
    FList.append(np.sum(F))

plt.plot(keList2, '.', lw=1.5, label="velocity")
plt.legend()
plt.gca().set_yscale('log')
plt.savefig('ke.png')
plt.cla()
plt.semilogy(FList, '.', label="force")
plt.savefig('force.png')
plt.cla()
# plt.plot(VList, '.', label="volumes")
# plt.savefig('vol.png')
