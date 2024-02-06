from tropy.veronese import newton_polynomial, simplex_lattice_points
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def op_norm(d, s, varying_d):
  lattice_points = list(simplex_lattice_points(d, s))
  V, _ = newton_polynomial(lattice_points, np.zeros(len(lattice_points)), d)
  th_lo, th_up, th_up_lim = 1/s, d/(s+d) + (s*d*d)/((s+d)**2), (1+s if varying_d else d*(d+1)/s)
  opnorm = np.max(np.sum(np.abs(np.linalg.inv(V.T @ V) @ V.T), axis=1))
  return th_lo, opnorm, th_up, th_up_lim

S = 3
vals = []
for d in tqdm(range(1,40)):
  vals.append(op_norm(d, S, True))

fig, axes = plt.subplots(2, 2, figsize=(14,8))

axes[0, 0].plot(vals, label=['lo', 'norm', 'hi', '$y=1+s$'])
axes[0, 0].set_xlabel('d = dimension')
axes[0, 0].set_ylabel('margin factor')
axes[0, 0].set_title(f'fixed polynomial degree $s = {S}$')
axes[0, 0].legend()

D = 3
vals2 = []
for s in tqdm(range(1,20)):
  vals2.append(op_norm(D, s, False))

axes[0, 1].plot(vals2, label=['lo', 'norm', 'hi', '$y=d(d+1)/s$'])
axes[0, 1].set_xlabel('s = polynomial degree')
axes[0, 1].set_ylabel('margin factor')
axes[0, 1].set_title(f'fixed dimension $d = {D}$')
axes[0, 1].legend()


axes[1, 0].plot([(vals[i][0]/vals[i][1], 1, vals[i][2]/vals[i][1], vals[i][3]/vals[i][1]) for i in range(len(vals))],
                label=['lo/norm', 'ref', 'hi/norm', 'hi asympt./norm'])
axes[1, 0].set_xlabel('d = dimension')
axes[1, 0].set_ylabel('margin factor')
axes[1, 0].set_title(f'fixed polynomial degree $s = {S}$')
axes[1, 0].legend()

axes[1, 1].plot([(vals2[i][0]/vals2[i][1], 1, vals2[i][2]/vals2[i][1], vals2[i][3]/vals2[i][1]) for i in range(len(vals2))],
                label=['lo/norm', 'ref', 'hi/norm', 'hi asympt./norm'])
axes[1, 1].set_xlabel('s = polynomial degree')
axes[1, 1].set_ylabel('margin factor')
axes[1, 1].set_title(f'fixed dimension $d = {D}$')
axes[1, 1].legend()


plt.tight_layout()
plt.show()