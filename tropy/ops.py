import numpy as np
from .utils import min_min2_idx, max_max2_idx


# Projects x over convex hull of C (optionally diagonally-free)
def proj(C: np.ndarray,
         x: np.ndarray,
         DF: bool = False,
         min_plus: bool = False) -> np.ndarray:
  assert C.ndim == 2 and x.ndim == 1
  d, p = C.shape
  Caug = -C + x[:, None]
  ext_i, ext2_i = min_min2_idx(Caug) if not min_plus else max_max2_idx(Caug)
  if DF:
    choices = np.where((ext_i == np.arange(d)[:, None]), ext2_i, ext_i)
  else:
    choices = ext_i
  return np.max(C + Caug[choices, np.arange(p)], axis=1)


# Projects point x over hyperplane of apex
def proj_hyperplane(apex: np.ndarray,
                    x: np.ndarray,
                    min_plus: bool = False) -> np.ndarray:
  assert apex.ndim == 1, x.ndim == 1
  diff = x - apex
  ext_i, ext2_i = min_min2_idx(diff) if min_plus else max_max2_idx(diff)
  vec = x.copy()
  vec[ext_i] = apex[ext_i] + diff[ext2_i]
  return vec


# Computes Veronese embedding of point clouds C
def veronese(lattice_points: list[np.ndarray], C: list[np.ndarray]):
  Cprime = [np.zeros((len(lattice_points), C[i].shape[1])) for i in range(len(C))]
  for i, (comb, values) in enumerate(lattice_points):
    for idx, val in zip(comb, values):
      for k in range(len(C)):
        Cprime[k][i, :] += C[k][idx, :] * val
  return Cprime
