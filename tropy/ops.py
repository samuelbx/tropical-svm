import numpy as np
from .utils import min_min2_idx, max_max2_idx

_min_plus_mode: bool = False


def _set_mode(min_plus: bool = False) -> None:
  _min_plus_mode = min_plus


def dot(A: np.ndarray, B: np.ndarray, force_min_plus: bool = None):
  assert A.ndim == 1 and B.ndim == 1
  min_plus = force_min_plus if force_min_plus is not None else _min_plus_mode
  return np.min(A + B) if min_plus else np.max(A + B)


def matmul(A: np.ndarray, B: np.ndarray, force_min_plus: bool = None):
  assert A.ndim == 2 and B.ndim == 2
  min_plus = force_min_plus if force_min_plus is not None else _min_plus_mode
  if min_plus:
    return np.min(A[:, np.newaxis, :] + B.T[np.newaxis, :, :], axis=2)
  return np.max(A[:, np.newaxis, :] + B.T[np.newaxis, :, :], axis=2)


def matvec(A: np.ndarray, v: np.ndarray, force_min_plus: bool = None):
  assert A.ndim == 2 and v.ndim == 1
  min_plus = force_min_plus if force_min_plus is not None else _min_plus_mode
  if min_plus:
    return np.min(A + v[np.newaxis, :], axis=1)
  return np.max(A + v[np.newaxis, :], axis=1)


# Projects x over convex hull of C (optionally diagonally-free)
def proj(C: np.ndarray,
         x: np.ndarray,
         DF: bool = False,
         force_min_plus: bool = None) -> np.ndarray:
  assert C.ndim == 2 and x.ndim == 1
  d, p = C.shape
  Caug = -C + x[:, None]
  min_plus = force_min_plus if force_min_plus is not None else _min_plus_mode
  ext_i, ext2_i = min_min2_idx(Caug) if not min_plus else max_max2_idx(Caug)
  if DF:
    choices = np.where((ext_i == np.arange(d)[:, None]), ext2_i, ext_i)
    return np.max(C + Caug[choices, np.arange(p)], axis=1)
  else:
    return np.max(C + Caug[ext_i, np.arange(p)], axis=1)


# Projects point x over hyperplane of apex
def proj_hyperplane(apex: np.ndarray,
                    x: np.ndarray,
                    force_min_plus: bool = False) -> np.ndarray:
  assert apex.ndim == 1, x.ndim == 1
  min_plus = force_min_plus if force_min_plus is not None else _min_plus_mode
  diff = x - apex
  ext_i, ext2_i = min_min2_idx(diff) if min_plus else max_max2_idx(diff)
  vec = x.copy()
  vec[ext_i] = apex[ext_i] + diff[ext2_i]
  return vec


# Computes Veronese embedding of point clouds C
def veronese(lattice_points: list[np.ndarray], C: np.ndarray):
  Cprime = np.zeros((len(lattice_points), C.shape[1]))
  for i, (comb, values) in enumerate(lattice_points):
    for idx, val in zip(comb, values):
      Cprime[i, :] += C[idx, :] * val
  return Cprime
