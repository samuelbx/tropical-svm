import numpy as np
from .utils import min_min2_idx, max_max2_idx


def proj(C: np.ndarray, x: np.ndarray, DF: bool = False, min_plus: bool = False) -> np.ndarray:
  """Project x over convex hull of C (optionally diagonally-free)"""
  assert C.ndim == 2 and x.ndim == 1
  d, p = C.shape
  Caug = -C + x[:, None]
  ext_i, ext2_i = min_min2_idx(Caug) if not min_plus else max_max2_idx(Caug)
  if DF:
    choices = np.where((ext_i == np.arange(d)[:, None]), ext2_i, ext_i)
  else:
    choices = ext_i
  return np.max(C + Caug[choices, np.arange(p)], axis=1)


def proj_hyperplane(apex: np.ndarray, x: np.ndarray, min_plus: bool = False) -> np.ndarray:
  """Project x over hyperplane(apex)"""
  assert apex.ndim == 1, x.ndim == 1
  diff = x - apex
  ext_i, ext2_i = min_min2_idx(diff) if min_plus else max_max2_idx(diff)
  vec = x.copy()
  vec[ext_i] = apex[ext_i] + diff[ext2_i]
  return vec


def veronese(lattice_points: list[np.ndarray], data_classes: list[np.ndarray]) -> list[np.ndarray]:
  """Compute Veronese embedding of some list of data classes"""
  aug_data = [np.zeros((len(lattice_points), data_classes[i].shape[1])) for i in range(len(data_classes))]
  for i, (comb, values) in enumerate(lattice_points):
    for idx, val in zip(comb, values):
      for k in range(len(data_classes)):
        aug_data[k][i, :] += data_classes[k][idx, :] * val
  return aug_data
