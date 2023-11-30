import numpy as np
from itertools import permutations, combinations


def min_min2_idx(a: np.ndarray) -> np.ndarray:
  return np.argpartition(a, 1, axis=0)[:2]


def max_max2_idx(a: np.ndarray) -> tuple[float, float]:
  return min_min2_idx(-a)


# Returns the number of points of class C in each sector wrt apex
def count_points_sectors(C: np.ndarray, apex: np.ndarray) -> np.ndarray:
  assert C.ndim == 2, apex.ndim == 1
  I = np.zeros(C.shape[0])
  for col in C.T:
    max_i, max2_i = max_max2_idx(col - apex)
    if max_i != max2_i:
      I[max_i] += 1
  return I


# Applies Normal Gaussian noise to data
def apply_noise(C: np.ndarray, seed: int = 2024) -> np.ndarray:
  np.random.seed(seed)
  r, c = C.shape
  C += np.random.randn(r, c)


# Generates sorted integer values whose sum is smaller than size
def lattice_values(d: int, size: int):
  if size == 0:
    yield []
  if size == 1:
    yield [1]
  else:
    for k in range(1, size + 1):
      for cur in lattice_values(d, size - k):
        cur.append(k)
        yield cur


# Generates the integer points of the dilated simplex
def simplex_lattice_points(d: int, size: int):
  for items in lattice_values(d, size):
    for permutated_items in permutations(items):
      for comb in combinations(np.arange(d), len(permutated_items)):
        yield comb, permutated_items
