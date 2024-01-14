import numpy as np


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
def apply_noise(C: np.ndarray, noise: int = 1, seed: int = 2024) -> np.ndarray:
  np.random.seed(seed)
  r, c = C.shape
  C += noise * np.random.randn(r, c)


def build_toy_dataset(n_points: int, n_features: int, n_positive_sectors: int, 
                      noise: bool = False, L: int = 10, seed: int = 2024) -> tuple[np.ndarray, np.ndarray]:
  assert n_positive_sectors <= n_features
  np.random.seed(seed)
  C = np.random.uniform(-L, L, (n_features, n_points))
  max_indices = np.argmax(C, axis=0)
  positive_mask = np.isin(max_indices, range(n_positive_sectors))
  negative_mask = ~positive_mask
  Cplus = C[:, positive_mask]
  Cminus = C[:, negative_mask]
  if noise:
    apply_noise(Cplus, seed)
    apply_noise(Cminus, seed)
  return Cplus, Cminus