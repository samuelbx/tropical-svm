import numpy as np


def min_min2_idx(a: np.ndarray) -> np.ndarray:
  """Get the indices of minimum and 2nd minimum of array"""
  return np.argpartition(a, 1, axis=0)[:2]


def max_max2_idx(a: np.ndarray) -> np.ndarray:
  """Get the indices of maximum and 2nd maximum of array"""
  return min_min2_idx(-a)


def count_points_sectors(C: np.ndarray, apex: np.ndarray) -> np.ndarray:
  """Count the number of data points in each sector wrt some apex"""
  assert C.ndim == 2, apex.ndim == 1
  differences = C - apex[:, np.newaxis]
  max_indices = np.argmax(differences, axis=0)
  I = np.zeros(C.shape[0])
  np.add.at(I, max_indices, 1)
  return I


def apply_noise(C: np.ndarray, mu: int = 1, seed: int = 2024) -> np.ndarray:
  """Apply Gaussian noise of amplitude mu to data"""
  np.random.seed(seed)
  r, c = C.shape
  C += mu * np.random.randn(r, c)
  return C


def build_toy_tropical_data(n_points: int, n_features: int, n_positive_sectors: int, 
                      noise: bool = False, L: int = 10, seed: int = 2024) -> tuple[np.ndarray, np.ndarray]:
  """Build a toy tropically separable dataset"""
  assert n_positive_sectors <= n_features
  np.random.seed(seed)
  C = np.random.uniform(-L, L, (n_features, n_points))
  max_indices = np.argmax(C, axis=0)
  positive_mask = np.isin(max_indices, range(n_positive_sectors))
  negative_mask = ~positive_mask
  Cplus = C[:, positive_mask]
  Cminus = C[:, negative_mask]
  if noise:
    apply_noise(Cplus, seed=seed)
    apply_noise(Cminus, seed=seed)
  return Cplus, Cminus