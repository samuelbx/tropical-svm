import numpy as np
from .ops import proj, proj_hyperplane
from .utils import count_points_sectors
from itertools import combinations
from tqdm import tqdm


def fit_hyperplane(Cplus: np.ndarray,
                   Cminus: np.ndarray,
                   old_method: bool = False,
                   N: int = 100,
                   tol: float = 1e-6):
  Cplus_loc, Cminus_loc = Cplus.copy(), Cminus.copy()

  x, l = _inrad_eigenpair([Cplus_loc, Cminus_loc], N)
  separable = l <= tol
  if not separable:
    Cplus_loc, Cminus_loc = _separating_trans(Cplus_loc, x, l), _separating_trans(Cminus_loc, x, l)
    l = 0
  if old_method:
    x, l = _cyclic_eigenpair(Cplus_loc, Cminus_loc, N)

  return Cplus_loc, Cminus_loc, x, l


def fit_multiple_onevsall(Clist: list[np.ndarray],
                          N: int = 50,
                          x0: np.ndarray = None):
  apices = []
  for i, C in enumerate(Clist):
    Cminus = np.concatenate([Clist[j] for j in range(len(Clist)) if j != i], axis=1)
    apices.append(_inrad_eigenpair([Cminus, C], N, x0)[0])
  return apices


def fit_classifier_onevsall(Clist: list[np.ndarray],
                            x0: np.ndarray = None) -> list[callable]:
  # HACK: Sectors are attributed with majoritary population
  apices = fit_multiple_onevsall(Clist, x0=x0)
  sector_indicators = np.zeros((len(Clist), Clist[0].shape[0]))
  for i, apex in enumerate(apices):
    Counts = np.array([count_points_sectors(C, apex) for C in Clist])
    sector_indicators[i] = np.where(np.argmax(Counts, axis=0) == i, 1, 0)
  return sector_indicators, apices


def predict_onevsall(sector_indicators, apices, Clist) -> int:

  def pred(x):
    predictions = np.zeros(len(apices))
    for i, apex in enumerate(apices):
      sector = np.argmax(x - apex)
      predictions[i] = sector_indicators[i][sector] * Clist[i].shape[1]
    if predictions.sum() != 0:
      predictions /= predictions.sum()
      choic = np.random.choice(len(predictions), p=predictions)
      return choic
    else:
      return -1

  return pred


# Describes the intersection between two convex hulls
def _inrad_op(Clist: list[np.ndarray], old_method: bool = False) -> callable:
  if not old_method:

    def op(x):
      d = Clist[0].shape[0]
      comb_idxes = list(combinations(range(len(Clist)), 2))
      projections = np.zeros((len(Clist), d))
      for i, C in enumerate(Clist):
        projections[i] = proj(C, x, DF=True)
      minimums = np.zeros((len(comb_idxes), d))
      for k, (i, j) in enumerate(comb_idxes):
        minimums[k] = np.minimum(projections[i], projections[j])
      return np.maximum.reduce(minimums, axis=0)
  else:

    def op(x):
      d = Clist[0].shape[0]
      projections = np.zeros((len(Clist), d))
      for i, C in enumerate(Clist):
        projections[i] = proj(C, x, DF=True)
      return np.minimum.reduce(projections, axis=0)

  return op


def _cyclic_projection(Cplus: np.ndarray, Cminus: np.ndarray) -> callable:
  return lambda x: proj(Cminus, proj(Cplus, x))


# Projects points from C close to an hyperplane over it
# TODO: use true criteria for moving "active" points
def _separating_trans(C: np.ndarray,
                      apex: np.ndarray,
                      r: float,
                      tol: float = 1e-3) -> np.ndarray:
  D = C.copy()
  for j, col in enumerate(D.T):
    proj_col = proj_hyperplane(apex, col)
    norm = np.linalg.norm(proj_col - col)
    if 0 < norm and norm <= r + tol:
      u = (proj_col - col) / norm
      D[:, j] = col + min(norm, r) * u
  return D


# Determine eigenpairs of Shapley operators
def _krasnoselskii_mann(op: callable, N: int,
                        x0: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, float]:
  x, z = x0.copy(), np.zeros_like(x0)
  for _ in tqdm(range(N)):
    z = (x + op(x)) / 2
    new_x = z - np.max(z) * np.ones_like(x0)
    if np.linalg.norm(new_x - x) < tol:
      x = new_x
      break
    x = new_x
  return x - x.mean(), 2 * np.max(z)


# Separation using inner radius operator
def _inrad_eigenpair(Clist: list[np.ndarray],
                     N: int = 50,
                     x0=None,
                     old_method=False):
  if x0 is None:
    x0 = np.ones(Clist[0].shape[0])
  return _krasnoselskii_mann(_inrad_op(Clist, old_method), N, x0)


def fit_classifier(Clist: list[np.ndarray], x: np.ndarray):
  # HACK: Sectors are attributed with majoritary population
  Counts = np.array([count_points_sectors(C, x) for C in Clist])
  sector_indicator = np.argmax(Counts, axis=0)
  zero_mask = np.all(Counts == 0, axis=0)
  sector_indicator[zero_mask] = -1

  def predict(point: np.ndarray) -> int:
    sector = np.argmax(point - x)
    return sector_indicator[sector]

  return predict, sector_indicator


# Generating vectors of Span(C) + B(0, r) (TODO: vectorize)
def _outer_generating_set(C: np.ndarray, radius: float) -> np.ndarray:
  d, p = C.shape
  output = np.zeros((d, d * p))
  for i, col in enumerate(C.T):
    for j in range(d):
      output[:, i * d + j] = col
      output[j, i * d + j] += radius
  return output


# Separation using cyclic projection algorithm
def _cyclic_eigenpair(Cplus: np.ndarray,
                      Cminus: np.ndarray,
                      N: int = 1000) -> tuple[np.ndarray, float]:
  _, l = _krasnoselskii_mann(_cyclic_projection(Cplus, Cminus), N,
                             np.ones(Cplus.shape[0]))
  Cplus_outer = _outer_generating_set(Cplus, -l / 2)
  Cminus_outer = _outer_generating_set(Cminus, -l / 2)
  x, _ = _krasnoselskii_mann(_cyclic_projection(Cplus_outer, Cminus_outer), N,
                             np.ones(Cplus.shape[0]))
  return x, -l / 2
