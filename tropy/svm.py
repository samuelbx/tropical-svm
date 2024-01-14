import numpy as np
from .ops import proj, veronese
from .utils import count_points_sectors
from .veronese import map_to_exponential_space, simplex_lattice_points
from itertools import combinations
from sklearn.svm import LinearSVC


class TropicalSVM():
  
  predictor = None

  def fit(self, data_classes: np.ndarray, veronese_size: int = 1, one_vs_all: bool = False):
    """
    TODO: write documentation
    """
    d = data_classes[0].shape[0]
    self.veronese_coefficients = list(simplex_lattice_points(d, veronese_size))
    aug_data_classes = veronese(self.veronese_coefficients, data_classes)
    self.apex, self.eigval = _inrad_eigenpair(aug_data_classes, N=50)

    if not one_vs_all:
      # TODO: Reset apex coordinates corresponding to unreached sectors to infty?
      Counts = np.array([count_points_sectors(C, self.apex) for C in aug_data_classes])
      zero_mask = np.all(Counts == 0, axis=0)
      self.apex[zero_mask] = np.inf
      sector_indicator = np.argmax(Counts, axis=0)
      sector_indicator[zero_mask] = -1

      def _predict(point: np.ndarray) -> int:
        sector = np.argmax(point - self.apex)
        return sector_indicator[sector]

      self.predictor, self.sector_indicator = _predict, sector_indicator
    
    else:
      # Useful for testing purposes, might be removed in the final version
      apices = _apply_inrad_onevsall(aug_data_classes, x0=self.apex)
      sector_indicators = np.zeros((len(aug_data_classes), aug_data_classes[0].shape[0]))
      for i, apex in enumerate(apices):
        Counts = np.array([count_points_sectors(C, apex) for C in aug_data_classes])
        sector_indicators[i] = np.where(np.argmax(Counts, axis=0) == i, 1, 0)

      # HACK: Random prediction based on population ratios 
      def _predict(x):
        predictions = np.zeros(len(apices))
        for i, apex in enumerate(apices):
          sector = np.argmax(x - apex)
          predictions[i] = sector_indicators[i][sector] * aug_data_classes[i].shape[1]
        if predictions.sum() != 0:
          predictions /= predictions.sum()
          choic = np.random.choice(len(predictions), p=predictions)
          return choic
        else:
          return -1

    self._predictor = _predict

  def predict(self, data_points: np.ndarray):
    assert self._predictor is not None, "Model must be trained before prediction"
    aug_data_points = veronese(self.veronese_coefficients, [data_points])[0]
    return [self._predictor(row) for row in aug_data_points.T]


def _apply_inrad_onevsall(Clist: list[np.ndarray],
                          N: int = 50,
                          x0: np.ndarray = None):
  apices = []
  if x0 is None:
    x0 = np.ones(Clist[0].shape[0])
  for i, C in enumerate(Clist):
    Cminus = np.concatenate([Clist[j] for j in range(len(Clist)) if j != i], axis=1)
    apices.append(_inrad_eigenpair([Cminus, C], N, x0.copy())[0])
  return apices


# Describes the intersection between two convex hulls
def _inrad_op(Clist: list[np.ndarray]) -> callable:
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

  return op


def fit_tropicalized_linear_SVM(Xtrain: np.ndarray, beta: float = 1):
  xtrain, ytrain = map_to_exponential_space(Xtrain, beta)
  model = LinearSVC(dual=True, fit_intercept=False)
  clf = model.fit(xtrain.T, ytrain)
  w = clf.coef_[0]
  return model, w


# Determine eigenpairs of Shapley operators
def _krasnoselskii_mann(op: callable, N: int,
                        x0: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, float]:
  x, z = x0.copy(), np.zeros_like(x0)
  for _ in range(N):
    z = (x + op(x)) / 2
    new_x = z - np.max(z) * np.ones_like(x0)
    if np.linalg.norm(new_x - x) < tol:
      x = new_x
      break
    x = new_x
  return x - x.mean(), 2 * np.max(z)


# Separation using inner radius operator
def _inrad_eigenpair(Clist: list[np.ndarray], N: int = 20, x0=None):
  if x0 is None:
    x0 = np.ones(Clist[0].shape[0])
  return _krasnoselskii_mann(_inrad_op(Clist), N, x0)
