import csv
import numpy as np
import pandas as pd
from .ops import proj, veronese
from .utils import count_points_sectors, max_max2_idx
from .veronese import map_to_exponential_space, simplex_lattice_points, newton_polynomial
from itertools import combinations
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time


class TropicalSVC():
  
  _monomials, _coeffs = None, None

  def fit(self, data_classes: list[np.ndarray], poly_degree: int = 1, native_tropical_data: bool = False, log_linear_beta: bool = None, feature_selection: int = None) -> None:
    """Fit the model according to the given training data.
    data_classes: list of classes (2D NumPy arrays whose columns are the data points)
    poly_degree: degree of the tropical polynomial to fit
    native_tropical_data: whether the data is intrinsically tropical or not (default: False)
    log_linear_beta: option to use Maslov's dequantization trick
    feature_selection: (experimental) heuristic for adaptive monomial choice, overrides poly_degree
    """
    # By default, map d-dimensional data into subspace H: x1+...+x^{d+1}=0 of R^{d+1}
    data_classes_copy = data_classes.copy()
    if not native_tropical_data:
      for i, data in enumerate(data_classes_copy):
        data_classes_copy[i] = np.vstack((-np.sum(data, axis=0), data))
    d = data_classes_copy[0].shape[0]
    self._tropical_data, self._log_linear_beta, self._poly_degree = native_tropical_data, log_linear_beta, d
    self._data_classes = data_classes_copy
    
    # Choose features using combinations from d-dimensional simplex, or experimental feature selection heuristic
    if feature_selection is None:
      monomials_idxes = list(simplex_lattice_points(d, poly_degree))
      aug_data_classes = veronese(monomials_idxes, data_classes_copy)
    else:
      aug_data_classes, monomials_idxes = _experimental_feature_selection(data_classes_copy, d, feature_selection)

    # Handle log-linear mode
    if log_linear_beta:
      assert len(aug_data_classes) == 2
      self._log_lin_model, w = fit_tropicalized_linear_SVM(aug_data_classes, log_linear_beta)
      self._apex = np.sign(w) * np.log(np.sign(w) * w)/log_linear_beta
    else:
      # Compute apex
      self._apex, self._eigval = _inrad_eigenpair(aug_data_classes, N=1000)

    # TODO: use less lines of code
    projections = _tropical_projections(aug_data_classes, self._apex)
    sector_indicator = np.argmax(projections, axis=0)
    idxes = max_max2_idx(projections)
    max_idxes, max2_idxes = idxes[0], idxes[1]
    sector_indicator = max_idxes
    for i in range(len(sector_indicator)):
      if np.isclose(projections[max_idxes[i], i], projections[max2_idxes[i], i]):
        sector_indicator[i] = -1
    self._sector_indicator = sector_indicator
    #zero_mask = np.where(self._sector_indicator == -1)[0]

    """# Assign secotrs based on majoritary population
    Counts = np.zeros((len(aug_data_classes), self._apex.shape[0]))
    for i, C in enumerate(aug_data_classes):
      Counts[i] = count_points_sectors(C, self._apex)
    zero_mask = np.all(Counts == 0, axis=0)
    sector_indicator = np.argmax(Counts, axis=0)
    sector_indicator[zero_mask] = -1
    self._sector_indicator = sector_indicator"""
    
    # Save model weights
    self._monomials, self._coeffs = newton_polynomial(monomials_idxes, self._apex, self._poly_degree)

    # Only keep active monomials
    #self._monomials = self._monomials[zero_mask == False]
    #self._coeffs = self._coeffs[zero_mask == False]
    #self._sector_indicator = self._sector_indicator[zero_mask == False]

  def predict(self, data: np.ndarray) -> list[int]:
    """Predict the labels of some data points (as a 2D matrix whose columns are the points)"""
    assert self._monomials is not None, "Model must be trained before prediction"
    data_copy = data.copy()
    if not self._tropical_data:
      data_copy = np.vstack((-np.sum(data_copy, axis=0), data_copy))
    evaluation = self._monomials @ data_copy + self._coeffs[:, np.newaxis]
    return self._sector_indicator[np.argmax(evaluation, axis=0)]
  
  def accuracy(self, data_classes: np.ndarray) -> float:
    true_labels = []
    predicted_labels = []
    for label, points in enumerate(data_classes):
      true_labels.extend([label] * points.shape[1])
      predicted_labels.extend(self.predict(points))
    return accuracy_score(true_labels, predicted_labels)

  def export_weights(self, file: str) -> None:
    assert self._monomials is not None, "Model must be trained before prediction"
    data_matrix = np.column_stack((self._monomials, self._coeffs, self._sector_indicator))
    valid_rows = data_matrix[self._sector_indicator != -1]
    with open(file, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerows(valid_rows)
  
  def load_weights(self, file) -> None:
    weights = pd.read_csv(file).to_numpy()
    self._monomials = weights[:, :-2]
    self._coeffs = weights[:, -2]
    self._sector_indicator = weights[:, -1]
  
  def margin(self) -> float:
    """Margin of the tropical classifier"""
    op_norm = np.max(np.sum(np.abs(self._monomials), axis=1)) # L^inf norm of Veronese embedding
    print(self._eigval, op_norm)
    return max(0.0, -self._eigval) / op_norm


def _tropical_projections(Clist: list[np.ndarray], x: np.ndarray) -> np.ndarray:
  d = Clist[0].shape[0]
  projections = np.zeros((len(Clist), d))
  for i, C in enumerate(Clist):
    projections[i] = proj(C, x, DF=True)
  return projections


# TODO: use previous function and max2
def _inrad_op(Clist: list[np.ndarray]) -> callable:
  """Returns the Shapley operator for tropical linear classification"""
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


def fit_tropicalized_linear_SVM(data_classes: np.ndarray, beta: float = 1) -> tuple[LinearSVC, np.ndarray]:
  """Compute an approximating tropical hyperplane by Maslov's dequantization"""
  xtrain, ytrain = map_to_exponential_space(data_classes, beta)
  model = LinearSVC(dual=True, fit_intercept=False)
  clf = model.fit(xtrain.T, ytrain)
  print('iter', clf.n_iter_)
  w = clf.coef_[0]
  return model, w


KM_TIME, KM_ITER = 0, 0
def _krasnoselskii_mann(op: callable, N: int,
                        x: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, float]:
  """Compute the eigenpair of a Shapley operator using Krasnoselskii-Mann iterations"""
  global KM_TIME, KM_ITER
  z = None
  t1 = time.time()
  for i in range(N):
    op_eval = op(x)
    z = (x + op_eval) / 2
    new_x = z - np.max(z)
    criterion = op_eval - x
    if np.max(criterion) - np.min(criterion) < tol * (np.max(x) - np.min(x)):
      x = new_x
      break
    x = new_x
  if i == N-1:
    print(f"WARN: KM did not converge in {N} iterations")
  else:
    print(f"KM converged in {i+1} iterations")
  t2 = time.time()
  KM_TIME, KM_ITER = t2 - t1, i
  return x - x.mean(), 2 * np.max(z)


def get_km_time():
  global KM_TIME
  return KM_TIME

def get_km_iter():
  global KM_ITER
  return KM_ITER


def _inrad_eigenpair(Clist: list[np.ndarray], N: int = 20) -> tuple[np.ndarray, float]:
  """Compute the eigenpair of the multi-class inner radius operator"""
  return _krasnoselskii_mann(_inrad_op(Clist), N, np.ones(Clist[0].shape[0]))


def _experimental_feature_selection(X: list[np.ndarray], d: int, no_samples: int):
  """(Experimental) feature selection heuristic"""
  # Sample random data points
  sampled_points = []
  for data_class in X:
    np.random.seed(42)
    sampled_indices = np.random.choice(data_class.shape[1], size=no_samples, replace=False)
    for idx in sampled_indices:
      sampled_points.append(data_class[:, idx])

  # Generate vectors between sampled points and monomials
  monomials = []
  for i in range(len(sampled_points)):
    p = sampled_points[i] - np.mean(sampled_points[i])
    for j in range(i+1, len(sampled_points)):
      p2 = sampled_points[j] - np.mean(sampled_points[j])
      segment = (p2 - p)/(np.linalg.norm(p2-p)**1)
      slope_vector = -d * segment + np.sum(segment)
      monomials.append(segment)
      monomials.append(segment + slope_vector)
      monomials.append(segment - slope_vector)
      monomials.append(-segment)
      monomials.append(-segment - slope_vector)
      monomials.append(-segment + slope_vector)

  # HACK: Generate list of tuples (index, monomial)
  monomials_idxes = [(tuple(range(d)), monomial.tolist()) for monomial in monomials]
  aug_data_classes = veronese(monomials_idxes, X)
  return aug_data_classes, monomials_idxes