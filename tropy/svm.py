import csv
import numpy as np
import pandas as pd
from .ops import proj, veronese
from .utils import count_points_sectors
from .veronese import map_to_exponential_space, simplex_lattice_points, newton_polynomial
from itertools import combinations
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


class TropicalSVC():
  
  _monomials, _coeffs = None, None

  def fit(self, data_classes: list[np.ndarray], poly_degree: int = 1, native_tropical_data: bool = False, log_linear_beta: bool = None, feature_selection: int = None) -> None:
    """Fit the model according to the given training data.
    data_classes: list of classes (2D numpy arrays whose columns are the data points)
    poly_degree: complexity parameter of the model
    tropical_data: whether the data is intrinsically tropical or not (default: False)
    log_linear_beta: option to use "linear hyperplane on logarithmic paper" trick
    feature_selection: (experimental) heuristic for adaptive monomial choice
    """
    # Map d-dimensional data into subspace H: x1+...+x^{d+1}=0 of R^{d+1}
    data_classes_copy = data_classes.copy()
    if not native_tropical_data:
      for i, data in enumerate(data_classes_copy):
        data_classes_copy[i] = np.vstack((-np.sum(data, axis=0), data))
    d = data_classes_copy[0].shape[0]
    self._tropical_data, self._log_linear_beta, self._poly_degree = native_tropical_data, log_linear_beta, d
    self._data_classes = data_classes_copy # TODO: remove
    
    if feature_selection is None: 
      # Default way of doing: compute coefficients from d-dimensional simplex
      monomials_idxes = list(simplex_lattice_points(d, poly_degree))
      aug_data_classes = veronese(monomials_idxes, data_classes_copy)
      self._veronese_coefficients = monomials_idxes
    
    else:
      # (Experimental) feature selection heuristic: first sample random data points
      sampled_points = []
      for data_class in data_classes_copy:
        np.random.seed(42)
        sampled_indices = np.random.choice(data_class.shape[1], size=feature_selection, replace=False)
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
      aug_data_classes = veronese(monomials_idxes, data_classes_copy)
      self._veronese_coefficients = monomials_idxes


    # Handle log-linear mode
    if log_linear_beta:
      assert len(aug_data_classes) == 2
      self._log_lin_model, w = fit_tropicalized_linear_SVM(aug_data_classes, log_linear_beta)
      self._apex = np.sign(w) * np.log(np.sign(w) * w)/log_linear_beta
    else:
      # Compute apex
      self._apex, self._eigval = _inrad_eigenpair(aug_data_classes, N=15)

    # Assign secotrs based on majoritary population
    Counts = np.zeros((len(aug_data_classes), self._apex.shape[0]))
    for i, C in enumerate(aug_data_classes):
      Counts[i] = count_points_sectors(C, self._apex)
    zero_mask = np.all(Counts == 0, axis=0)
    sector_indicator = np.argmax(Counts, axis=0)
    sector_indicator[zero_mask] = -1
    self._sector_indicator = sector_indicator
    
    # Save model weights
    self._monomials, self._coeffs = newton_polynomial(monomials_idxes, self._apex, self._poly_degree)

    # Only keep active monomials
    self._monomials = self._monomials[zero_mask == False]
    self._coeffs = self._coeffs[zero_mask == False]
    self._sector_indicator = self._sector_indicator[zero_mask == False]

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

  def export_weights(self, file) -> None:
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


def _inrad_op(Clist: list[np.ndarray]) -> callable:
  """Returns the Shapley operator describing the overlap between convex hulls"""
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
  """Compute an approximating tropical hyperplane by taking the logarithm of a linear SVM"""
  xtrain, ytrain = map_to_exponential_space(data_classes, beta)
  model = LinearSVC(dual=True, fit_intercept=False)
  clf = model.fit(xtrain.T, ytrain)
  w = clf.coef_[0]
  return model, w


def _krasnoselskii_mann(op: callable, N: int,
                        x0: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, float]:
  """Compute the eigenpair of a Shapley operator using Krasnoselskii-Mann iterations"""
  x, z = x0.copy(), np.zeros_like(x0)
  for _ in range(N):
    z = (x + op(x)) / 2
    new_x = z - np.max(z) * np.ones_like(x0)
    if np.linalg.norm(new_x - x) < tol:
      x = new_x
      break
    x = new_x
  return x - x.mean(), 2 * np.max(z)


def _inrad_eigenpair(Clist: list[np.ndarray], N: int = 20, x0=None) -> tuple[np.ndarray, float]:
  """Compute the eigenpair of the multi-class inner radius operator"""
  if x0 is None:
    x0 = np.ones(Clist[0].shape[0])
  return _krasnoselskii_mann(_inrad_op(Clist), N, x0)
