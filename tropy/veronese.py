from itertools import combinations
import numpy as np


def lattice_values(d: int, size: int):
  """Generate sorted integer values whose sum is smaller than size"""
  if size == 0:
    yield []
  if size == 1:
    yield [1]
  else:
    for k in range(1, size + 1):
      for cur in lattice_values(d, size - k):
        cur.append(k)
        yield cur


def simplex_lattice_points(d: int, size: int):
  """Generate the lattice points of dilated simplex"""
  for items in lattice_values(d, size):
    for comb in combinations(np.arange(d), len(items)):
      yield comb, items


def newton_polynomial(lattice_points: list, apex: np.ndarray, d: int) -> tuple[list[list[float]], list[float]]:
  """Generate the Newton polynomial associated with some dilated simplex"""
  coefficients = (-apex).tolist()
  monomials = []
  for elem in lattice_points:
    monomials.append([0]*d)
    for k, v in zip(elem[0], elem[1]):
      monomials[-1][k] = v
  return monomials, coefficients


def hypersurface_polymake_code(lattice_points: list, apex: np.ndarray, initial_dim: int) -> str:
  """Generate the polymake code for visualizing some tropical polynomial"""
  lattice_list, coef_list = newton_polynomial(lattice_points, apex, initial_dim)
  return f'$C = new Hypersurface<Max>(MONOMIALS=>{str(lattice_list)}, COEFFICIENTS=>{str(coef_list)});'


def hypersurface_nodes(monomials: list[np.ndarray], coefficients: list[float], d: int, tol: float = 1e-6) -> list[tuple[np.ndarray, int]]:
  """Computes the nodes of a tropical polynomial's hypersurface"""
  results = []
  for comb in combinations(np.arange(len(coefficients)), d):
    M, b = np.zeros((d, d)), np.zeros(d)
    M[0] += 1
    for i in range(1, d):
      M[i] = monomials[comb[i]] - monomials[comb[0]]
      b[i] = - (coefficients[comb[i]] - coefficients[comb[0]])
    if not np.isclose(np.linalg.det(M), 0):
      try:
        x = np.linalg.solve(M, b)
        # Verify that x is effectively in a maximum sector
        expected_max = np.sum(monomials[comb[0]] * x) + coefficients[comb[0]]
        ignore = False
        for i in range(0, len(coefficients)):
          if i not in comb:
            val = np.sum(monomials[i] * x) + coefficients[i]
            if val > expected_max + tol:
              ignore = True
              break
        if not ignore:
          results.append((x, comb))
      except np.linalg.LinAlgError:
        pass
  return results


def map_to_exponential_space(Xlist: np.ndarray, beta: float):
  """Maps some point clouds using exp(beta*.) transformation, useful in tropicalization"""
  Xplus, Xminus = Xlist
  xplus, xminus = np.exp(beta * Xplus), np.exp(beta * Xminus)
  x = np.concatenate((xplus, xminus), axis=1)
  y = np.array([1] * xplus.shape[1] + [-1] * xminus.shape[1])
  return x, y