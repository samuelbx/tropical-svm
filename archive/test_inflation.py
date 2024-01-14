import numpy as np
import matplotlib.pyplot as plt
from tropy.learn import fit_hyperplane, _inrad_eigenpair
from tropy.graph import init_ax, plot_points, plot_hyperplane, plot_ball
from tropy.ops import proj


def ctype(C: np.ndarray, x: np.ndarray, min_plus: bool = False) -> np.ndarray:
  _, p = C.shape
  L = np.zeros(p)
  for i, p in enumerate(C.T):
    L[i] = np.argmin(x - p) if min_plus else np.argmax(x - p)
  return L


def ctype_matrix(C: np.ndarray, min_plus: bool = False) -> np.ndarray:
  _, p = C.shape
  L = np.zeros((p, p))
  for i, p in enumerate(C.T):
    L[i, :] = ctype(C, p, min_plus)
  return L


def get_branches(C: np.ndarray):
  ctype_mat = ctype_matrix(C, min_plus=True)
  idxes = []
  for i, row in enumerate(ctype_mat):
    val = row[0] if i > 0 else row[1]
    row -= val
    row[i] += val
    if not row.any():
      idxes.append(i)
  return idxes


def inflate_branches_half(C: np.ndarray,
                          Cprime: np.ndarray,
                          radius: float,
                          tol: float = 1e-6) -> np.ndarray:
  d, p = C.shape
  branches = get_branches(C)

  # Check whether they are included in convex hull of rest
  for i, b in enumerate(branches):
    vec = C[:, b]
    dist = np.linalg.norm(vec - proj(Cprime, vec))
    if dist > tol:
      del branches[i]

  # Inflate branches
  C_new = np.zeros((d, p + len(branches) * d))
  C_new[:, :p] = C
  for i, b in enumerate(branches):
    C_new[:, p + d * i:p + d * (i + 1)] = (C[:, b] + np.eye(d) * radius).T

  return C_new


def inflate_branches(Cplus: np.ndarray, Cminus: np.ndarray,
                     radius: float) -> tuple[np.ndarray]:
  return inflate_branches_half(Cplus, Cminus, radius), inflate_branches_half(
      Cminus, Cplus, radius)


if __name__ == "__main__":
  # Data points
  Cplus = np.array(
      [
          [11, -13, 2],
          [13, -14, 1],
          [14, -7, -7],
          [10, 1, -11],
          [3, -3, 0],
      ],
      dtype=float,
  ).T
  Cminus = np.array([[10, -5, -5], [-3, 6, -3], [-2, -5, 7]], dtype=float).T

  L = 15  # Plot scale factor

  fig = plt.figure()
  ax0 = init_ax(fig, 221, L)
  ax1 = init_ax(fig, 222, L)
  ax2 = init_ax(fig, 223, L)
  ax3 = init_ax(fig, 224, L)

  # Example, inseparable case
  ax0.set_title("1: Raw inseparable")
  plot_points(ax0, Cplus, L, "r", marker="o")
  plot_points(ax0, Cminus, L, "b", linestyle="dashed", marker="v")
  x, l = _inrad_eigenpair(Cplus, Cminus)
  plot_ball(ax0, x, l)

  Cplus_trans, Cminus_trans, x, l = fit_hyperplane(Cplus,
                                                   Cminus,
                                                   old_method=False,
                                                   noise=True)
  plot_hyperplane(ax1, "2: Inrad method", Cplus_trans, Cminus_trans, x, l, L)

  ax2.set_title("3: Inflated relevant branches")
  Cplus_infl, Cminus_infl = inflate_branches(Cplus_trans, Cminus_trans, 1)
  plot_points(ax2, Cplus_infl, L, "r", marker="o")
  plot_points(ax2, Cminus_infl, L, "b", linestyle="dashed", marker="v")
  x, l = _inrad_eigenpair(Cplus_infl, Cminus_infl)
  plot_ball(ax2, x, l)

  Cplus_trans2, Cminus_trans2, x, l = fit_hyperplane(Cplus_infl,
                                                     Cminus_infl,
                                                     old_method=False,
                                                     noise=True)
  plot_hyperplane(ax3, "4: New inrad method", Cplus_trans2, Cminus_trans2, x, l,
                  L)

  plt.show()
