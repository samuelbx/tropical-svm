import numpy as np
import matplotlib.pyplot as plt
from tropy.learn import fit_hyperplane
from tropy.graph import init_ax, plot_points, plot_hyperplane
from tropy.utils import apply_noise

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
  apply_noise(Cplus)
  apply_noise(Cminus)

  Cplus_sep, Cminus_sep = Cplus.copy(), Cminus.copy()
  Cplus_sep[1, :] -= 10
  Cminus_sep[1, :] += 10

  L = 15  # Plot scale factor

  fig = plt.figure()
  ax0 = init_ax(fig, 221, L)
  ax1 = init_ax(fig, 222, L)
  ax2 = init_ax(fig, 223, L)
  ax3 = init_ax(fig, 224, L)

  # Example, separable case
  ax0.set_title("Raw separable")
  plot_points(ax0, Cplus_sep, L, "r", marker="o")
  plot_points(ax0, Cminus_sep, L, "b", linestyle="dashed", marker="v")
  plot_hyperplane(ax1, "Inrad method, separable",
                  *fit_hyperplane(Cplus_sep, Cminus_sep), L)

  # Example, inseparable case
  ax2.set_title("Raw inseparable")
  plot_points(ax2, Cplus, L, "r", marker="o")
  plot_points(ax2, Cminus, L, "b", linestyle="dashed", marker="v")
  plot_hyperplane(ax3, "Inrad method, inseparable",
                  *fit_hyperplane(Cplus, Cminus), L)

  plt.show()
