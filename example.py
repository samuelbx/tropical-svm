import numpy as np
import matplotlib.pyplot as plt
from tropy.svm import TropicalSVM
from tropy.graph import init_ax, plot_classes, plot_hyperplane, set_title, get_ignored
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
  apply_noise(Cplus, 0.3)
  apply_noise(Cminus, 0.3)
  Cplus_sep, Cminus_sep = Cplus.copy(), Cminus.copy()
  Cplus_sep[1, :] -= 10
  Cminus_sep[1, :] += 10

  # Fitting tropical SVM
  model1, model2 = TropicalSVM(), TropicalSVM()
  model1.fit([Cplus_sep, Cminus_sep])
  model2.fit([Cplus, Cminus])

  L = 15  # Plot scale factor
  fig = plt.figure()
  ax1 = init_ax(fig, 121, L)
  ax2 = init_ax(fig, 122, L)

  # Example, separable case
  plot_classes(ax1, [Cplus_sep, Cminus_sep], L, show_lines=True)
  ignored_branch = get_ignored(Cplus_sep, Cminus_sep, model1.apex)
  plot_hyperplane(ax1, model1.apex, model1.eigval, L, ignored_branch)
  set_title(ax1, "Inrad method, separable", model1.apex, model1.eigval)

  # Example, inseparable case
  plot_classes(ax2, [Cplus, Cminus], L, show_lines=True)
  ignored_branch = get_ignored(Cplus, Cminus, model2.apex)
  plot_hyperplane(ax2, model2.apex, model2.eigval, L, ignored_branch)
  set_title(ax2, "Inrad method, inseparable", model2.apex, model2.eigval)

  plt.show()
