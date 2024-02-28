import numpy as np
import matplotlib.pyplot as plt
from tropy.svm import TropicalSVC
from tropy.graph import init_ax, plot_classes, plot_hyperplane_3d
from tropy.utils import apply_noise

if __name__ == "__main__":
  # Toy data points
  Cplus = np.array([[11, -13, 2], [13, -14, 1], [14, -7, -7], [10, 1, -11], [3, -3, 0]], dtype=float).T
  Cminus = np.array([[10, -5, -5], [-3, 6, -3], [-2, -5, 7]], dtype=float).T
  apply_noise(Cplus, 0.3)
  apply_noise(Cminus, 0.3)
  Cplus[1, :] -= 10
  Cminus[1, :] += 10

  # Fitting tropical SVM
  model, model2 = TropicalSVC(), TropicalSVC()
  model.fit([Cplus, Cminus], native_tropical_data=True)

  # Graphing (possible because toy data is 3D)
  L = 15  # Plot scale factor
  fig = plt.figure()
  ax = init_ax(fig, 111)
  plot_classes(ax, [Cplus, Cminus], L, show_lines=True)
  plot_hyperplane_3d(ax, model._apex, model._eigval, L, model._sector_indicator)
  ax.set_title(
    f"using mean payoff games, degree 1\n (apex = {np.round(model._apex, 2)}, {'margin' if model._eigval <= 0 else 'inrad(intersection)'} = {np.round(np.abs(model._eigval), 2)})"
  )

  plt.show()
