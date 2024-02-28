from tropy.svm import TropicalSVC, fit_tropicalized_linear_SVM
from tropy.utils import build_toy_tropical_data, apply_noise
from tropy.graph import init_ax, plot_hyperplane_3d, plot_classes

import matplotlib.pyplot as plt
import numpy as np


# Binary classification only
if __name__ == '__main__':
  L = 10  # Graph scale parameter

  Xplus, Xminus = build_toy_tropical_data(80, 3, 2)
  Xplus = apply_noise(Xplus, 1)
  Xminus = apply_noise(Xminus, 1)
  Xplus[2, :] -= 5
  Xminus[2, :] += 5
  Xtrain = [Xplus, Xminus]

  # Tropical support vector machine
  model = TropicalSVC()
  model.fit(Xtrain, native_tropical_data=True)
  apex, l = model._apex, model._eigval

  # Classic "tropicalized" approximation using exponential kernel
  decision_frontiers = []
  dim = Xtrain[0].shape[0]
  for beta in [0.3, 20]:
    model, w = fit_tropicalized_linear_SVM(Xtrain, beta)

    # Compute decision frontier
    if dim == 3:
      z = lambda x,y: (-w[0]*x -w[1]*y) / w[2]
      linspace = np.linspace(-beta*L, beta*L, 19)
      xx, yy = np.meshgrid(linspace, linspace)
      zz = np.log(z(np.exp(xx), np.exp(yy)))
      decision_frontiers.append((xx/beta, yy/beta, zz/beta))

  if dim == 3:
    fig = plt.figure(figsize=(9,9))
    ax = init_ax(fig, 111, L, mode_3d=False)
    ax.view_init(elev=29, azim=45)
    sur = decision_frontiers[0]
    ax.plot_surface(sur[0], sur[1], sur[2], alpha=0.8, color="orange")
    sur_last = decision_frontiers[-1]
    ax.plot_surface(sur_last[0], sur_last[1], sur_last[2], alpha=0.8, color="r")
    plot_classes(ax, [Xplus, Xminus], L)
    plot_hyperplane_3d(ax, apex, l, L)
    plt.show()