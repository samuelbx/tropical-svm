from tropy.learn import fit_classifier, _inrad_eigenpair
from tropy.metrics import accuracy_multiple
from tropy.utils import build_toy_dataset
from tropy.graph import init_ax, plot_hyperplane
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


DATASET = 'toy_noised' # Can be: cancer, wine, toy, toy_noised


def tropical_kernel(Xlist: np.ndarray, beta: float):
  Xplus, Xminus = Xlist
  xplus, xminus = np.exp(beta * Xplus), np.exp(beta * Xminus)
  x = np.concatenate((xplus, xminus), axis=1)
  y = np.array([1] * xplus.shape[1] + [-1] * xminus.shape[1])
  return x, y


# Binary classification only
if __name__ == '__main__':

  L = 10

  # Read data
  if DATASET == 'cancer':
    base_df = pd.read_csv('./data/breast_cancer.csv')
    X = np.log(base_df.loc[:, 'radius_mean':'fractal_dimension_worst'].to_numpy())
    y = base_df['diagnosis'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = np.where(y_train == 'M', 1, -1)
    y_test = np.where(y_test == 'M', 1, -1)
    Xplus_train, Xminus_train = X_train[y_train == 1].T, X_train[y_train == -1].T
    Xplus_test, Xminus_test = X_test[y_test == 1].T, X_test[y_test == -1].T
  elif DATASET == 'wine':
    Xplus = np.log(pd.read_csv('./data/winequality-red.csv', delimiter=';', dtype=float).loc[:, 'fixed acidity':'alcohol'].to_numpy())
    Xminus = np.log(pd.read_csv('./data/winequality-white.csv', delimiter=';', dtype=float).loc[:, 'fixed acidity':'alcohol'].to_numpy())
    Xplus_train, Xplus_test = train_test_split(Xplus, test_size=0.2, random_state=42)
    Xminus_train, Xminus_test = train_test_split(Xminus, test_size=0.2, random_state=42)
    y_train = np.array([1] * Xplus_train.shape[0] + [-1] * Xminus_train.shape[0])
    y_test = np.array([1] * Xplus_test.shape[0] + [-1] * Xminus_test.shape[0])
    Xplus_train, Xplus_test = Xplus_train.T, Xplus_test.T 
    Xminus_train, Xminus_test = Xminus_train.T, Xminus_test.T
  elif DATASET == 'toy' or DATASET == 'toy_noised':
    Xplus, Xminus = build_toy_dataset(1000, 3, 2, noise=('noised' in DATASET))
    Xplus_train, Xplus_test = train_test_split(Xplus.T, test_size=0.2, random_state=42)
    Xminus_train, Xminus_test = train_test_split(Xminus.T, test_size=0.2, random_state=42)
    y_train = np.array([1] * Xplus_train.shape[1] + [-1] * Xminus_train.shape[1])
    y_test = np.array([1] * Xplus_test.shape[1] + [-1] * Xminus_test.shape[1])
    Xplus_train, Xplus_test = Xplus_train.T, Xplus_test.T 
    Xminus_train, Xminus_test = Xminus_train.T, Xminus_test.T

  Xtrain, Xtest = [Xplus_train, Xminus_train], [Xplus_test, Xminus_test]

  # Tropical support vector machine
  apex, l = _inrad_eigenpair(Xtrain)
  predictor = fit_classifier(Xtrain, apex)[0]
  tropical_accuracy = accuracy_multiple(predictor, Xtest)

  # Classic "tropicalized" approximation using exponential kernel
  classic_accuracies = []
  separation_surfaces = []
  Beta = [1, 2, 5, 10, 12, 15]
  for beta in tqdm(Beta):
    xtrain, ytrain = tropical_kernel(Xtrain, beta)
    xtest, ytest = tropical_kernel(Xtest, beta)
    model = LinearSVC(dual=True, fit_intercept=False)
    clf = model.fit(xtrain.T, ytrain)
    z = lambda x,y: (-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
    datarange = np.linspace(-beta*L, beta*L, 100)
    xx, yy = np.meshgrid(datarange, datarange)
    Z = np.log(z(np.exp(xx), np.exp(yy)))/beta
    separation_surfaces.append((xx/beta, yy/beta, Z))

    ypred = model.predict(xtest.T)
    classic_accuracies.append(accuracy_score(ytest, ypred))

  print(f'Tropical accuracy: {tropical_accuracy}')
  print(f'Classic accuracies: {classic_accuracies}')

  fig = plt.figure()
  ax0 = init_ax(fig, 111, L, mode_3d=True)

  h2, h3 = plot_hyperplane(ax0, "native tropical", Xplus_test, Xminus_test, apex, l, L, no_branches=True, mode_3d=True)
  idx = -2
  sur = separation_surfaces[idx]
  mask = (sur[0] >= -30) & (sur[1] >= -30) & (sur[2] >= -30)
  hp = None
  
  switch = False

  def animate(i):
    global switch, idx, hp, h2, h3, ax0
    ax0.view_init(elev=28, azim=45+i//2)
    if i % 60 == 0:
      h2.set_visible(switch)
      h3.set_visible(switch)
      if not switch:
        idx = (idx+1) % len(Beta)
        sur = separation_surfaces[idx]
        hp = ax0.plot_surface(sur[0], sur[1], sur[2], alpha=0.5, color='orange')
        ax0.set_title(f"linear SVM, exp kernel (beta={Beta[idx]}), test points")
      else:
        hp.remove()
        ax0.set_title(f"tropical SVM, test points")
      switch = not switch
    return fig, h2, h3, hp,

  anim = animation.FuncAnimation(fig, animate, frames=tqdm(range(720)), interval=50, blit=False)
  anim.save('animation.gif', writer='imagemagick', fps=20)
  plt.show()