from tropy.svm import TropicalSVC, fit_tropicalized_linear_SVM
from tropy.utils import build_toy_dataset
from tropy.veronese import map_to_exponential_space
from tropy.graph import init_ax, plot_hyperplane, plot_classes, set_title, get_ignored

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATASET = 'toy_separated' # Can be: cancer, wine, toy, toy_noised, toy_separated


# Binary classification only
if __name__ == '__main__':
  L = 10  # Graph scale parameter

  # Read data
  if DATASET == 'cancer':
    base_df = pd.read_csv('./notebooks/data/breast_cancer.csv')
    X = np.log(base_df.loc[:, 'radius_mean':'fractal_dimension_worst'].to_numpy())
    y = base_df['diagnosis'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train, y_test = np.where(y_train == 'M', 1, -1), np.where(y_test == 'M', 1, -1)
    Xplus_train, Xminus_train = X_train[y_train == 1].T, X_train[y_train == -1].T
    Xplus_test, Xminus_test = X_test[y_test == 1].T, X_test[y_test == -1].T
  elif DATASET == 'wine':
    Xplus = np.log(pd.read_csv('./notebooks/data/winequality-red.csv', delimiter=';', dtype=float)
                   .loc[:, 'fixed acidity':'alcohol'].to_numpy())
    Xminus = np.log(pd.read_csv('./notebooks/data/winequality-white.csv', delimiter=';', dtype=float)
                    .loc[:, 'fixed acidity':'alcohol'].to_numpy())
    Xplus_train, Xplus_test = train_test_split(Xplus, test_size=0.2, random_state=42)
    Xminus_train, Xminus_test = train_test_split(Xminus, test_size=0.2, random_state=42)
    y_train = np.array([1] * Xplus_train.shape[0] + [-1] * Xminus_train.shape[0])
    y_test = np.array([1] * Xplus_test.shape[0] + [-1] * Xminus_test.shape[0])
    Xplus_train, Xplus_test = Xplus_train.T, Xplus_test.T 
    Xminus_train, Xminus_test = Xminus_train.T, Xminus_test.T
  elif DATASET == 'toy' or DATASET == 'toy_noised' or DATASET == 'toy_separated':
    Xplus, Xminus = build_toy_dataset(1000, 3, 2, noise=('noised' in DATASET))
    if 'separated' in DATASET:
      Xplus[2, :] -= 5
      Xminus[2, :] += 5
    Xplus_train, Xplus_test = train_test_split(Xplus.T, test_size=0.2, random_state=42)
    Xminus_train, Xminus_test = train_test_split(Xminus.T, test_size=0.2, random_state=42)
    y_train = np.array([1] * Xplus_train.shape[1] + [-1] * Xminus_train.shape[1])
    y_test = np.array([1] * Xplus_test.shape[1] + [-1] * Xminus_test.shape[1])
    Xplus_train, Xplus_test = Xplus_train.T, Xplus_test.T 
    Xminus_train, Xminus_test = Xminus_train.T, Xminus_test.T

  Xtrain, Xtest = [Xplus_train, Xminus_train], [Xplus_test, Xminus_test]

  # Tropical support vector machine
  model = TropicalSVC()
  model.fit(Xtrain, tropical_data=True)
  apex, l = model._apex, model._eigval
  tropical_accuracy = model.accuracy(Xtest)

  # Classic "tropicalized" approximation using exponential kernel
  classic_accuracies, decision_frontiers, classic_apices = [], [], []
  Beta = np.arange(1, 23, 1)
  dim = Xtrain[0].shape[0]
  for beta in tqdm(Beta):
    model, w = fit_tropicalized_linear_SVM(Xtrain, beta)

    # Compute decision frontier
    if dim == 3:
      z = lambda x,y: (-w[0]*x -w[1]*y) / w[2]
      linspace = np.linspace(-beta*L, beta*L, 100)
      xx, yy = np.meshgrid(linspace, linspace)
      zz = np.log(z(np.exp(xx), np.exp(yy)))
      decision_frontiers.append((xx/beta, yy/beta, zz/beta))

    # Compute limiting apex
    wbeta = np.abs(w)
    winf_k = 1/beta * np.log(wbeta/np.linalg.norm(wbeta)**2) * np.sign(w)
    classic_apices.append(winf_k)

    # Compute accuracy
    xtest, ytest = map_to_exponential_space(Xtest, beta)
    ypred = model.predict(xtest.T)
    classic_accuracies.append(accuracy_score(ytest, ypred))

  if dim == 3:
    fig = plt.figure()
    ax0 = init_ax(fig, 111, L, mode_3d=True)
    sur = decision_frontiers[0]
    ax0.plot_surface(sur[0], sur[1], sur[2], alpha=0.5, color="orange")
    sur_last = decision_frontiers[-1]
    ax0.plot_surface(sur_last[0], sur_last[1], sur_last[2], alpha=0.5, color="r")
    ax0.scatter([winf_k[0]], [winf_k[1]], [winf_k[2]], marker='s')
    ignored_branch = get_ignored(Xplus_test, Xminus_test, apex)
    plot_classes(ax0, [Xplus_test, Xminus_test], L)
    plot_hyperplane(ax0, apex, l, L, ignored_branch, mode_3d=True)
    set_title(ax0, "Test points and classifiers", apex, l)
    plt.show()

  """fig = plt.figure()
  ax0 = fig.add_subplot(121)
  ax0.set_title("Accuracy: tropicalization vs. native tropical")
  ax0.set_xlabel("beta")
  ax0.set_ylabel("accuracy")
  ax0.plot(Beta, classic_accuracies)
  ax0.axhline(y=tropical_accuracy, color='r', linestyle='--', label='Native tropical accuracy')
  ax0.text(0.95, tropical_accuracy, f'{100*tropical_accuracy:1f}%', va='center', ha='right', transform=ax0.get_yaxis_transform())
  ax1 = fig.add_subplot(122)
  ax1.set_title("Distance between tropicalized and native tropical apex")
  lis = [np.max(classic_apices[i] - apex) - np.min(classic_apices[i] - apex) for i in range(len(classic_apices))]
  plt.plot(Beta, lis)
  plt.show()"""