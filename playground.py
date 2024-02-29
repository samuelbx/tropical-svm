#!/usr/bin/env python3

import argparse
import sys
from tropy.graph import init_ax, plot_classes, plot_polynomial_hypersurface_3d
from tropy.svm import TropicalSVC
from tropy.utils import apply_noise, build_toy_tropical_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


def toy_gaussian(center: list) -> np.ndarray:
  return apply_noise(np.array([center] * 10, dtype=float).T, mu=0.1, seed=42)


def generate_toy_data(dataset: str) -> tuple[list[np.ndarray], bool]:
  centers_max_plus = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
  centers_min_plus = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
  centers = centers_max_plus if not 'reverse' in dataset else centers_min_plus
  np.random.seed(42)
  if 'bintoy' in dataset:
    native_tropical = True
    Xplus, Xminus = build_toy_tropical_data(80, 3, 2)
    if 'separated' in dataset:
      Xplus[2, :] -= 5
      Xminus[2, :] += 5
    elif 'mixed' in dataset:
      Xplus[2, :] += 3
      Xminus[2, :] -= 3
    apply_noise(Xplus, seed=2024, mu=0.2)
    apply_noise(Xminus, seed=2024, mu=0.2)
    data_classes = [Xplus, Xminus]
  elif dataset == 'moons':
    native_tropical = False
    data, labels = make_moons(noise=0.2, random_state=1)
    Xminus, Xplus = data[labels == 0], data[labels == 1]
    data_classes = [Xplus.T, Xminus.T]
  elif dataset == 'circular':
    native_tropical = True
    C = np.random.normal([[0, 0, 0]]*100, 10, (100, 3)).T
    positive_mask = np.max(C, axis=0) - np.min(C, axis=0) > 15
    negative_mask = ~positive_mask
    C[:, negative_mask] /= 3
    Xplus = C[:, positive_mask]
    Xminus = C[:, negative_mask]
    data_classes = [Xplus, Xminus]
  elif 'toy' in dataset:
    native_tropical = True
    data_classes = [np.array([c]).T if 'centers' in dataset else toy_gaussian(c) for c in centers]
  elif 'iris' in dataset:
    native_tropical = False
    base_df = pd.read_csv('./data/iris.csv')
    df = base_df.loc[:, ['sepal_width', 'petal_width']]
    classes = ["Iris-virginica", "Iris-versicolor"]
    if not 'binary' in dataset:
      classes.insert(0, "Iris-setosa")

    def class_df(class_name):
      df_class = df[base_df["species"].str.contains(class_name)]
      X = df_class.to_numpy(dtype=float).T
      return X
    
    data_classes = []
    for class_name in classes:
      train = class_df(class_name)
      data_classes.append(train)
  else:
    raise ValueError(f'Choose a possible dataset')

  return data_classes, native_tropical


def main(args):
  save = args.save
  degree = args.degree
  dataset = args.dataset
  log_linear_beta = args.beta
  simplified = args.simplified
  feature_selection = args.feature_selection

  data_classes, native_tropical = generate_toy_data(dataset)

  model = TropicalSVC()
  model.fit(data_classes, degree, native_tropical_data=native_tropical, log_linear_beta=log_linear_beta, feature_selection=feature_selection)

  fig = plt.figure(figsize=(9,9) if not save else (6, 6))
  ax = init_ax(fig, 111, L=10, mode_3d=False)
  if log_linear_beta is not None:
    method = f'linear SVM on log paper, $\\beta = {log_linear_beta}$'
  else:
    method = 'mean payoff games'
  if feature_selection is not None:
    features = f'(experimental) feature selection, {feature_selection} points per class'
  else:
    features = f'$deg = {degree}$'
  if not save:
    ax.set_title(f'{features}, using {method}', fontsize='small', loc='left')
  plot_classes(ax, model._data_classes, L=10)
  plot_polynomial_hypersurface_3d(ax, model._monomials, model._coeffs, L=10, sector_indicator=model._sector_indicator, simplified_mode=simplified, margin=(model.margin() if log_linear_beta is None else 0))

  if save:
    plt.savefig(f'{dataset}_{degree}.svg')

  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Fitting and plotting tropical piecewise linear classifiers on 3D datasets")
  parser.add_argument("dataset", type=str, choices=['iris', 'iris-binary', 'moons', 'toy', 'toy-centers', 'toy-reverse', 'toy-centers-reverse', 'bintoy', 'bintoy-separated', 'bintoy-mixed', 'circular'], help="Dataset to classify")
  parser.add_argument("degree", nargs='?', type=int, default=1, help="Degree of tropical polynomial")
  parser.add_argument("-s", "--save", action="store_true", help="Save the figure (.PGF)")
  parser.add_argument("--beta", type=float, default=None, help="If specified, Beta value for using 'linear SVM on log paper' trick")
  parser.add_argument("--simplified", action="store_true", help="Provide a simplified view of the hypersurface, with the decision boundary only")
  parser.add_argument("--feature-selection", type=int, help="Experimental: heuristic to generate more relevant monomials based on data. Specify the number of points to sample per class if wanted. Bypasses degree option.")

  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

  args = parser.parse_args()
  main(args)
