#!/usr/bin/env python3

import argparse
import sys
from tropy.graph import init_ax, plot_classes, plot_polynomial_hypersurface_3d, plot_ball
from tropy.svm import TropicalSVC
from tropy.utils import apply_noise, build_toy_dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(args):
  save = args.save
  degree = args.degree
  DATASET = args.dataset
  log_linear_beta = args.beta
  show_ball = args.show_ball
  grid_mode = args.grid_mode

  def toy_gaussian(center: list):
    return apply_noise(np.array([center] * 10, dtype=float).T, mu=0.3, seed=None)

  centers_max_plus = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
  centers_min_plus = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
  centers = centers_max_plus if not 'reverse' in DATASET else centers_min_plus
  if 'bintoy' in DATASET:
    native_tropical = True
    Xplus, Xminus = build_toy_dataset(80, 3, 2, noise=True)
    if 'separated' in DATASET:
      Xplus[2, :] -= 5
      Xminus[2, :] += 5
    elif 'mixed' in DATASET:
      Xplus[2, :] += 3
      Xminus[2, :] -= 3
    apply_noise(Xplus, seed=2024, mu=0.2)
    apply_noise(Xminus, seed=2024, mu=0.2)
    data_classes = [Xplus, Xminus]
  if 'circular' in DATASET:
    native_tropical = True
    C = np.random.normal([[0, 0, 0]]*100, 10, (100, 3)).T
    positive_mask = np.max(C, axis=0) - np.min(C, axis=0) > 15
    negative_mask = ~positive_mask
    C[:, negative_mask] /= 3
    Xplus = C[:, positive_mask]
    Xminus = C[:, negative_mask]
    data_classes = [Xplus, Xminus]
  elif 'toy' in DATASET:
    native_tropical = True
    data_classes = [np.array([c]).T if 'centers' in DATASET else toy_gaussian(c) for c in centers]
  elif 'iris' in DATASET:
    native_tropical = False
    base_df = pd.read_csv('./notebooks/data/IRIS.csv')
    df = base_df.loc[:, ['sepal_width', 'petal_width']]
    classes = ["Iris-virginica", "Iris-versicolor"]
    if not 'binary' in DATASET:
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

  if not grid_mode:
    model = TropicalSVC()
    model.fit(data_classes, degree, tropical_data=native_tropical, log_linear_beta=log_linear_beta)

    fig = plt.figure(figsize=(9,9) if not save else (6, 6))
    ax = init_ax(fig, 111, L=10)
    if log_linear_beta is not None:
      method = f'linear SVM on log paper, $\\beta = {log_linear_beta}$'
    else:
      method = 'mean payoff games'
    if show_ball and degree == 1 and model._eigval < 0:
      plot_ball(ax, model._apex, np.abs(model._eigval))
    if not save:
      ax.set_title(f'$deg = {degree}$, using {method}', fontsize='small', loc='left')
    plot_classes(ax, model._data_classes, L=10, show_balls=(0 if degree==1 or show_ball == False or model._eigval >= 0 else np.abs(model._eigval)/degree))
    plot_polynomial_hypersurface_3d(ax, model._monomials, model._coeffs, L=10, sector_indicator=model._sector_indicator)

  else:
    fig = plt.figure(figsize=(9, 9))
    for degree in tqdm(range(1, 5), "Building grid"):
      model = TropicalSVC()
      data = data_classes.copy()
      model.fit(data, degree, tropical_data=native_tropical, log_linear_beta=log_linear_beta)

      ax = init_ax(fig, [2, 2, degree], L=10)
      plot_classes(ax, model._data_classes, L=10, show_balls=(0 if degree==1 or show_ball == False or model._eigval >= 0 else np.abs(model._eigval)/degree))
      plot_polynomial_hypersurface_3d(ax, model._monomials, model._coeffs, L=10, sector_indicator=model._sector_indicator)
      ax.set_title(f'$deg = {degree}$', fontsize='small', loc='left')
      if show_ball and degree == 1 and model._eigval < 0:
        plot_ball(ax, model._apex, np.abs(model._eigval))

  if save:
    plt.savefig(f'{DATASET}_{degree}.pgf')

  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Fitting and plotting tropical piecewise linear classifiers on 3D datasets")
  parser.add_argument("dataset", type=str, choices=['iris', 'iris-binary', 'toy', 'toy-centers', 'toy-reverse', 'toy-centers-reverse', 'bintoy', 'bintoy-separated', 'bintoy-mixed', 'circular'], help="Dataset to classify")
  parser.add_argument("degree", nargs='?', type=int, default=1, help="Degree of tropical polynomial")
  parser.add_argument("-s", "--save", action="store_true", help="Save the figure (.PGF)")
  parser.add_argument("--beta", type=float, default=None, help="If specified, Beta value for using 'linear SVM on log paper' trick")
  parser.add_argument("--show-ball", action="store_true", help="Show the Hilbert ball corresponding to inner radius or margin")
  parser.add_argument("--grid-mode", action="store_true", help="Show a grid of multiple tropical polynomials")

  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

  args = parser.parse_args()
  main(args)
