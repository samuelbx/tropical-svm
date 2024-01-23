from tropy.graph import init_ax, plot_classes, plot_polynomial_hypersurface_3d
from tropy.svm import TropicalSVM
from tropy.utils import apply_noise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

save = False  # Save the figure (.PGF) or show it
size = 3  # Dilatation factor of 2-simplex
grid_mode = True
DATASET = 'toy-centers'

# Possible values for previous parameters
possible_datasets = 'iris, toy-(centers)-(reverse)'

def toy_gaussian(center: list):
    return apply_noise(np.array([center] * 10, dtype=float).T, mu=0.05)

centers_max_plus = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
centers_min_plus = [[0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
centers = centers_max_plus if not 'reverse' in DATASET else centers_min_plus
if 'toy' in DATASET:
  Clist_train = [np.array([c]).T if 'centers' in DATASET else toy_gaussian(c) for c in centers]
elif 'iris' in DATASET:
  base_df = pd.read_csv('./notebooks/data/IRIS.csv')
  df = base_df.loc[:, 'sepal_length':'petal_length']
  features = df.columns.to_list()
  classes = ["Iris-setosa", "Iris-versicolor"]

  def class_df(class_name):
    df_class = df[base_df["species"].str.contains(class_name)]
    X = df_class.to_numpy(dtype=float).T
    return X
  
  data_classes = []
  for class_name in classes:
    train = class_df(class_name, size)
    data_classes.append(train)
else:
  raise ValueError(f'Choose a dataset from: {possible_datasets}')

if not grid_mode:
  model = TropicalSVM()
  model.fit(Clist_train, size)

  fig = plt.figure(figsize=(12,12))
  ax = init_ax(fig, 111, L=10)
  plot_classes(ax, Clist_train, L=10, features=["x", "y", "z"])
  plot_polynomial_hypersurface_3d(ax, model.veronese_coefficients, model.apex, L=10)

else:
  fig = plt.figure(figsize=(24, 24))
  for size in tqdm(range(1, 17), "Building grid"):
    model = TropicalSVM()
    model.fit(Clist_train, size)

    ax = init_ax(fig, [4, 4, size], L=10)
    plot_classes(ax, Clist_train, L=10, features=["x", "y", "z"])
    plot_polynomial_hypersurface_3d(ax, model.veronese_coefficients, model.apex, L=25)
    ax.set_title(f'Size: {size}')

if save:
  plt.savefig(f'A_{size}.pgf')
else:
  plt.show()