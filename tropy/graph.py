import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import matplotlib.pyplot as plt


def plot_triedra(ax, center, size, color, linestyle="-") -> None:
  line_data = [(c, c + s) for c, s in zip(center, size)]
  line = Line3D(*line_data, color=color, linestyle=linestyle)
  ax.add_line(line)


def plot_point(ax,
               x,
               length,
               color,
               linestyle="-",
               marker=None,
               ignored_branch=None, maxplus=False) -> None:
  for i in range(3):
    size = np.zeros(3)
    size[i] = length
    if i != ignored_branch:
      plot_triedra(ax, x, ((-1) if maxplus else 1) * size, color, linestyle)
  if marker:
    ax.plot(x[0], x[1], x[2], color=color, marker=marker)


# TODO: add parameter majoritary or strict
def get_reached_sectors(C: np.ndarray, apex: np.ndarray) -> set[int]:
  I = []
  for point in C.T:
    diff = point - apex
    max = np.max(diff)
    for i, v in enumerate(diff):
      if v == max:
        I.append(i)
  return set(I)


def get_ignored(Cplus: np.ndarray, Cminus: np.ndarray, apex: np.ndarray) -> int:
  Iplus, Iminus = get_reached_sectors(Cplus, apex), get_reached_sectors(Cminus, apex)
  if len(Iplus) == 1 and len(Iminus) == 2:
    return next(iter(Iplus))
  elif len(Iminus) == 1 and len(Iplus) == 2:
    return next(iter(Iminus))
  else:
    return None


def plot_class(
    ax,
    points: np.ndarray,
    length: float,
    color: str,
    linestyle: str = "dotted",
    marker: str = None,
):
  for col in points.T:
    plot_point(ax, col, length, color, linestyle, marker)


def plot_classes(ax, data_classes, L, features=None, show_lines=False):
  colors = ['#FF934F', '#2D3142', '#058ED9']
  markers = ['o', 'v', '+']
  linestyles = ['dotted', 'dashed', 'dashdot']
  if features is not None:
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
  for i, clas in enumerate(data_classes):
    ls = "None" if not show_lines else linestyles[i]
    plot_class(ax, clas, L, colors[i], linestyle=ls, marker=markers[i])


def plot_ball(ax, center, length):
  half_len = length / 2
  vertices = [
      [center[0] + i * half_len, center[1] + j * half_len, center[2] + k * half_len]
      for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
  ]
  edges = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
  faces = [[vertices[i] for i in edge] for edge in edges]
  ax.add_collection3d(Poly3DCollection(faces, color="gray", alpha=0.1))


def init_ax(fig, config: int, L: float, mode_3d: bool = False):
  ax = fig.add_subplot(config, projection="3d", proj_type="ortho")
  ax.view_init(elev=28, azim=45)
  ax.set_xlim([-L, L])
  ax.set_ylim([-L, L])
  ax.set_zlim([-L, L])
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  if not mode_3d:
    ax.disable_mouse_rotation()
  return ax



def plot3d_hyperplane_branch(ax, axis: int, y, constants: tuple[float], ignored_branch: int, branch_index: int):
  # Create an hyperplane surface based on the given axis and constants.
  if ignored_branch != branch_index:
    dim1, dim2 = np.meshgrid(y, y)
    if axis == 0:  # X
      dim3 = np.where(dim2 - constants[2] >= dim1 - constants[0], constants[1] + dim2 - constants[2], np.nan)
    elif axis == 1:  # Y
      dim3 = np.where(dim1 - constants[0] >= dim2 - constants[1], constants[2] + dim1 - constants[0], np.nan)
    else:  # Z
      dim3 = np.where(dim2 - constants[1] >= dim1 - constants[2], constants[0] + dim2 - constants[1], np.nan)
    
    if axis == 0:
      return ax.plot_surface(dim1, dim3, dim2, color='black', alpha=0.3)
    elif axis == 1:
      return ax.plot_surface(dim3, dim1, dim2, color='black', alpha=0.3)
    else:
      return ax.plot_surface(dim1, dim2, dim3, color='black', alpha=0.3)
  
  return None


def plot_hyperplane(ax, x: np.ndarray, l: int, L: int, ignored_branch: int = None, mode_3d: bool = False) -> None:
  l = np.abs(l)
  if l > 0:
    plot_ball(ax, x, l)

  if mode_3d:
    y = np.linspace(-L, L, 1000)
    a, b, c = x
    surfaces = []
    for i in range(3):
      surfaces.append(plot3d_hyperplane_branch(ax, i, y, (a, b, c), ignored_branch, i))
    return surfaces
  else:
    plot_point(ax, x, -10 * L, "black", ignored_branch=ignored_branch)


def set_title(ax, title: str, x: np.ndarray, l: float):
  ax.set_title(
    f"{title} \n (apex = {np.round(x, 2)}, {'margin' if l <= 0 else 'inrad(intersection)'} = {np.round(np.abs(l), 2)})"
  )


# TODO: Make better & handle multi-class
def plot_confusion_matrix(conf_matrix: np.ndarray) -> None:
  plt.figure(figsize=(6, 6))
  plt.imshow(conf_matrix, cmap='Blues', interpolation='None', vmin=0)
  plt.colorbar()
  plt.xticks([0, 1], ['Predicted Positive', 'Predicted Negative'])
  plt.yticks([0, 1], ['Actual Positive', 'Actual Negative'])

  for i in range(2):
    for j in range(2):
      plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='Black')

  plt.title('Confusion Matrix')
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.show()
