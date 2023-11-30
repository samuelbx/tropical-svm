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
               ignored_branch=None) -> None:
  x -= x.mean()
  for i in range(3):
    size = np.zeros(3)
    size[i] = length
    if i != ignored_branch:
      plot_triedra(ax, x, size, color, linestyle)
  if marker:
    ax.plot(x[0], x[1], x[2], color=color, marker=marker)


def get_sectors(C: np.ndarray, apex: np.ndarray) -> set[int]:
  I = []
  for point in C.T:
    diff = point - apex
    max = np.max(diff)
    for i, v in enumerate(diff):
      if v == max:
        I.append(i)
  return set(I)


def get_ignored(Iplus: set[int], Iminus: set[int]) -> int:
  if len(Iplus) == 1 and len(Iminus) == 2:
    return next(iter(Iplus))
  elif len(Iminus) == 1 and len(Iplus) == 2:
    return next(iter(Iminus))
  else:
    return None


def plot_points(
    ax,
    points: np.ndarray,
    length: float,
    color: str,
    linestyle: str = "dotted",
    marker: str = None,
):
  for col in points.T:
    plot_point(ax, col, length, color, linestyle, marker)


def plot_ball(ax, center, length):
  half_len = length / 2
  vertices = [
      [center[0] + i * half_len, center[1] + j * half_len, center[2] + k * half_len]
      for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
  ]
  edges = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
  faces = [[vertices[i] for i in edge] for edge in edges]
  ax.add_collection3d(Poly3DCollection(faces, color="gray", alpha=0.1))


def init_ax(fig, config: int, L: float):
  ax = fig.add_subplot(config, projection="3d", proj_type="ortho")
  ax.view_init(elev=28, azim=45)
  ax.set_xlim([-L, L])
  ax.set_ylim([-L, L])
  ax.set_zlim([-L, L])
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.disable_mouse_rotation()
  return ax


def plot_hyperplane(ax,
                    title: str,
                    Cplus: np.ndarray,
                    Cminus: np.ndarray,
                    x: np.ndarray,
                    l: float,
                    L: int,
                    no_branches: bool = False) -> None:
  ax.set_title(
      f"{title} \n (apex = {np.round(x, 2)}, margin = {np.round(np.abs(l), 2)})"
  )
  plot_ball(ax, x, l)
  Iplus, Iminus = get_sectors(Cplus, x), get_sectors(Cminus, x)
  ignored_branch = get_ignored(Iplus, Iminus)
  
  plot_points(ax, Cplus, L, "r", linestyle="dotted" if not no_branches else "", marker="o")
  plot_points(ax, Cminus, L, "b", linestyle="dashed" if not no_branches else "", marker="v")
  plot_point(ax, x, -10 * L, "black", ignored_branch=ignored_branch)


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
