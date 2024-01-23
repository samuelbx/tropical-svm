from typing import Union
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import matplotlib.pyplot as plt
from .utils import get_reached_sectors, max_max2_idx
from .veronese import hypersurface_nodes, newton_polynomial


def plot_point(ax, x, length, color, linestyle="-", marker=None, ignored_branch=None, maxplus=False) -> None:
  """Plot point of specified marker and associated tropical hyperplane"""
  for i in range(3):
    size = np.zeros(3)
    size[i] = length
    if i != ignored_branch:
      size *= -1 if maxplus else 1
      line_data = [(c, c + s) for c, s in zip(x, size)]
      line = Line3D(*line_data, color=color, linestyle=linestyle)
      ax.add_line(line)
  if marker:
    ax.plot(x[0], x[1], x[2], color=color, marker=marker)


def get_ignored(Cplus: np.ndarray, Cminus: np.ndarray, apex: np.ndarray) -> int:
  """For binary classification in 3D, get branches to ignore"""
  Iplus, Iminus = get_reached_sectors(Cplus, apex), get_reached_sectors(Cminus, apex)
  if len(Iplus) == 1 and len(Iminus) == 2:
    return next(iter(Iplus))
  elif len(Iminus) == 1 and len(Iplus) == 2:
    return next(iter(Iminus))
  else:
    return None


def plot_classes(ax, data_classes, L, features=None, show_lines=False):
  """Plot multiple classes of points (maximum 3 for now)"""
  colors = ['#FF934F', '#2D3142', '#058ED9', '#cc2d35']
  markers = ['o', 'v', '+', '*']
  linestyles = ['dotted', 'dashed', 'dashdot', 'dotted']
  min_array = np.array([np.inf, np.inf, np.inf])
  max_array = np.array([-np.inf, -np.inf, -np.inf])

  if features is not None:
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
  for i, clas in enumerate(data_classes):
    ls = "None" if not show_lines else linestyles[i]
    for col in clas.T:
      min_array = np.minimum(min_array, col)
      max_array = np.maximum(max_array, col)
      plot_point(ax, col, L, colors[i], ls, markers[i])
  max_range = np.max(max_array - min_array)
  mid_x = (max_array[0] + min_array[0]) / 2
  mid_y = (max_array[1] + min_array[1]) / 2
  mid_z = (max_array[2] + min_array[2]) / 2
  mean_mid = (mid_x+mid_y+mid_z)/3
  mid_x -= mean_mid
  mid_y -= mean_mid
  mid_z -= mean_mid
  ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
  ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
  ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


def plot_ball(ax, center, length):
  """Plot tropical Hilbert ball of specified center and radius"""
  half_len = length / 2
  vertices = [
      [center[0] + i * half_len, center[1] + j * half_len, center[2] + k * half_len]
      for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
  ]
  edges = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
  faces = [[vertices[i] for i in edge] for edge in edges]
  ax.add_collection3d(Poly3DCollection(faces, color="gray", alpha=0.1))


def init_ax(fig, config: Union[int, list[int]], L: float, mode_3d: bool = False):
  """Initialize plot in the projective space R^3/(1,1,1)"""
  sns.set_style("white")
  sns.set_context("notebook")
  if type(config) == list:
    ax = fig.add_subplot(*config, projection="3d", proj_type="ortho")
  else:
    ax = fig.add_subplot(config, projection="3d", proj_type="ortho")
  ax.view_init(elev=28, azim=45)
  ax.set_xlim([-L, L])
  ax.set_ylim([-L, L])
  ax.set_zlim([-L, L])
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.set_axis_off()
  if not mode_3d:
    ax.disable_mouse_rotation()
  return ax


def plot3d_hyperplane_branch(ax, axis: int, y, constants: tuple[float], ignored_branch: int, branch_index: int):
  """Create an hyperplane surface based on the given axis and constants"""
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
  """Plot a tropical hyperplane as a 2D projection or a 3D hypersurface"""
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


def draw_segments(ax, monomials, coeffs, nodes, i, lis):
  node = nodes[i]
  xpt, ypt, zpt = node[0]
  neighboring_sectors = node[1]
  for node_bis in (nodes[j] for j in range(len(nodes)) if j != i):
    neighboring_bis = node_bis[1]
    if any(sector in neighboring_bis for sector in neighboring_sectors):
      apt, bpt, cpt = node_bis[0]
      xmd, ymd, zmd = (xpt + apt) / 2, (ypt + bpt) / 2, (zpt + cpt) / 2
      val = evaluate_3d(monomials, coeffs, (xmd, ymd, zmd))
      idxes = max_max2_idx(val)
      if np.isclose(val[idxes[0]], val[idxes[1]]):
        ax.plot([xpt, apt], [ypt, bpt], [zpt, cpt], color='black', linestyle='-')
        lis[0] += 1


def draw_rays(ax, monomials, coeffs, nodes, i, idx1, idx2, lis, L):
  """Draws half rays out of some node"""
  node = nodes[i]
  xpt, ypt, zpt = node[0]
  neighboring_sectors = node[1]
  a = monomials[neighboring_sectors[idx1]][0] - monomials[neighboring_sectors[idx2]][0]
  b = monomials[neighboring_sectors[idx1]][1] - monomials[neighboring_sectors[idx2]][1]
  c = monomials[neighboring_sectors[idx1]][2] - monomials[neighboring_sectors[idx2]][2]

  for sign in [-1, 1]:
    aprime, bprime = sign * 10 * L * (b - c), -sign * 10 * L * (a - c)
    cprime = -(aprime + bprime)
    apt, bpt, cpt = xpt + aprime, ypt + bprime, zpt + cprime
    val = evaluate_3d(monomials, coeffs, (apt, bpt, cpt))
    idxes = max_max2_idx(val)
    if np.isclose(val[idxes[0]], val[idxes[1]]):
      ax.plot([xpt, apt], [ypt, bpt], [zpt, cpt], color='black', linestyle='-')
      lis[0] += 1


def evaluate_3d(monomials, coeffs, point):
    """Evaluates the polynomial at the given 3D point."""
    return coeffs + np.sum(monomials * np.array(point), axis=1)


def plot_polynomial_hypersurface_3d(ax, lattice_points, apex, L):
  monomials, coeffs = newton_polynomial(lattice_points, apex, 3)
  monomials = [np.array(elem) for elem in monomials]
  nodes = hypersurface_nodes(monomials, coeffs, 3)
  for i, node in enumerate(nodes):
    plot_point(ax, node[0], 2*L, 'black', linestyle="None", marker='.', maxplus=True)
    lis = [0]
    draw_segments(ax, monomials, coeffs, nodes, i, lis)
    for idx1, idx2 in [(0, 1), (0, 2), (1, 2)]:
      draw_rays(ax, monomials, coeffs, nodes, i, idx1, idx2, lis, L)


def set_title(ax, title: str, x: np.ndarray, l: float):
  """Helper function to set the title of the graph"""
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
