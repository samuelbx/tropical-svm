from typing import Union
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
import matplotlib.pyplot as plt
from .utils import max_max2_idx
from .veronese import hypersurface_nodes


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


def plot_classes(ax, data_classes, L, features=None, show_lines=False, balls_radius: float =None):
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

  if balls_radius is not None:
    for i, clas in enumerate(data_classes):
      for col in clas.T:
        plot_ball(ax, col, balls_radius, colors[i], alpha=.05)
  
  for i, clas in enumerate(data_classes):
    ls = "None" if not show_lines else linestyles[i]
    for col in clas.T:
      min_array = np.minimum(min_array, col)
      max_array = np.maximum(max_array, col)
      plot_point(ax, col, L, colors[i], ls, markers[i])
    
  # Automatic size for graph
  max_range = np.max(max_array - min_array)
  mid_x = (max_array[0] + min_array[0])
  mid_y = (max_array[1] + min_array[1])
  mid_z = (max_array[2] + min_array[2])
  ax.set_xlim((mid_x - max_range) / 2, (mid_x + max_range) / 2)
  ax.set_ylim((mid_y - max_range) / 2, (mid_y + max_range) / 2)
  ax.set_zlim((mid_z - max_range) / 2, (mid_z + max_range) / 2)


def plot_ball(ax, center, length, color="gray", alpha=.1):
  """Plot tropical Hilbert ball of specified center and radius"""
  half_len = length / 2
  vertices = [
      [center[0] + i * half_len, center[1] + j * half_len, center[2] + k * half_len]
      for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]
  ]
  edges = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
  faces = [[vertices[i] for i in edge] for edge in edges]
  ax.add_collection3d(Poly3DCollection(faces, color=color, alpha=alpha, linewidths=0))


def init_ax(fig, config: Union[int, list[int]], L: float = 10, mode_3d: bool = False):
  """Initialize plot in the projective space R^3/(1,1,1)"""
  sns.set_context("paper")
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


def plot_hyperplane_3d(ax, x: np.ndarray, l: int, L: int, sector_indicator: np.ndarray = None) -> None:
  """Plot a tropical hyperplane as a 2D projection of a 3D hypersurface"""
  l = np.abs(l)
  if l > 0:
    plot_ball(ax, x, l)
  plot_polynomial_hypersurface_3d(ax, np.eye(x.shape[0]), -x, L, sector_indicator)


def draw_segments(ax, monomials, coeffs, nodes, i, sector_indicator=None, simplified_mode=False):
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
        linestyle, color = '-', 'black'
        contiguous_sectors = sector_indicator is not None and sector_indicator[idxes[0]] == sector_indicator[idxes[1]]
        if contiguous_sectors:
          linestyle, color = 'dotted', 'lightgray'
          if simplified_mode:
            break
        ax.plot([xpt, apt], [ypt, bpt], [zpt, cpt], color=color, linestyle=linestyle)


def draw_rays(ax, monomials, coeffs, nodes, i, idx1, idx2, L, sector_indicator=None, simplified_mode=False):
  """Draws half rays out of some node"""
  node = nodes[i]
  xpt, ypt, zpt = node[0]
  neighboring_sectors = node[1]
  a = monomials[neighboring_sectors[idx1], 0] - monomials[neighboring_sectors[idx2], 0]
  b = monomials[neighboring_sectors[idx1], 1] - monomials[neighboring_sectors[idx2], 1]
  c = monomials[neighboring_sectors[idx1], 2] - monomials[neighboring_sectors[idx2], 2]

  for sign in [-1, 1]:
    aprime, bprime = sign * 10 * L * (b - c), -sign * 10 * L * (a - c)
    cprime = -(aprime + bprime)
    apt, bpt, cpt = xpt + aprime, ypt + bprime, zpt + cprime
    val = evaluate_3d(monomials, coeffs, (apt, bpt, cpt))
    idxes = max_max2_idx(val)
    if np.isclose(val[idxes[0]], val[idxes[1]]):
      linestyle, color = '-', 'black'
      contiguous_sectors = sector_indicator is not None and sector_indicator[idxes[0]] == sector_indicator[idxes[1]]
      if contiguous_sectors:
        linestyle, color = 'dotted', 'lightgray'
        if simplified_mode:
          break
      ax.plot([xpt, apt], [ypt, bpt], [zpt, cpt], color=color, linestyle=linestyle)


def evaluate_3d(monomials: np.ndarray, coeffs: np.ndarray, point: tuple[float]) -> np.ndarray:
  """Evaluates the polynomial at the given 3D point."""
  return coeffs + np.sum(monomials * np.array(point), axis=1)


def plot_polynomial_hypersurface_3d(ax, monomials, coeffs, L, sector_indicator=None, simplified_mode=False):
  nodes = hypersurface_nodes(monomials, coeffs, 3)
  for i, node in enumerate(nodes):
    if not simplified_mode:
      plot_point(ax, node[0], 2*L, 'black', linestyle="None", marker='.', maxplus=True)
    draw_segments(ax, monomials, coeffs, nodes, i, sector_indicator=sector_indicator, simplified_mode=simplified_mode)
    for idx1, idx2 in [(0, 1), (0, 2), (1, 2)]:
      draw_rays(ax, monomials, coeffs, nodes, i, idx1, idx2, L, sector_indicator=sector_indicator, simplified_mode=simplified_mode)
