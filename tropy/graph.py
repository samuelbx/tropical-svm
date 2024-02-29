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
    
  # Automatic size for graph
  max_range = np.max(max_array - min_array)
  mid_x = (max_array[0] + min_array[0])
  mid_y = (max_array[1] + min_array[1])
  mid_z = (max_array[2] + min_array[2])
  ax.set_xlim((mid_x - max_range) / 2.5, (mid_x + max_range) / 2.5)
  ax.set_ylim((mid_y - max_range) / 2.5, (mid_y + max_range) / 2.5)
  ax.set_zlim((mid_z - max_range) / 2.5, (mid_z + max_range) / 2.5)


def init_ax(fig, config: Union[int, list[int]], L: float = 10, mode_3d: bool = False):
  """Initialize plot in the projective space R^3/(1,1,1)"""
  sns.set_context("paper")
  if type(config) == list:
    ax = fig.add_subplot(*config, projection="3d", proj_type="ortho", computed_zorder=False)
  else:
    ax = fig.add_subplot(config, projection="3d", proj_type="ortho", computed_zorder=False)
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


def plot_hyperplane_3d(ax, x: np.ndarray, margin: float, L: int, sector_indicator: np.ndarray = None) -> None:
  """Plot a tropical hyperplane as a 2D projection of a 3D hypersurface"""
  plot_polynomial_hypersurface_3d(ax, np.eye(x.shape[0]), -x, L, sector_indicator, margin = margin)


def draw_segments(ax, monomials, coeffs, nodes, i, sector_indicator=None, simplified_mode=False, margin=None):
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
        if margin is not None and not contiguous_sectors:
          draw_margin(ax, np.array([xpt, ypt, zpt]), np.array([apt, bpt, cpt]), margin)


def draw_rays(ax, monomials, coeffs, nodes, i, idx1, idx2, L, sector_indicator=None, simplified_mode=False, margin=None):
  """Draws half rays out of some node"""
  node = nodes[i]
  xpt, ypt, zpt = node[0]
  neighboring_sectors = node[1]
  a = monomials[neighboring_sectors[idx1], 0] - monomials[neighboring_sectors[idx2], 0]
  b = monomials[neighboring_sectors[idx1], 1] - monomials[neighboring_sectors[idx2], 1]
  c = monomials[neighboring_sectors[idx1], 2] - monomials[neighboring_sectors[idx2], 2]

  for sign in [-1, 1]:
    aprime, bprime = sign * L * (b - c), -sign * L * (a - c)
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
      if margin is not None and not contiguous_sectors:
        draw_margin(ax, np.array([xpt, ypt, zpt]), np.array([apt, bpt, cpt]), margin)


def evaluate_3d(monomials: np.ndarray, coeffs: np.ndarray, point: tuple[float]) -> np.ndarray:
  """Evaluates the polynomial at the given 3D point."""
  return coeffs + np.sum(monomials * np.array(point), axis=1)


def plot_polynomial_hypersurface_3d(ax, monomials, coeffs, L, sector_indicator=None, simplified_mode=False, margin=0):
  nodes = hypersurface_nodes(monomials, coeffs, 3)
  for i, node in enumerate(nodes):
    if not simplified_mode:
      plot_point(ax, node[0], 2*L, 'black', linestyle="None", marker='.', maxplus=True)
    draw_segments(ax, monomials, coeffs, nodes, i, sector_indicator=sector_indicator, simplified_mode=simplified_mode, margin=margin)
    for idx1, idx2 in [(0, 1), (0, 2), (1, 2)]:
      draw_rays(ax, monomials, coeffs, nodes, i, idx1, idx2, L, sector_indicator=sector_indicator, simplified_mode=simplified_mode, margin=margin)


def draw_margin(ax, start, end, margin, color="#FFE4C9", alpha=1):
  if np.isclose(margin, 0):
    return

  def cube_vertices(center, size):
      vertices = np.array([
          [-size, -size, -size],
          [+size, -size, -size],
          [+size, +size, -size],
          [-size, +size, -size],
          [-size, -size, +size],
          [+size, -size, +size],
          [+size, +size, +size],
          [-size, +size, +size]
      ])
      return vertices + center

  vertices1 = cube_vertices(start, margin/2)
  vertices2 = cube_vertices(end, margin/2)

  faces = [
      [vertices1[0], vertices1[1], vertices1[2], vertices1[3]],
      [vertices1[4], vertices1[5], vertices1[6], vertices1[7]],
      [vertices1[0], vertices1[1], vertices1[5], vertices1[4]],
      [vertices1[2], vertices1[3], vertices1[7], vertices1[6]],
      [vertices1[1], vertices1[2], vertices1[6], vertices1[5]],
      [vertices1[0], vertices1[3], vertices1[7], vertices1[4]],
      [vertices2[0], vertices2[1], vertices2[2], vertices2[3]],
      [vertices2[4], vertices2[5], vertices2[6], vertices2[7]],
      [vertices2[0], vertices2[1], vertices2[5], vertices2[4]],
      [vertices2[2], vertices2[3], vertices2[7], vertices2[6]],
      [vertices2[1], vertices2[2], vertices2[6], vertices2[5]],
      [vertices2[0], vertices2[3], vertices2[7], vertices2[4]],
  ]

  faces_join = [
    [vertices1[0], vertices1[1], vertices2[1], vertices2[0]],
    [vertices1[1], vertices1[2], vertices2[2], vertices2[1]],
    [vertices1[2], vertices1[3], vertices2[3], vertices2[2]],
    [vertices1[3], vertices1[0], vertices2[0], vertices2[3]],
    [vertices1[4], vertices1[5], vertices2[5], vertices2[4]],
    [vertices1[5], vertices1[6], vertices2[6], vertices2[5]],
    [vertices1[6], vertices1[7], vertices2[7], vertices2[6]],
    [vertices1[7], vertices1[4], vertices2[4], vertices2[7]],
    [vertices1[0], vertices1[4], vertices2[4], vertices2[3]],
    [vertices1[1], vertices1[5], vertices2[5], vertices2[2]],
    [vertices1[2], vertices1[6], vertices2[6], vertices2[1]],
    [vertices1[3], vertices1[7], vertices2[7], vertices2[0]],
]

  poly1 = Poly3DCollection(faces, color=color, alpha=alpha, linewidths=0)
  poly2 = Poly3DCollection(faces_join, color=color, alpha=alpha, linewidths=0)

  ax.add_collection3d(poly1)
  ax.add_collection3d(poly2)