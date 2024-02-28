import numpy as np


def veronese_feature_names(feature_names: list[str], lattice_points: list[np.ndarray]):
  ext_feature_names = []
  for comb, vals in lattice_points:
    subfeatures = []
    for c, v in zip(comb, vals):
      feature = feature_names[c]
      subfeatures.append(feature if v == 1 else f"{v}*{feature}")
    ext_feature_names.append(" + ".join(subfeatures))
  return ext_feature_names


def print_features_per_class(class_names: list[str], feature_names: list[str],
                             sector_indicator: list[int]) -> None:
  features = {cl: [] for cl in class_names}
  for id, feature in zip(sector_indicator, feature_names):
    if id != -1:
      features[class_names[id]].append(feature)
  print("Dominant features for each class:")
  for k, v in features.items():
    print(f'- {k}: {", ".join(v)}')
