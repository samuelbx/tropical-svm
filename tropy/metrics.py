import numpy as np


def veronese_feature_names(feature_names: list[str], monomials: list[np.ndarray]):
  ext_feature_names = []
  for monomial in monomials:
    subfeatures = []
    for i, coef in enumerate(monomial):
      feature = feature_names[i]
      if coef != 0:
        subfeatures.append(feature if coef == 1 else f"{int(coef)}*{feature}")
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
