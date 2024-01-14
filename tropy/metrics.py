import numpy as np
from sklearn.metrics import accuracy_score


# TODO: Make one main function to have model profile & accuracies & etc


# TODO: Make cleaner, no sklearn
def accuracy_multiple(pred: callable, Clist: list[np.ndarray]) -> float:
  true_labels = []
  predicted_labels = []
  for label, points in enumerate(Clist):
    true_labels.extend([label] * points.shape[1])
    predicted_labels.extend(pred(points))
  return accuracy_score(true_labels, predicted_labels)


def veronese_feature_names(feature_names: list[str],
                           lattice_points: list[np.ndarray]):
  ext_feature_names = []
  for comb, vals in lattice_points:
    subfeatures = []
    for c, v in zip(comb, vals):
      feature = feature_names[c]
      subfeatures.append(feature if v == 1 else f"{v}*{feature}")
    ext_feature_names.append(" + ".join(subfeatures))
  return ext_feature_names


def print_features_per_class(class_names: list[str],
                             feature_names: list[str],
                             sector_indicator: list[int]):
  features = {cl: [] for cl in class_names}
  for id, feature in zip(sector_indicator, feature_names):
    if id != -1:
      features[class_names[id]].append(feature)
  print("Dominant features for each class:")
  for k, v in features.items():
    print(f'- {k}: {", ".join(v)}')
