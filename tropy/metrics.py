import numpy as np
from sklearn.metrics import accuracy_score


def accuracy_precision_recall_confusion(
    pred: callable, Ptest: np.ndarray,
    Ntest: np.ndarray) -> tuple[float, float, float, np.ndarray]:
  TP, TN = 0, 0
  for row in Ptest.T:
    TP += pred(row)
  for row in Ntest.T:
    TN += 1 - pred(row)
  FP, FN = len(Ptest.T) - TP, len(Ntest.T) - TN
  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = TP / (TP + FP) if TP + FP != 0 else 0
  recall = TP / (TP + FN) if TP + FN != 0 else 0
  conf_matrix = np.array([[TP, FP], [FN, TN]])
  return accuracy, precision, recall, conf_matrix


# TODO: Make cleaner, no sklearn
def accuracy_multiple(pred: callable, Clist: list[np.ndarray]) -> float:
  true_labels = []
  predicted_labels = []
  for label, points in enumerate(Clist):
    true_labels.extend([label] * points.shape[1])
    predicted_labels.extend([pred(point) for point in points.T])
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
