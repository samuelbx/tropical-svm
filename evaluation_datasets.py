import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_diabetes
from sklearn.datasets import fetch_openml
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
import os
import psutil
import warnings
warnings.filterwarnings('ignore')

from tropy.svm import TropicalSVC, get_km_time, get_km_iter

def get_memory_usage():
  process = psutil.Process(os.getpid())
  return process.memory_info().rss / (1024 * 1024)

def format_data_for_tropical(X, y):
  classes = np.unique(y)
  data_classes = []
  
  for c in classes:
    class_data = X[y == c].T
    data_classes.append(class_data)
  
  return data_classes

def compare_models(X, y, feature_selection=None, cv_folds=5, random_state=42):
  results = {
    'tropical_accuracies': [],
    'linear_accuracies': [],
    'tropical_times': [],
    'tropical_iters': [],
    'linear_times': [],
    'tropical_n_monomials': [],
    'spectral_radiuses': []
  }
  
  cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=1, random_state=random_state)
  
  for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
    print(f"Fold {fold+1}/{cv_folds}")
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    linear_start_time = time.time()
    linear_model = LinearSVC(dual='auto', random_state=random_state)
    linear_model.fit(X_train_scaled, y_train)
    linear_time = time.time() - linear_start_time
    
    linear_preds = linear_model.predict(X_test_scaled)
    linear_acc = accuracy_score(y_test, linear_preds)
    
    train_classes = format_data_for_tropical(X_train_scaled, y_train)
    
    tropical_model = TropicalSVC()
    tropical_model.fit(train_classes, poly_degree=1, native_tropical_data=False, 
              feature_selection=feature_selection)
    tropical_time = get_km_time()
    tropical_iter = get_km_iter()

    n_monomials = len(tropical_model._monomials) if hasattr(tropical_model, '_monomials') else 0
    spectral_radius = getattr(tropical_model, '_eigval', None)
    
    tropical_preds = tropical_model.predict(X_test_scaled.T)
    tropical_acc = accuracy_score(y_test, tropical_preds)
    
    print(f"  Accuracy: {tropical_acc:.4f}, Time: {tropical_time:.4f}s, Monomials: {n_monomials}")
    
    results['tropical_accuracies'].append(tropical_acc)
    results['linear_accuracies'].append(linear_acc)
    results['tropical_times'].append(tropical_time)
    results['tropical_iters'].append(tropical_iter)
    results['linear_times'].append(linear_time)
    results['tropical_n_monomials'].append(n_monomials)
    results['spectral_radiuses'].append(spectral_radius)
    
    print(f"  Tropical SVM: Acc={tropical_acc:.4f}, Time={tropical_time:.4f}s")
    print(f"  Linear SVC:   Acc={linear_acc:.4f}, Time={linear_time:.4f}s")
      
  if results['tropical_accuracies']:
    n_folds = len(results['tropical_accuracies'])
    summary = {
      'tropical_mean_acc': np.mean(results['tropical_accuracies']),
      'tropical_std_acc': np.std(results['tropical_accuracies']),
      'tropical_ci_lower': np.mean(results['tropical_accuracies']) - 1.96 * np.std(results['tropical_accuracies']) / np.sqrt(n_folds),
      'tropical_ci_upper': np.mean(results['tropical_accuracies']) + 1.96 * np.std(results['tropical_accuracies']) / np.sqrt(n_folds),
      'linear_mean_acc': np.mean(results['linear_accuracies']),
      'linear_std_acc': np.std(results['linear_accuracies']),
      'linear_ci_lower': np.mean(results['linear_accuracies']) - 1.96 * np.std(results['linear_accuracies']) / np.sqrt(n_folds),
      'linear_ci_upper': np.mean(results['linear_accuracies']) + 1.96 * np.std(results['linear_accuracies']) / np.sqrt(n_folds),
      'tropical_mean_time': np.mean(results['tropical_times']),
      'tropical_km_iterations': np.mean(results['tropical_iters']),
      'linear_mean_time': np.mean(results['linear_times']),
      'tropical_mean_monomials': np.mean(results['tropical_n_monomials']),
      'feature_selection': feature_selection,
      'mean_spectral_radius': np.mean([r for r in results['spectral_radiuses'] if r is not None]),
      'raw': results
    }
    return summary
  
  return None

def load_datasets():
  """Load benchmark datasets from scikit-learn and OpenML"""
  datasets = {}
  
  print("Loading scikit-learn datasets...")
  datasets['Breast Cancer'] = load_breast_cancer()
  datasets['Iris'] = load_iris()
  datasets['Wine'] = load_wine()
  datasets['Diabetes'] = load_diabetes()   #   442 samples, 10 features
  
  print("Attempting to load OpenML datasets...")
  openml_datasets = [
  #  ('waveform-5000', 'Waveform'),
  ]
  
  for dataset_id, name in openml_datasets:
    try:
      print(f"Loading {name} dataset...")
      data = fetch_openml(dataset_id, parser='auto', as_frame=False)
      datasets[name] = data
      print(f"{name} dataset loaded successfully - {data.data.shape[0]} samples, {data.data.shape[1]} features")
    except Exception as e:
      print(f"Could not load {name} dataset: {e}")
  
  print(f"Successfully loaded {len(datasets)} datasets")
  return datasets

def run_benchmark():
  """Run benchmarks with different feature selection settings"""
  np.random.seed(42)
  
  os.makedirs('results', exist_ok=True)
  
  datasets = load_datasets()
  
  feature_selections = [None, 4]
  all_results = {fs: [] for fs in feature_selections}
  
  for fs in feature_selections:
    print(f"\n{'='*70}")
    print(f"RUNNING BENCHMARK WITH FEATURE_SELECTION={fs}")
    print(f"{'='*70}\n")
    
    for name, dataset in datasets.items():
      print(f"\n{'='*50}\nEvaluating dataset: {name} (feature_selection={fs})\n{'='*50}")
      
      if hasattr(dataset, 'data') and hasattr(dataset, 'target'):
        # scikit-learn dataset format
        X, y = dataset.data, dataset.target
      else:
        # OpenML dataset format
        X, y = dataset.data, dataset.target
        if hasattr(y, 'dtype') and y.dtype.kind == 'O':
          print("Converting string targets to integers...")
          labels = {label: i for i, label in enumerate(np.unique(y))}
          y = np.array([labels[label] for label in y])
      
      X = np.asarray(X, dtype=np.float64)
      y = np.asarray(y, dtype=np.int32)
      
      if X.shape[0] > 10000:
        print(f"Skipping {name} - too large ({X.shape[0]} samples)")
        continue
      
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)
      
      result = compare_models(X_scaled, y, feature_selection=fs)
      
      if result:
        result['dataset'] = name
        result['samples'] = X.shape[0]
        result['dimensions'] = X.shape[1]
        result['classes'] = len(np.unique(y))
        
        all_results[fs].append(result)
        
        print(f"\nResults for {name} (feature_selection={fs}):")
        print(f"Tropical SVM: {result['tropical_mean_acc']:.4f} ± {result['tropical_std_acc']:.4f}")
        print(f"Linear SVC:   {result['linear_mean_acc']:.4f} ± {result['linear_std_acc']:.4f}")
        print(f"Average number of monomials: {result['tropical_mean_monomials']:.1f}")
        print(f"Tropical training time: {result['tropical_mean_time']:.4f}s")
        print(f"Tropical #KM iterations: {result['tropical_km_iterations']:.4f}")
        print(f"Linear training time: {result['linear_mean_time']:.4f}s")
        print(f"Mean spectral radius: {result['mean_spectral_radius']:.6f}")
  
  return all_results

def create_csv_results(all_results):
  """Convert benchmark results to DataFrame format for CSV export"""
  rows = []
  
  for fs, results in all_results.items():
    for result in results:
      rows.append({
        'Dataset': result['dataset'],
        'Feature Selection': fs,
        'Classes': result['classes'],
        'Samples': result['samples'],
        'Dimensions': result['dimensions'],
        'Tropical Accuracy': result['tropical_mean_acc'],
        'Tropical Std Dev': result['tropical_std_acc'],
        'Linear Accuracy': result['linear_mean_acc'],
        'Linear Std Dev': result['linear_std_acc'],
        'Tropical Time': result['tropical_mean_time'],
        'Tropical #KM iterations': result['tropical_km_iterations'],
        'Linear Time': result['linear_mean_time'],
        'Num Monomials': result['tropical_mean_monomials'],
        'Spectral Radius': result['mean_spectral_radius']
      })
  
  return pd.DataFrame(rows)

def main():
  os.makedirs('results', exist_ok=True)
  print("Starting benchmark of Tropical SVM vs Linear SVC...")
  start_time = time.time()
  all_results = run_benchmark()
  total_time = time.time() - start_time
  
  print(f"\nBenchmark completed in {total_time:.2f}s")
  successful_count = sum(len(results) for results in all_results.values())
  print(f"Successfully evaluated {successful_count} dataset benchmarks")
  
  results_df = create_csv_results(all_results)
  results_df.to_csv('results/benchmark_results.csv', index=False)
  
  for fs, results in all_results.items():
    fs_name = str(fs) if fs is not None else "standard"
    if results:
      pd.DataFrame([{
        'Dataset': r['dataset'],
        'Classes': r['classes'],
        'Samples': r['samples'],
        'Dimensions': r['dimensions'],
        'Tropical Accuracy': r['tropical_mean_acc'],
        'Linear Accuracy': r['linear_mean_acc'],
        'Tropical Time': r['tropical_mean_time'],
        'Tropical #KM iterations': r['tropical_km_iterations'],
        'Linear Time': r['linear_mean_time'],
        'Num Monomials': r['tropical_mean_monomials'],
        'Spectral Radius': r['mean_spectral_radius']
      } for r in results]).to_csv(f'results/benchmark_fs{fs_name}.csv', index=False)
  
  if results_df.empty:
    print("No valid results to report.")
  else:
    print("\nResults Summary:")
    summary_cols = ['Dataset', 'Feature Selection', 'Classes', 'Samples', 'Tropical Accuracy', 'Linear Accuracy']
    print(results_df[summary_cols].to_string(index=False))
    
    print("\nRaw results saved to 'results/benchmark_results.csv'")
  
  print("\nBenchmark completed.")

if __name__ == "__main__":
  main()