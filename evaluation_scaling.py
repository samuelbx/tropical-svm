import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import psutil
import warnings
from datetime import datetime
from pathlib import Path
import pickle
import traceback
import matplotlib

matplotlib.use('Agg')

from tropy.svm import TropicalSVC, get_km_time

warnings.filterwarnings('ignore')

class CheckpointManager:
  def __init__(self, checkpoint_dir='mnist_checkpoints'):
    self.checkpoint_dir = Path(checkpoint_dir)
    self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
  def save_checkpoint(self, data, name):
    checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
      
    with open(checkpoint_file, 'wb') as f:
      pickle.dump(data, f)
    
    print(f"Checkpoint saved to {checkpoint_file}")
  
  def load_checkpoint(self, name):
    checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
    
    if not checkpoint_file.exists():
      return None
    
    try:
      with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
      print(f"Checkpoint loaded from {checkpoint_file}")
      return data
    except Exception as e:
      print(f"Error loading checkpoint: {e}")
      return None
  
  def list_checkpoints(self):
    checkpoints = list(self.checkpoint_dir.glob("*.pkl"))
    return [cp.name for cp in checkpoints]

checkpoint_mgr = CheckpointManager()

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

def count_effective_monomials(model):
  if not hasattr(model, '_monomials'):
    return 0
  
  return len(model._monomials)

def run_pca_degree_experiment(X, y, pca_dimensions=[10, 20, 50], degrees=[1, 2, 3], skip_experiments=True):
  results_name = "pca_degree_results"
  existing_results = checkpoint_mgr.load_checkpoint(results_name)
  
  if existing_results is not None:
    results = existing_results
    done_experiments = set((row['pca_dim'], row['degree']) for _, row in results.iterrows())
  else:
    results = pd.DataFrame(columns=[
      'pca_dim', 'degree', 'training_time', 'memory_usage',
      'n_monomials', 'effective_monomials', 'accuracy'
    ])
    done_experiments = set()
  
  os.makedirs('mnist_results', exist_ok=True)
  
  if skip_experiments and not results.empty:
    print("Skipping experiments as requested. Regenerating plots using existing results.")
    plot_pca_degree_results(results, save_pgf=True)
    return results
  
  for pca_dim in pca_dimensions:
    for degree in degrees:
      if (pca_dim, degree) in done_experiments:
        print(f"Skipping already completed experiment: PCA={pca_dim}, degree={degree}")
        continue
      
      print(f"\nRunning experiment: PCA dimensions={pca_dim}, degree={degree}")
      
      try:
        experiment_name = f"pca{pca_dim}_degree{degree}"
        experiment_result = checkpoint_mgr.load_checkpoint(experiment_name)
        
        if experiment_result is not None:
          print(f"Loading results from checkpoint for PCA={pca_dim}, degree={degree}")
          new_row = experiment_result
        else:
          # Apply PCA
          pca = PCA(n_components=pca_dim)
          X_pca = pca.fit_transform(X)
          print(f"Applied PCA: reduced to {pca_dim} dimensions, explained variance: {sum(pca.explained_variance_ratio_):.4f}")
          
          scaler = StandardScaler()
          X_scaled = scaler.fit_transform(X_pca)
          
          X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
          train_classes = format_data_for_tropical(X_train, y_train)
          
          memory_before = get_memory_usage()
          
          print(f"Training model with {len(y_train)} samples, {pca_dim} dimensions, degree {degree}...")
          model = TropicalSVC()
          model.fit(train_classes, poly_degree=degree, native_tropical_data=False, feature_selection=None)
          train_time = get_km_time()
          
          memory_after = get_memory_usage()
          memory_used = memory_after - memory_before
          
          total_monomials = len(model._monomials) if hasattr(model, '_monomials') else 0
          effective_monomials = count_effective_monomials(model)
          
          predictions = model.predict(X_test.T)
          accuracy = np.mean(predictions == y_test)
          
          new_row = {
            'pca_dim': pca_dim,
            'degree': degree,
            'training_time': train_time,
            'memory_usage': memory_used,
            'n_monomials': total_monomials,
            'effective_monomials': effective_monomials,
            'accuracy': accuracy
          }
          
          checkpoint_mgr.save_checkpoint(new_row, experiment_name)
          
          print(f"Result: Time={train_time:.2f}s, Memory={memory_used:.2f}MB, "
            f"Monomials={total_monomials} (effective: {effective_monomials}), "
            f"Accuracy={accuracy:.4f}")
        
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
        done_experiments.add((pca_dim, degree))
        checkpoint_mgr.save_checkpoint(results, results_name)
        plot_pca_degree_results(results, save_pgf=True)
        
      except Exception as e:
        print(f"Error in experiment PCA={pca_dim}, degree={degree}: {e}")
        traceback.print_exc()

  return results

def run_sample_size_experiment(X, y, pca_dim=10, degree=3, sample_percentages=[10, 20, 30, 50, 75, 100], skip_experiments=True):
  results_name = "sample_size_results"
  existing_results = checkpoint_mgr.load_checkpoint(results_name)
  
  if existing_results is not None:
    results = existing_results
    done_experiments = set(row['sample_percentage'] for _, row in results.iterrows())
  else:
    results = pd.DataFrame(columns=[
      'sample_percentage', 'n_samples', 'training_time', 'memory_usage',
      'n_monomials', 'effective_monomials', 'accuracy'
    ])
    done_experiments = set()
  
  os.makedirs('mnist_results', exist_ok=True)
  
  if skip_experiments and not results.empty:
    print("Skipping experiments as requested. Regenerating plots using existing results.")
    plot_sample_size_results(results, save_pgf=True)
    return results
  
  pca = PCA(n_components=pca_dim)
  X_pca = pca.fit_transform(X)
  print(f"Applied PCA: reduced to {pca_dim} dimensions, explained variance: {sum(pca.explained_variance_ratio_):.4f}")
  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_pca)
  
  X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
  total_samples = len(y_train_full)
  
  for percentage in sample_percentages:
    if percentage in done_experiments:
      print(f"Skipping already completed experiment: sample percentage={percentage}%")
      continue
    
    n_samples = int(total_samples * percentage / 100)
    print(f"\nRunning experiment: sample percentage={percentage}%, n_samples={n_samples}")
    
    try:
      indices = np.random.RandomState(42).choice(total_samples, n_samples, replace=False)
      X_train = X_train_full[indices]
      y_train = y_train_full.iloc[indices]
      
      train_classes = format_data_for_tropical(X_train, y_train)
      
      memory_before = get_memory_usage()
      
      print(f"Training model with {n_samples} samples, {pca_dim} dimensions, degree {degree}...")
      model = TropicalSVC()
      model.fit(train_classes, poly_degree=degree, native_tropical_data=False, feature_selection=None)
      train_time = get_km_time()
      memory_after = get_memory_usage()
      memory_used = memory_after - memory_before
      
      total_monomials = len(model._monomials) if hasattr(model, '_monomials') else 0
      effective_monomials = count_effective_monomials(model)
      
      predictions = model.predict(X_test.T)
      accuracy = np.mean(predictions == y_test)
      
      new_row = {
        'sample_percentage': percentage,
        'n_samples': n_samples,
        'training_time': train_time,
        'memory_usage': memory_used,
        'n_monomials': total_monomials,
        'effective_monomials': effective_monomials,
        'accuracy': accuracy
      }
      results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
      
      print(f"Result: Time={train_time:.2f}s, Memory={memory_used:.2f}MB, "
          f"Monomials={total_monomials} (effective: {effective_monomials}), "
          f"Accuracy={accuracy:.4f}")
      
      checkpoint_mgr.save_checkpoint(results, results_name)
      plot_sample_size_results(results, save_pgf=True)
      
    except Exception as e:
      print(f"Error in experiment sample percentage={percentage}%: {e}")
      traceback.print_exc()
  
  return results

def save_figure_both_formats(fig, base_path):
  png_path = f"{base_path}.png"
  fig.savefig(png_path, dpi=300, bbox_inches='tight')
  
  pgf_path = f"{base_path}.pgf"
  try:
    plt.rcParams.update({
      "pgf.texsystem": "pdflatex",
      "pgf.preamble": r"\usepackage{amsmath,amsfonts,amssymb}",
    })
    fig.savefig(pgf_path, bbox_inches='tight')
    print(f"Figures saved to {png_path} and {pgf_path}")
  except Exception as e:
    print(f"Could not save PGF figure: {e}")
    print(f"PNG figure saved to {png_path}")

def plot_pca_degree_results(results, save_pgf=True, save_path='mnist_results/pca_degree_scaling'):
  if results.empty:
    print("No results to plot")
    return
  
  fig = plt.figure(figsize=(4, 4))
  ax1 = fig.add_subplot(111)
  
  sns.scatterplot(data=results, x='effective_monomials', y='training_time', s=100, ax=ax1)

  ax1.set_xlabel('number of monomials')
  ax1.set_ylabel('training time (s)')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.grid(True, linestyle='--', alpha=0.7)
  
  plt.tight_layout()
  
  if save_pgf:
    save_figure_both_formats(fig, save_path)
  else:
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}.png")
  
  plt.close()

def plot_sample_size_results(results, save_pgf=True, save_path='mnist_results/sample_size_scaling'):
  if results.empty:
    print("No results to plot")
    return
  
  fig = plt.figure(figsize=(4, 4))
  ax1 = fig.add_subplot(111)
  
  results_sorted = results.sort_values('n_samples')
  
  ax1.plot(results_sorted['n_samples'], results_sorted['training_time'], 'o', markersize=8)
  ax1.set_xlabel('size of training set')
  ax1.set_ylabel('training time (s)')
  ax1.grid(True, linestyle='--', alpha=0.7)
  
  plt.tight_layout()
  
  if save_pgf:
    save_figure_both_formats(fig, save_path)
  else:
    fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}.png")
  
  plt.close()
  
def main():
  print("Starting MNIST scaling experiments visualization...")
  
  os.makedirs('mnist_results', exist_ok=True)
  
  print("Checking for MNIST dataset checkpoint...")
  mnist_data = checkpoint_mgr.load_checkpoint("mnist_data")
  if mnist_data is None:
    print("MNIST dataset checkpoint not found. Downloading dataset...")
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    
    print(f"MNIST dataset downloaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    mnist_data = (X, y)
    checkpoint_mgr.save_checkpoint(mnist_data, "mnist_data")

  X, y = mnist_data
  print(f"MNIST dataset checkpoint loaded: {X.shape[0]} samples, {X.shape[1]} features")
  
  print("\n==== VISUALIZING EXPERIMENT 1: PCA Dimensions and Polynomial Degrees ====")
  run_pca_degree_experiment(
    X, y, 
    pca_dimensions=[10, 20, 30, 40, 50],
    degrees=[1, 2, 3],
    skip_experiments=False
  )
  
  print("\n==== VISUALIZING EXPERIMENT 2: Sample Size Scaling ====")
  run_sample_size_experiment(
    X, y,
    pca_dim=10,
    degree=3,
    sample_percentages=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    skip_experiments=False
  )
  
if __name__ == "__main__":
  main()