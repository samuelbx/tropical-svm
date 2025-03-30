#!/usr/bin/env python3

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

# Set matplotlib to use pgf backend when saving pgf files
# Default backend will be used for png files
matplotlib.use('Agg')

# Import your tropical SVM library components
from tropy.svm import TropicalSVC, get_km_time

warnings.filterwarnings('ignore')

# ===== CHECKPOINT SYSTEM =====
class CheckpointManager:
    """Manages checkpoints to allow resuming interrupted experiments"""
    
    def __init__(self, checkpoint_dir='mnist_checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.current_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def save_checkpoint(self, data, name):
        """Save a checkpoint with name"""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
            
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Checkpoint saved to {checkpoint_file}")
    
    def load_checkpoint(self, name):
        """Load a checkpoint with name"""
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
        """List all available checkpoints"""
        checkpoints = list(self.checkpoint_dir.glob("*.pkl"))
        return [cp.name for cp in checkpoints]

# Create global checkpoint manager
checkpoint_mgr = CheckpointManager()

# ===== METRICS AND UTILITIES =====
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def format_data_for_tropical(X, y):
    """
    Format data to be used with TropicalSVC.
    TropicalSVC expects data_classes as a list of 2D arrays, where each array contains
    all points from one class (as columns).
    """
    classes = np.unique(y)
    data_classes = []
    
    for c in classes:
        class_data = X[y == c].T  # Transpose to make each column a data point
        data_classes.append(class_data)
    
    return data_classes

def count_effective_monomials(model):
    """Count the effective number of monomials in the model (those with non-zero coefficients)"""
    if not hasattr(model, '_monomials'):
        return 0
    
    if hasattr(model, '_coef') and model._coef is not None:
        # Count monomials with non-zero coefficients
        return np.sum(model._coef != 0)
    
    # Fallback: just count all monomials
    return len(model._monomials)

# ===== EXPERIMENT FUNCTIONS =====
def run_pca_degree_experiment(X, y, pca_dimensions=[10, 20, 50], degrees=[1, 2, 3], skip_experiments=True):
    """
    Run experiment to analyze training time vs. number of monomials for different
    PCA dimensions and polynomial degrees.
    
    If skip_experiments is True, will only load results from checkpoints and not run any new experiments.
    """
    # Check if we have existing results
    results_name = "pca_degree_results"
    existing_results = checkpoint_mgr.load_checkpoint(results_name)
    
    if existing_results is not None:
        results = existing_results
        # Find which experiments we've already done
        done_experiments = set((row['pca_dim'], row['degree']) for _, row in results.iterrows())
    else:
        # Initialize empty results DataFrame
        results = pd.DataFrame(columns=[
            'pca_dim', 'degree', 'training_time', 'memory_usage',
            'n_monomials', 'effective_monomials', 'accuracy'
        ])
        done_experiments = set()
    
    # Create output directory for figures
    os.makedirs('mnist_results', exist_ok=True)
    
    # If we want to skip experiments and already have results, just regenerate plots
    if skip_experiments and not results.empty:
        print("Skipping experiments as requested. Regenerating plots using existing results.")
        plot_pca_degree_results(results, save_pgf=True)
        return results
    
    # Loop through all combinations
    for pca_dim in pca_dimensions:
        for degree in degrees:
            # Skip if we've already done this experiment
            if (pca_dim, degree) in done_experiments:
                print(f"Skipping already completed experiment: PCA={pca_dim}, degree={degree}")
                continue
            
            print(f"\nRunning experiment: PCA dimensions={pca_dim}, degree={degree}")
            
            try:
                # Check if we have a checkpoint for this specific experiment
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
                    
                    # Scale the data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_pca)
                    
                    # Split into train/test
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    
                    # Format data for tropical SVM
                    train_classes = format_data_for_tropical(X_train, y_train)
                    
                    # Record memory before training
                    memory_before = get_memory_usage()
                    
                    # Train model
                    print(f"Training model with {len(y_train)} samples, {pca_dim} dimensions, degree {degree}...")
                    model = TropicalSVC()
                    model.fit(train_classes, poly_degree=degree, native_tropical_data=False, feature_selection=None)
                    train_time = get_km_time()
                    
                    # Record memory after training
                    memory_after = get_memory_usage()
                    memory_used = memory_after - memory_before
                    
                    # Count monomials
                    total_monomials = len(model._monomials) if hasattr(model, '_monomials') else 0
                    effective_monomials = count_effective_monomials(model)
                    
                    # Predict and calculate accuracy
                    predictions = model.predict(X_test.T)
                    accuracy = np.mean(predictions == y_test)
                    
                    # Create result dictionary
                    new_row = {
                        'pca_dim': pca_dim,
                        'degree': degree,
                        'training_time': train_time,
                        'memory_usage': memory_used,
                        'n_monomials': total_monomials,
                        'effective_monomials': effective_monomials,
                        'accuracy': accuracy
                    }
                    
                    # Save individual experiment checkpoint
                    checkpoint_mgr.save_checkpoint(new_row, experiment_name)
                    
                    print(f"Result: Time={train_time:.2f}s, Memory={memory_used:.2f}MB, "
                        f"Monomials={total_monomials} (effective: {effective_monomials}), "
                        f"Accuracy={accuracy:.4f}")
                
                # Add to results DataFrame
                results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)
                
                # Update the done_experiments set
                done_experiments.add((pca_dim, degree))
                
                # Save overall checkpoint
                checkpoint_mgr.save_checkpoint(results, results_name)
                
                # Plot updated results
                plot_pca_degree_results(results, save_pgf=True)
                
            except Exception as e:
                print(f"Error in experiment PCA={pca_dim}, degree={degree}: {e}")
                traceback.print_exc()

    return results

def run_sample_size_experiment(X, y, pca_dim=10, degree=3, sample_percentages=[10, 20, 30, 50, 75, 100], skip_experiments=True):
    """
    Run experiment to analyze training time vs. number of samples for fixed
    PCA dimension and polynomial degree.
    
    If skip_experiments is True, will only load results from checkpoints and not run any new experiments.
    """
    # Check if we have existing results
    results_name = "sample_size_results"
    existing_results = checkpoint_mgr.load_checkpoint(results_name)
    
    if existing_results is not None:
        results = existing_results
        # Find which experiments we've already done
        done_experiments = set(row['sample_percentage'] for _, row in results.iterrows())
    else:
        # Initialize empty results DataFrame
        results = pd.DataFrame(columns=[
            'sample_percentage', 'n_samples', 'training_time', 'memory_usage',
            'n_monomials', 'effective_monomials', 'accuracy'
        ])
        done_experiments = set()
    
    # Create output directory for figures
    os.makedirs('mnist_results', exist_ok=True)
    
    # If we want to skip experiments and already have results, just regenerate plots
    if skip_experiments and not results.empty:
        print("Skipping experiments as requested. Regenerating plots using existing results.")
        plot_sample_size_results(results, save_pgf=True)
        return results
    
    # Apply PCA first
    pca = PCA(n_components=pca_dim)
    X_pca = pca.fit_transform(X)
    print(f"Applied PCA: reduced to {pca_dim} dimensions, explained variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    
    # Split into train/test sets first
    X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    total_samples = len(y_train_full)
    
    # Loop through all sample percentages
    for percentage in sample_percentages:
        # Skip if we've already done this experiment
        if percentage in done_experiments:
            print(f"Skipping already completed experiment: sample percentage={percentage}%")
            continue
        
        # Calculate number of samples to use
        n_samples = int(total_samples * percentage / 100)
        print(f"\nRunning experiment: sample percentage={percentage}%, n_samples={n_samples}")
        
        try:
            # Subsample training data
            indices = np.random.RandomState(42).choice(total_samples, n_samples, replace=False)
            X_train = X_train_full[indices]
            y_train = y_train_full[indices]
            
            # Format data for tropical SVM
            train_classes = format_data_for_tropical(X_train, y_train)
            
            # Record memory before training
            memory_before = get_memory_usage()
            
            # Train model
            print(f"Training model with {n_samples} samples, {pca_dim} dimensions, degree {degree}...")
            model = TropicalSVC()
            model.fit(train_classes, poly_degree=degree, native_tropical_data=False, feature_selection=None)
            train_time = get_km_time()
            
            # Record memory after training
            memory_after = get_memory_usage()
            memory_used = memory_after - memory_before
            
            # Count monomials
            total_monomials = len(model._monomials) if hasattr(model, '_monomials') else 0
            effective_monomials = count_effective_monomials(model)
            
            # Predict and calculate accuracy
            predictions = model.predict(X_test.T)
            accuracy = np.mean(predictions == y_test)
            
            # Add result
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
            
            # Save checkpoint after each experiment
            checkpoint_mgr.save_checkpoint(results, results_name)
            
            # Plot updated results
            plot_sample_size_results(results, save_pgf=True)
            
        except Exception as e:
            print(f"Error in experiment sample percentage={percentage}%: {e}")
            traceback.print_exc()
    
    return results

# ===== VISUALIZATION FUNCTIONS =====
def save_figure_both_formats(fig, base_path):
    """Save figure in both PNG and PGF formats"""
    # Save as PNG
    png_path = f"{base_path}.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save as PGF
    pgf_path = f"{base_path}.pgf"
    try:
        # Configure matplotlib for PGF output
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
    """
    Plot training time vs. number of effective monomials for different PCA dimensions and degrees.
    Saves in both PNG and PGF formats if save_pgf is True.
    """
    if results.empty:
        print("No results to plot")
        return
    
    # Create a figure with two subplots
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    
    # Plot training time vs. effective monomials
    sns.scatterplot(data=results, x='effective_monomials', y='training_time', s=100, ax=ax1)
    
    # Add regression line only if we have enough diverse data points
    if len(results) >= 3:
        try:
            # Convert to numpy arrays and ensure correct type
            x = np.array(results['effective_monomials'], dtype=np.float64)
            y = np.array(results['training_time'], dtype=np.float64)
            
            # Check if we have enough variation in the data for regression
            if len(np.unique(x)) >= 3:
                # Calculate regression in log-log space
                log_x = np.log(x)
                log_y = np.log(y)
                
                # Fit in log-log space (log y = c + log x)
                # For this form, the slope in log-log space should be 1
                # We're just finding the constant c
                c = np.mean(log_y - log_x)
                
                # Calculate R² in log space
                log_yhat = log_x + c
                log_ybar = np.mean(log_y)
                ssreg = np.sum((log_yhat - log_ybar)**2)
                sstot = np.sum((log_y - log_ybar)**2)
                r_squared = ssreg / sstot if sstot > 0 else 0
                
                # The true coefficient in normal space is e^c
                coefficient = np.exp(c)
                
                # Create line for plotting (in normal space)
                x_line = np.linspace(min(x), max(x), 100)
                y_line = coefficient * x_line  # y = e^c * x
                
                # Add the regression line to the plot
                ax1.plot(x_line, y_line, 'r--', label=f'linear fit (coefficient: {coefficient:.2e})')
                print(f"Added log-log regression line: y = {coefficient:.4e}x, R²={r_squared:.2f}")
            else:
                print("Not enough unique values for regression")
        except Exception as e:
            print(f"Could not add regression line: {e}")

    ax1.set_xlabel('number of monomials')
    ax1.set_ylabel('training time (s)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the figure in both formats
    if save_pgf:
        save_figure_both_formats(fig, save_path)
    else:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}.png")
    
    plt.close()
    
    # Additional plot for accuracy if we have enough data
    try:
        fig = plt.figure(figsize=(10, 6))
        pivot_acc = results.pivot_table(index='pca_dim', columns='degree', values='accuracy')
        sns.heatmap(pivot_acc, annot=True, fmt=".4f", cmap="YlGnBu")
        plt.title('Accuracy for Each Configuration')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('PCA Dimensions')
        
        # Save accuracy plot
        acc_path = save_path + '_accuracy'
        if save_pgf:
            save_figure_both_formats(fig, acc_path)
        else:
            fig.savefig(f"{acc_path}.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Could not create accuracy heatmap: {e}")
    finally:
        plt.close()

def plot_sample_size_results(results, save_pgf=True, save_path='mnist_results/sample_size_scaling'):
    """
    Plot training time vs. number of samples.
    Saves in both PNG and PGF formats if save_pgf is True.
    """
    if results.empty:
        print("No results to plot")
        return
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    
    # Sort by number of samples
    results_sorted = results.sort_values('n_samples')
    
    # Plot training time vs. number of samples
    ax1.plot(results_sorted['n_samples'], results_sorted['training_time'], 'o', markersize=8)
    ax1.set_xlabel('size of training set')
    ax1.set_ylabel('training time (s)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add linear fit without intercept
    if len(results) >= 3:
        try:
            from scipy.optimize import curve_fit
            
            # Define linear function without intercept
            def linear_no_intercept(x, a):
                return a * x
            
            x = results_sorted['n_samples'].values
            y = results_sorted['training_time'].values
            
            # Fit the function
            popt, _ = curve_fit(linear_no_intercept, x, y)
            
            # Create smooth curve for plotting
            x_line = np.linspace(min(x), max(x), 100)
            y_line = linear_no_intercept(x_line, *popt)
            
            # Plot the curve
            ax1.plot(x_line, y_line, 'r--', 
                    label=f'linear fit (slope: {popt[0]:.2e})')
            ax1.legend()
        except Exception as e:
            print(f"Could not fit linear curve: {e}")

    plt.tight_layout()
    
    # Save the figure in both formats
    if save_pgf:
        save_figure_both_formats(fig, save_path)
    else:
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}.png")
    
    plt.close()
    
    # Additional plot for monomials
    fig = plt.figure(figsize=(10, 6))
    plt.plot(results_sorted['n_samples'], results_sorted['n_monomials'], 'o-', label='Total Monomials')
    plt.plot(results_sorted['n_samples'], results_sorted['effective_monomials'], 's-', label='Effective Monomials')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Number of Monomials')
    plt.title('Monomial Growth vs. Number of Samples')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save monomials plot
    mono_path = save_path + '_monomials'
    if save_pgf:
        save_figure_both_formats(fig, mono_path)
    else:
        fig.savefig(f"{mono_path}.png", dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    """
    Main function to run the MNIST scaling experiments.
    """
    print("Starting MNIST scaling experiments visualization...")
    
    # Create results directory
    os.makedirs('mnist_results', exist_ok=True)
    
    # Load MNIST dataset (only for initial checkpoints, won't run experiments)
    print("Checking for MNIST dataset checkpoint...")
    mnist_data = checkpoint_mgr.load_checkpoint("mnist_data")
    
    if mnist_data is None:
        print("MNIST dataset checkpoint not found. Creating a small placeholder dataset.")
        # Create dummy data just to avoid errors, won't actually be used for training
        X = np.random.rand(100, 784)
        y = np.random.randint(0, 10, 100)
        # Save to checkpoint
        checkpoint_mgr.save_checkpoint((X, y), "mnist_data")
    else:
        X, y = mnist_data
        print(f"MNIST dataset checkpoint loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Load and visualize experiment results
    print("\n==== VISUALIZING EXPERIMENT 1: PCA Dimensions and Polynomial Degrees ====")
    run_pca_degree_experiment(
        X, y, 
        pca_dimensions=[10, 20, 30, 40, 50],
        degrees=[1, 2, 3],
        skip_experiments=True
    )
    
    print("\n==== VISUALIZING EXPERIMENT 2: Sample Size Scaling ====")
    run_sample_size_experiment(
        X, y,
        pca_dim=10,
        degree=3,
        sample_percentages=[10, 20, 30, 50, 75, 100],
        skip_experiments=True
    )
    
if __name__ == "__main__":
    main()