# tropical-svm

tropy is a Python library for efficiently fitting piecewise linear models using tropical geometry and mean payoff games. It is distributed under the MIT License.

[Read the (draft) paper behind this library.](https://raw.githubusercontent.com/samuelbx/tropical-svm/main/tex/Report_WIP.pdf)

## Installation

If you already have a working installation of NumPy, the easiest way is to clone this repository.

### Dependencies

tropy was tested on Python 3.9 and NumPy 1.25. Graphing capabilities require Matplotlib. tropy still relies on Pandas (for saving weights), scikit-learn (for computing accuracies) and tqdm, even though these dependencies could easily be removed.

## Quickstart

### Playground

The easiest way to play with tropy and to reproduce graphs made in the paper is to use script `playground.py`. It allows to fit tropical polynomials on provided 3D datasets and to graph the results.

```
usage: playground.py [-h] [-s [file_path]] [--beta BETA] [--simplified]
                     [--feature-selection no_features]
                     {iris,iris-binary,moons,toy,toy-centers,toy-reverse,toy-centers-reverse,bintoy,bintoy-separated,bintoy-mixed,circular}
                     [degree]

Fitting and plotting tropical piecewise linear classifiers on 3D datasets

positional arguments:
  {...}                 Dataset to classify
  degree                Degree of tropical polynomial

optional arguments:
  -h, --help            Show this help message and exit
  -s [file_path], --save [file_path]
                        Save the figure (.PGF)
  --beta BETA           If specified, Beta value for using 'linear SVM on log paper' trick
  --simplified          Provide a simplified view of the hypersurface, with the decision boundary only
  --feature-selection no_features
                        Experimental: heuristic to generate more relevant
                        monomials based on data. Specify the number of points to sample per class if wanted. Bypasses degree option.

```

./playground.py moons 3 -s ./img/moons.svg
./playground.py moons --feature-selection 5 --simplified -s ./img/moons_feature_sel.svg
./playground.py circular 3 -s ./img/circular.svg
./playground.py circular --feature-selection 5 -s ./img/circular_feature_sel.svg
./playground.py bintoy-separated 1 ./img/bintoy-separated.svg
./playground.py bintoy-separated 1 --beta 10 -s ./img/bintoy-separated_feature_sel.svg
./playground.py toy-reverse 3 -s ./img/toy-reverse.svg
./playground.py toy-reverse --feature-selection 3 -s ./img/toy-reverse_feature_sel.svg


| Commands | Results |
|--|--|
| `./playground.py moons 3` <br /> `./playground.py moons --feature-selection 5 --simplified` | <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/moons.svg" width="40%"> <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/moons_feature_sel.svg" width="40%"> |
| `./playground.py circular 3` <br /> `./playground.py circular --feature-selection 5` | <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/circular.svg" width="40%"> <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/circular_feature_sel.svg" width="40%"> |
| `./playground.py bintoy-separated 1` <br /> `./playground.py bintoy-separated 1 --beta 10` | <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/bintoy-separated.svg" width="40%"> <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/bintoy-separated_feature_sel.svg" width="40%"> |
| `./playground.py toy-reverse 3` <br /> `./playground.py toy-reverse --feature-selection 3` | <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/toy-reverse.svg" width="40%"> <img src="https://raw.githubusercontent.com/samuelbx/tropical-svm/main/img/toy-reverse_feature_sel.svg" width="40%"> |


### Fitting custom datasets

tropy is only able of plotting tropical polynomials for 3-dimensional data, but it can perform classification in much higher dimensions.

To fit a classifier based on a cubic tropical polynomial, for instance:
```python
from tropy.svm import TropicalSVC
model = TropicalSVC()
model.fit(Xtrain, poly_degree = 3)
```
where `Xtrain` is a list of NumPy 2D arrays. Each of them corresponds to some data class, and stores data points as columns.

To evaluate model accuracy on test data:
```python
acc = model.accuracy(Xtest)
```

Methods `model.predict`, `model.export_weights` and `model.load_weights` are also provided.

### Plotting tropical hypersurfaces

Module `tropy.graph` comes with a handful of methods to plot tridimensional tropical hypersurfaces. Feel free to use them for custom purposes.

## Other experiments

- `tropicalization.py` shows that *Maslov's dequantization* of classical SVMs does not produce good results.
- `tightness.py` experimentally evaluates the tightness of the norm boundaries derived in Appendix A of the paper.
- `iris_multiclass.ipynb` shows how one could use tropy on real-world data to build a multi-class classifier.
