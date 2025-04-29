#!/bin/bash

# Repository
./playground.py moons 3 -s ./img/moons.svg
./playground.py moons --feature-selection 5 --simplified -s ./img/moons_feature_sel.svg
./playground.py circular 3 -s ./img/circular.svg
./playground.py circular --feature-selection 5 -s ./img/circular_feature_sel.svg
./playground.py bintoy-separated 1 -s ./img/bintoy-separated.svg
./playground.py bintoy-separated 1 --beta 10 -s ./img/bintoy-separated_maslov.svg
./playground.py toy-reverse 3 -s ./img/toy-reverse.svg
./playground.py toy-reverse --feature-selection 3 -s ./img/toy-reverse_feature_sel.svg

# Paper
./playground.py bintoy-separated 1 -s ./tex/figures/binary-toy-sep.pgf
./playground.py blobs 3 -s ./tex/figures/blobs.pgf
./playground.py moons --feature-selection 10 --simplified -s ./tex/figures/moons-feat-sel.pgf
./playground.py moons 2 --equations -s ./tex/figures/moons.pgf
./playground.py bintoy-separated --equations --show-axes -s ./tex/figures/example.pgf
python3 ./tropicalization.py