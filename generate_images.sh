#!/bin/bash

# Generate repository's demo figures
./playground.py moons 3 -s ./img/moons.svg
./playground.py moons --feature-selection 5 --simplified -s ./img/moons_feature_sel.svg
./playground.py circular 3 -s ./img/circular.svg
./playground.py circular --feature-selection 5 -s ./img/circular_feature_sel.svg
./playground.py bintoy-separated 1 -s ./img/bintoy-separated.svg
./playground.py bintoy-separated 1 --beta 10 -s ./img/bintoy-separated_feature_sel.svg
./playground.py toy-reverse 3 -s ./img/toy-reverse.svg
./playground.py toy-reverse --feature-selection 3 -s ./img/toy-reverse_feature_sel.svg

# Generate paper's figures
./playground.py bintoy-separated 1 -s ./tex/figures/bintoy-separated_1.pgf
./playground.py moons 3 -s ./tex/figures/moon_3.pgf
./playground.py circular 3 -s ./tex/figures/circular_3.pgf
./playground.py toy-reverse 3 -s ./tex/figures/toy-reverse_3.pgf
./playground.py circular --feature-selection 10 -s ./tex/figures/circular_3_feature_eng.pgf
./playground.py moons --feature-selection 5 --simplified -s ./tex/figures/moon_10_feature_eng.pgf
