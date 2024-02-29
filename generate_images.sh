#!/bin/bash

# Generate repository's demo figures
./playground.py moons 3 -s ./img/moons_3.svg
./playground.py circular --feature-selection 5 -s ./img/circular_1.svg
./playground.py bintoy-separated 1 --beta 10 -s ./img/bintoy-separated_1.svg
./playground.py toy-reverse 3 -s ./img/toy-reverse_3.svg

# Generate paper's figures
./playground.py bintoy-separated 1 -s ./tex/figures/bintoy-separated_1.pgf
./playground.py moons 3 -s ./tex/figures/moon_3.pgf
./playground.py circular 3 -s ./tex/figures/circular_3.pgf
./playground.py toy-reverse 3 -s ./tex/figures/toy-reverse_3.pgf
./playground.py circular --feature-selection 10 -s ./tex/figures/circular_3_feature_eng.pgf
./playground.py moons --feature-selection 5 --simplified -s ./tex/figures/moon_10_feature_eng.pgf
