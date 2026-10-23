#!/usr/bin/env bash
# Table 2: semi-supervised MNIST (1,000 labels), 10 seeds; KL weight = classification weight = 0.01 for all methods.
source "$(dirname "$0")/common.sh"
for p in $PRIORS; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    for m in aevb_semi aevb_iaf_semi aaeb_semi ddvi_semi; do
      train dataset=mnist_semi model=$m prior=$p seed=$s
    done
  done
done
elbo runs/mnist_semi/{aevb_semi,aevb_iaf_semi,ddvi_semi}/*/seed*
python aggregate.py runs
