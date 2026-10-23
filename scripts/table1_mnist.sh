#!/usr/bin/env bash
# Table 1: unsupervised MNIST, 3 seeds.
source "$(dirname "$0")/common.sh"
for p in $PRIORS; do
  for s in 0 1 2; do
    for m in aevb aevb_iaf aaeb h_aevb pis; do
      train dataset=mnist model=$m prior=$p seed=$s
    done
    train dataset=mnist model=ddvi_simplified prior=$p seed=$s   # stage 1
    train dataset=mnist model=ddvi prior=$p seed=$s              # stage 2 (initialized from stage 1)
  done
done
elbo runs/mnist/{aevb,aevb_iaf,h_aevb,pis,ddvi}/*/seed*
python aggregate.py runs
