#!/usr/bin/env bash
# Table 7: unsupervised CIFAR-10, 3 seeds (AAEB is trained for 100 epochs).
source "$(dirname "$0")/common.sh"
for p in $PRIORS; do
  for s in 0 1 2; do
    for m in aevb aevb_iaf h_aevb pis; do
      train dataset=cifar10 model=$m prior=$p seed=$s
    done
    train dataset=cifar10 model=aaeb prior=$p seed=$s model.num_epochs=100
    train dataset=cifar10 model=ddvi_simplified prior=$p seed=$s
    train dataset=cifar10 model=ddvi prior=$p seed=$s
  done
done
elbo runs/cifar10/{aevb,aevb_iaf,h_aevb,pis,ddvi}/*/seed*
python aggregate.py runs
