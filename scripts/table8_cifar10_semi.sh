#!/usr/bin/env bash
# Table 8: semi-supervised CIFAR-10 (10,000 labels), 3 seeds; KL weight = classification weight = 0.5 for all methods
# with these weights (AAEB has none and is trained for 100 epochs).
source "$(dirname "$0")/common.sh"
for p in $PRIORS; do
  for s in 0 1 2; do
    for m in aevb_semi aevb_iaf_semi ddvi_semi; do
      train dataset=cifar10_semi model=$m prior=$p seed=$s model.kld_weight=0.5
    done
    train dataset=cifar10_semi model=aaeb_semi prior=$p seed=$s model.num_epochs=100
  done
done
elbo runs/cifar10_semi/{aevb_semi,aevb_iaf_semi,ddvi_semi}/*/seed*
python aggregate.py runs
