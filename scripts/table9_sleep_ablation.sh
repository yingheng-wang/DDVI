#!/usr/bin/env bash
# Table 9: DDVI on MNIST without the sleep step (3 seeds). The other rows of Table 9 are the Latent NLL of the
# Table 1 runs; the ddvi_simplified runs of table1_mnist.sh are reused (trained here if missing).
source "$(dirname "$0")/common.sh"
for p in $PRIORS; do
  for s in 0 1 2; do
    [ -f runs/mnist/ddvi_simplified/$p/seed$s/model.pt ] || train dataset=mnist model=ddvi_simplified prior=$p seed=$s
    train dataset=mnist model=ddvi prior=$p seed=$s model.sleep_ct=0 tag=_nosleep
  done
done
python aggregate.py runs
