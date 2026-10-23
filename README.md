# Denoising Diffusion Variational Inference (DDVI)

Code for [Denoising Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors](https://arxiv.org/abs/2401.02739)
by Wasu Top Piriyakulkij\*, Yingheng Wang\*, and Volodymyr Kuleshov (\* equal contribution).

<p align="center"><img src="imgs/ddvi.png" width="80%"></p>

DDVI is an amortized variational inference method for latent variable models that uses a diffusion model as the
variational posterior: an encoder gives q(y_T | x), a reverse diffusion process q(z | y_T, x) refines y_T into the
latent z, and a forward diffusion process r(y | z) serves as the auxiliary reverse model of an auxiliary-variable ELBO.
Training alternates wake steps on (a surrogate of) this ELBO with sleep steps that fit the reverse process to samples
from the prior p(z) and the decoder.

This repository contains the MNIST and CIFAR-10 experiments of the paper (unsupervised and semi-supervised learning with
structured 2-d priors; Tables 1, 2, 7, 8 and 9) with the final settings.

## Installation

```bash
conda create -n ddvi python=3.9
conda activate ddvi
pip install -r requirements.txt
```

MNIST and CIFAR-10 are downloaded by torchvision into `data/` (`data_dir=...` to change it).

## Quick start

```bash
# simplified DDVI (x-independent reverse process); also the first training stage of DDVI
python train.py dataset=mnist model=ddvi_simplified prior=pinwheel seed=0
# DDVI with the x-conditional reverse process, initialized from the run above
python train.py dataset=mnist model=ddvi prior=pinwheel seed=0
# exact ELBO of the trained model
python evaluate_elbo.py runs/mnist/ddvi/pinwheel/seed0
```

Each run writes to `runs/<dataset>/<model>/<prior>/seed<seed>/`: `config.yaml`, `train.log`, `model.pt`,
`metrics.json` (test-set KNN accuracy, latent NLL and MMD), `latent.npy` / `latent.png` (test-set latents) and, after
`evaluate_elbo.py`, `elbo.json`. `python aggregate.py runs` prints mean ± 95% t-interval over seeds for every
dataset, method and prior.

Options are Hydra overrides, e.g. `model.kld_weight=0.5`, `model.num_epochs=50`, `run_root=/scratch/runs`.

## Reproducing the paper

| Paper | Script | Seeds |
|---|---|---|
| Table 1: unsupervised MNIST | `scripts/table1_mnist.sh` | 3 |
| Table 2: semi-supervised MNIST (1,000 labels) | `scripts/table2_mnist_semi.sh` | 10 |
| Table 7: unsupervised CIFAR-10 | `scripts/table7_cifar10.sh` | 3 |
| Table 8: semi-supervised CIFAR-10 (10,000 labels) | `scripts/table8_cifar10_semi.sh` | 3 |
| Table 9: DDVI without the sleep step | `scripts/table9_sleep_ablation.sh` | 3 |

The scripts run the jobs one after another; every (method, prior, seed) job is independent, except that `ddvi` starts
from the `ddvi_simplified` run with the same dataset, prior and seed. Set `LAUNCH` to a command prefix (for example
`LAUNCH="srun --gres=gpu:1"`) to run each job through a scheduler, or split the loops over GPUs.
In the semi-supervised tables, all methods with a KL weight use the same KL and classification weight: 0.01 on MNIST
(Table 2) and 0.5 on CIFAR-10 (Table 8). The unsupervised settings are in `conf/model/`.

| Config (`model=`) | Method in the paper |
|---|---|
| `aevb`, `aevb_semi` | AEVB (Gaussian posterior) |
| `aevb_iaf`, `aevb_iaf_semi` | AEVB-IAF (inverse autoregressive flow) |
| `h_aevb` | H-AEVB (hierarchical posterior) |
| `pis` | PIS (path integral sampler; our reimplementation) |
| `aaeb`, `aaeb_semi` | AAEB (adversarial autoencoder) |
| `ddvi_simplified` | simplified DDVI; first training stage of DDVI |
| `ddvi` | DDVI (unsupervised, Tables 1, 7, 9) |
| `ddvi_semi` | DDVI (semi-supervised, Tables 2, 8) |

Priors (`prior=`): `pinwheel`, `swiss_roll`, `square`. Datasets (`dataset=`): `mnist`, `mnist_semi`, `cifar10`,
`cifar10_semi`.

## Evaluation

- **Accuracy**: a 20-nearest-neighbor classifier fitted on the latents of the training images, evaluated on the test
  latents.
- **Latent NLL**: −E_{z∼p(z)}[log q(z)], with q(z) a Gaussian kernel density estimate of the test latents; it measures
  how well the aggregate posterior covers the prior.
- **MMD** between the test images and images decoded from 10,000 prior samples.
- **ELBO** (`evaluate_elbo.py`), in nats per image: E_q[log p(x|z) + log p(z) − log q(z|x)], minus KL(q(l|x) || p(l))
  in the semi-supervised setting. MNIST uses a Bernoulli decoder; CIFAR-10 a Gaussian decoder whose variance is
  fitted by maximum likelihood on the training set for each model. log p(z) is the kernel density estimate of the prior
  (`ddvi/priors.py`) that is also used in training. log q(z|x) is exact for AEVB, AEVB-IAF and DDVI (for DDVI, the
  deterministic DDIM map z = F_x(y_T) gives log q(z|x) = log q(y_T|x) − log |det ∂F_x/∂y_T|); H-AEVB and PIS use
  their auxiliary-variable and path-space bounds.

`val_split=5000` holds out 5,000 training images as a validation set that replaces the test set in training-time
evaluation, metrics and ELBO; we used it to choose the DDVI configuration.

## Code layout

```
train.py              training + metrics          ddvi/models/ddvi.py              DDVI (x-conditional reverse process)
evaluate_elbo.py      exact ELBO                  ddvi/models/ddvi_simplified.py   simplified and semi-supervised DDVI
aggregate.py          tables over seeds           ddvi/models/{vae,pis,aae}.py     baselines
conf/                 Hydra configs               ddvi/{data,priors,nets,flows,diffusion,metrics}.py
scripts/              one script per table
```

## Citation

```bibtex
@inproceedings{piriyakulkij-wang:aaai25,
  Author = {Piriyakulkij, Wasu Top and Wang, Yingheng and Kuleshov, Volodymyr},
  Booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  Title = {Denoising Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors},
  Year = {2025}}
```

## License

MIT (see `LICENSE`).
