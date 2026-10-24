# Reproducing the Paper

## Tables and scripts

| Paper | Script | Seeds |
|---|---|---|
| Table 1: unsupervised MNIST | [`scripts/table1_mnist.sh`](../scripts/table1_mnist.sh) | 3 |
| Table 2: semi-supervised MNIST (1,000 labels) | [`scripts/table2_mnist_semi.sh`](../scripts/table2_mnist_semi.sh) | 10 |
| Table 7: unsupervised CIFAR-10 | [`scripts/table7_cifar10.sh`](../scripts/table7_cifar10.sh) | 3 |
| Table 8: semi-supervised CIFAR-10 (10,000 labels) | [`scripts/table8_cifar10_semi.sh`](../scripts/table8_cifar10_semi.sh) | 3 |
| Table 9: DDVI without the sleep step | [`scripts/table9_sleep_ablation.sh`](../scripts/table9_sleep_ablation.sh) | 3 |

Each script trains every (method, prior, seed), computes the ELBO of the trained runs and prints the tables with
`aggregate.py`. The numbers reported in the paper are in [`results/paper_tables.json`](../results/paper_tables.json)
(mean and 95% t-interval half-width over seeds).

The jobs run one after another. They are independent, except that `ddvi` starts from the `ddvi_simplified` run with
the same dataset, prior and seed. Set `LAUNCH` to a command prefix (e.g. `LAUNCH="srun --gres=gpu:1"`) to send each job
to a scheduler, or split the loops over GPUs. On a single V100 or A100, a run takes about 0.3–2 hours for the
baselines, 2–3 hours for the two stages of unsupervised DDVI, and 7–11 hours for semi-supervised DDVI (Appendix H of
the paper).

## Settings

The model configs in [`conf/model/`](../conf/model) hold the paper's hyperparameters; the scripts only add the
dataset-specific overrides:
- **Semi-supervised tables.** All methods with a KL weight use the same KL and classification weight: 0.01 on MNIST
  (Table 2) and 0.5 on CIFAR-10 (Table 8, `model.kld_weight=0.5`). AAEB has neither weight.
- **AAEB** is trained for 200 epochs on MNIST and 100 epochs on CIFAR-10 (`model.num_epochs=100`).
- **DDVI** (`model=ddvi`) is trained in two stages: `ddvi_simplified` (100 epochs, x-independent reverse process),
  then `ddvi` (100 epochs, x-conditional reverse process, wake-sleep) initialized from it via `model.init_from`.
- **Table 9** is `model=ddvi model.sleep_ct=0` (no sleep step), saved under the tag `_nosleep`.

Other useful options:
- `model.denoiser_dir=<run folder>` reuses the warmed-up prior denoisers (`diff_model.pt`, or `diff_model<l>.pt` for
  the ten partitions of `ddvi_semi`) of an earlier run with the same prior instead of training new ones.
- `val_split=5000` holds out 5,000 training images as a validation set that replaces the test set in training-time
  evaluation, metrics and ELBO. We used it to choose the DDVI configuration.
- `eval_every=k` also computes the metrics every k epochs (`metrics_history.jsonl`).

## Outputs

Each run writes to `runs/<dataset>/<model><tag>/<prior>/seed<seed>/`:

| File | Content |
|---|---|
| `config.yaml` | the resolved config |
| `train.log` | training log |
| `model.pt` | the trained networks |
| `metrics.json` | test-set KNN accuracy, latent NLL and MMD |
| `latent.npy`, `latent.png` | test-set latents |
| `elbo.json` | ELBO and its terms (written by `evaluate_elbo.py`) |

`python aggregate.py runs` prints mean ± 95% t-interval over seeds for every dataset, method and prior, and
`--json OUT.json` also writes the per-seed values.

## Evaluation

- **Acc**: a 20-nearest-neighbor classifier fitted on the latents and labels of all training images, evaluated on the
  test latents.
- **Latent NLL**: −E_{z∼p(z)}[log q(z)], with q(z) a Gaussian kernel density estimate of the test latents (mixture over
  bandwidths 0.05, 0.1, 0.3, 0.5, 0.8) scored at 10,000 prior samples. It measures how well the aggregate posterior
  covers the prior.
- **MMD** between the test images and images decoded from 10,000 prior samples (sum of RBF kernels with scales 2, 5,
  10, 20, 40, 80).
- **ELBO** (`evaluate_elbo.py`), in nats per image: E_q[log p(x|z) + log p(z) − log q(z|x)], minus KL(q(l|x) || p(l))
  in the semi-supervised setting, with one posterior sample per test image.
  - log p(x|z): Bernoulli decoder for MNIST; Gaussian decoder for CIFAR-10 with one variance per trained model, fitted
    by maximum likelihood on the training set.
  - log p(z): the kernel density estimate of the prior (`ddvi/priors.py`) that is also used in training.
  - log q(z|x): Gaussian / flow density for AEVB and AEVB-IAF; for DDVI, the deterministic DDIM map z = F_x(y_T) gives
    log q(z|x) = log q(y_T|x) − log |det ∂F_x/∂y_T| (2 × 2 Jacobian by autograd); H-AEVB and PIS use their
    auxiliary-variable and path-space bounds.
  - As in the paper, the values follow common conventions under which they are not guaranteed lower bounds on
    log p(x): the MNIST Bernoulli likelihood is applied to grey-level pixels, the prior KDE log-density is clipped, the
    DDIM map is assumed to be injective, and the terms are conditional on the dropout masks where dropout is active at
    evaluation.

Two details of the evaluation match the runs behind the paper: dropout in the baseline networks (and in the
semi-supervised DDVI) stays active at evaluation, and the random horizontal flip of CIFAR-10 is applied to every
split.
