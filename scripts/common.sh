# Sourced by the table scripts. Jobs run one after another from the repository root; set LAUNCH to a command prefix
# (e.g. LAUNCH="srun --gres=gpu:1") to run each job elsewhere. All (prior, seed) jobs are independent, except that
# `ddvi` starts from the `ddvi_simplified` run with the same dataset, prior and seed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PRIORS="pinwheel swiss_roll square"
train() { echo "+ python train.py $*"; ${LAUNCH:-} python train.py "$@"; }
elbo() { ${LAUNCH:-} python evaluate_elbo.py "$@"; }
