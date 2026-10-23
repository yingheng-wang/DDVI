'''Train one model and evaluate its latent-space metrics.
    python train.py dataset=mnist model=ddvi prior=pinwheel seed=0
Writes to save_folder (default runs/<dataset>/<model><tag>/<prior>/seed<seed>): config.yaml, train.log, model.pt,
metrics.json (KNN-20 accuracy, latent NLL, MMD on the test set), latent.npy and latent.png. The ELBO is computed
afterwards from model.pt by evaluate_elbo.py.'''
import json
import logging
import os

import hydra
from omegaconf import OmegaConf

from ddvi.data import get_train_loader
from ddvi.metrics import evaluate_metrics
from ddvi.models import build_model
from ddvi.utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    os.makedirs(cfg.save_folder, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.save_folder, 'config.yaml'), resolve=True)
    log.info(f'Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')
    set_seed(cfg.seed)
    model = build_model(cfg)
    train_loader = get_train_loader(cfg)

    def callback(m, epoch):   # optional intermediate metrics (eval_every > 0)
        if cfg.eval_every and epoch % cfg.eval_every == 0 and epoch != m.num_epochs - 1:
            with open(os.path.join(cfg.save_folder, 'metrics_history.jsonl'), 'a') as f:
                f.write(json.dumps({'epoch': epoch, **evaluate_metrics(m)}) + '\n')

    model.train(train_loader, callback=callback)
    model.save(os.path.join(cfg.save_folder, 'model.pt'))
    evaluate_metrics(model, cfg.save_folder)


if __name__ == '__main__':
    main()
