'''CPU checks of the release: every model builds from its config, encodes, computes its ELBO terms, takes a training
step and survives a save/load round trip, on random tensors (no dataset download).
Run from the repository root: python -m unittest discover -s tests -v'''
import os
import tempfile
import unittest

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from ddvi.models import MODELS, build_model
from ddvi.utils import get_kld_weights

CONF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'conf')
DATASETS = ['mnist', 'mnist_semi', 'cifar10', 'cifar10_semi']


def make_cfg(dataset, model, prior='square'):
    with initialize_config_dir(config_dir=CONF, version_base=None):
        overrides = [f'dataset={dataset}', f'model={model}', f'prior={prior}', 'save_folder=unused']
        if model == 'ddvi':
            overrides.append('model.pretrain_iters=0')
        return compose(config_name='config', overrides=overrides)


def random_batch(dataset, n=8):
    x = torch.rand(n, 784) if dataset.startswith('mnist') else torch.rand(n, 3, 32, 32) * 2 - 1
    if dataset.endswith('_semi'):
        return x, torch.tensor([0, 1, 10, 10, 2, 10, 3, 10][:n])   # label 10 = unlabeled
    return x


def pairs():
    for d in DATASETS:
        for m in MODELS:
            if m.endswith('_semi') == d.endswith('_semi'):
                yield d, m


class TestModels(unittest.TestCase):
    def test_build_encode_elbo_train_save(self):
        for dataset, model_name in pairs():
            with self.subTest(dataset=dataset, model=model_name):
                torch.manual_seed(0)
                np.random.seed(0)
                cfg = make_cfg(dataset, model_name)
                model = build_model(cfg, training=False)
                batch = random_batch(dataset)
                x = batch[0] if isinstance(batch, tuple) else batch
                model.eval_mode()
                z = model._encode(x)
                self.assertEqual(tuple(z.shape), (x.size(0), 2))
                self.assertTrue(torch.isfinite(z).all())
                if not model_name.startswith('aaeb'):
                    with torch.no_grad():
                        terms = model.elbo_terms(x)
                    for t in terms:
                        self.assertEqual(tuple(t.shape), (x.size(0),))
                        self.assertTrue(torch.isfinite(t).all())
                losses = model._train_epoch([batch])
                self.assertTrue(np.all(np.isfinite(losses)))
                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, 'model.pt')
                    model.save(path)
                    other = build_model(cfg, training=False)
                    other.load(path)
                    for name in model.ckpt_modules:
                        a, b = getattr(model, name).state_dict(), getattr(other, name).state_dict()
                        self.assertEqual(set(a), set(b))
                        for k in a:
                            self.assertTrue(torch.equal(a[k], b[k]), f'{name}.{k}')


class TestSchedules(unittest.TestCase):
    def test_kld_weights(self):
        w = get_kld_weights(0.01, 200, 'linear')
        self.assertEqual(len(w), 200)
        self.assertEqual(w[0], 0.0)
        self.assertTrue(np.all(np.diff(w[:160]) > 0))
        self.assertTrue(np.allclose(w[160:], 0.01))
        self.assertTrue(np.allclose(get_kld_weights(0.5, 50, 'constant'), 0.5))
        self.assertTrue(np.allclose(get_kld_weights(0.01, 1, 'linear'), 0.01))


class TestPriors(unittest.TestCase):
    def test_partitions(self):
        from ddvi.priors import get_prior
        for prior in ['pinwheel', 'swiss_roll', 'square']:
            with self.subTest(prior=prior):
                p = get_prior(make_cfg('mnist_semi', 'aevb_semi', prior))
                z, labels = p.sample(500, with_classes=True)
                self.assertEqual(tuple(z.shape), (500, 2))
                self.assertTrue(((labels >= 0) & (labels < 10)).all())
                self.assertTrue(torch.isfinite(p.evaluate(z[:16], nored=True)).all())
                self.assertTrue(torch.isfinite(p.evaluate_partition(z[:16], 0)).all())


if __name__ == '__main__':
    unittest.main()
