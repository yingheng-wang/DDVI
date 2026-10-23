'''MNIST (flattened, pixels in [0, 1]) and CIFAR-10 (normalized to [-1, 1]) loaders.
Semi-supervised datasets keep the labels of n_labels training images chosen with the run seed; the other labels are set
to n_classes (= unlabeled). With val_split > 0, that many training images (a fixed permutation, independent of the run
seed) are held out and replace the test set everywhere, so that model choices can be made without the test set.'''
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as tsfm
from torchvision.datasets import CIFAR10, MNIST

VAL_SPLIT_SEED = 12345


class LabeledSubsetDataset(Dataset):
    def __init__(self, dataset, flatten=False, use_label=False, n_labels=-1, seed=None, n_classes=10):
        self.dataset = dataset
        self.flatten = flatten
        self.use_label = use_label
        if use_label:
            self.targets = np.asarray(dataset.targets)
            if n_labels != -1:
                if seed is not None:
                    np.random.seed(seed)
                n_total = len(self.targets)
                self.targets[np.random.choice(n_total, n_total - n_labels, replace=False)] = n_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        if self.flatten:
            x = x.flatten()
        return (x, self.targets[idx]) if self.use_label else x


def _raw(cfg, train):
    if cfg.dataset.name.startswith('mnist'):
        return MNIST(cfg.data_dir, train=train, download=True, transform=tsfm.ToTensor()), True
    # the random horizontal flip is applied to every split, as for all reported CIFAR-10 results
    transform = tsfm.Compose([tsfm.RandomHorizontalFlip(), tsfm.ToTensor(), tsfm.Normalize((0.5,) * 3, (0.5,) * 3)])
    return CIFAR10(cfg.data_dir, train=train, download=True, transform=transform), False


def _loader(cfg, ds, shuffle):
    return DataLoader(ds, batch_size=cfg.model.batch_size, shuffle=shuffle, num_workers=cfg.num_workers, pin_memory=True)


def _val_indices(n, n_val):
    perm = np.random.RandomState(VAL_SPLIT_SEED).permutation(n)
    return perm[n_val:], perm[:n_val]


def get_train_loader(cfg):
    '''Shuffled training loader, with labels for *_semi datasets (without the validation images if val_split > 0).'''
    semi = cfg.dataset.name.endswith('_semi')
    raw, flat = _raw(cfg, True)
    train = LabeledSubsetDataset(raw, flat, use_label=semi, n_labels=cfg.dataset.n_labels if semi else -1,
                                 seed=cfg.seed, n_classes=cfg.dataset.n_classes)
    n_val = cfg.get('val_split', 0) or 0
    if n_val:
        train = Subset(train, _val_indices(len(train), n_val)[0])
    return _loader(cfg, train, True)


def get_eval_data(cfg):
    '''Unshuffled, unlabeled training and test (or validation) loaders, with their labels.'''
    raw_tr, flat = _raw(cfg, True)
    train = LabeledSubsetDataset(raw_tr, flat)
    train_labels = np.asarray(raw_tr.targets)
    n_val = cfg.get('val_split', 0) or 0
    if not n_val:
        raw_te = _raw(cfg, False)[0]
        return (_loader(cfg, train, False), _loader(cfg, LabeledSubsetDataset(raw_te, flat), False),
                train_labels, np.asarray(raw_te.targets))
    tr_idx, va_idx = _val_indices(len(train), n_val)
    return (_loader(cfg, Subset(train, tr_idx), False), _loader(cfg, Subset(train, va_idx), False),
            train_labels[tr_idx], train_labels[va_idx])
