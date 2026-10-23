'''Collect metrics.json / elbo.json of all runs under RUN_ROOT (<dataset>/<model>/<prior>/seed<k>) and print one table per
dataset: mean +- 95% t-interval over seeds of the ELBO, KNN-20 accuracy, latent NLL and MMD.
Usage: python aggregate.py [RUN_ROOT=runs] [--json OUT.json]'''
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ORDER = ['aevb', 'aevb_iaf', 'aaeb', 'h_aevb', 'pis', 'ddvi_simplified', 'ddvi',
         'aevb_semi', 'aevb_iaf_semi', 'aaeb_semi', 'ddvi_semi']
PRIORS = ['pinwheel', 'swiss_roll', 'square']
METRICS = [('elbo', 'ELBO', 1), ('acc', 'Acc', 3), ('latent_nll', 'NLL', 2), ('mmd', 'MMD', 2)]


def ci(v, digits):
    v = np.asarray(v, float)
    if len(v) == 0:
        return '–'
    if len(v) == 1:
        return f'{v[0]:.{digits}f}'
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f'{v.mean():.{digits}f} ± {h:.{digits}f}'


def collect(root):
    out = {}
    for seed_dir in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'seed*'))):
        dataset, model, prior = seed_dir.split(os.sep)[-4:-1]
        vals = {}
        for fname in ('metrics.json', 'elbo.json'):
            p = os.path.join(seed_dir, fname)
            if os.path.exists(p):
                vals.update(json.load(open(p)))
        if vals:
            out.setdefault(dataset, {}).setdefault(model, {}).setdefault(prior, []).append(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='runs')
    ap.add_argument('--json', default=None, help='also write the per-seed values to this file')
    args = ap.parse_args()
    res = collect(args.root)
    summary = {}
    for dataset, models in res.items():
        priors = [p for p in PRIORS if any(p in v for v in models.values())]
        names = sorted(models, key=lambda m: (ORDER.index(m) if m in ORDER else len(ORDER), m))
        head = ['Method', 'Seeds'] + [f'{p} {label}' for p in priors for _, label, _ in METRICS]
        print(f'\n### {dataset}\n')
        print('| ' + ' | '.join(head) + ' |')
        print('|' + '---|' * len(head))
        for m in names:
            row = [m, '/'.join(str(len(models[m].get(p, []))) for p in priors)]
            for p in priors:
                runs = models[m].get(p, [])
                for key, _, digits in METRICS:
                    v = [r[key] for r in runs if key in r]
                    row.append(ci(v, digits))
                    summary.setdefault(dataset, {}).setdefault(m, {}).setdefault(p, {})[key] = v
            print('| ' + ' | '.join(row) + ' |')
    if args.json:
        json.dump(summary, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
