from .aae import AAEB, AAEB_semi
from .ddvi import DDVI
from .ddvi_simplified import DDVISemi, DDVISimplified
from .pis import PIS
from .vae import AEVB, AEVB_IAF, AEVB_IAF_semi, AEVB_semi, H_AEVB

MODELS = {
    'aevb': AEVB, 'aevb_iaf': AEVB_IAF, 'h_aevb': H_AEVB, 'pis': PIS, 'aaeb': AAEB,
    'aevb_semi': AEVB_semi, 'aevb_iaf_semi': AEVB_IAF_semi, 'aaeb_semi': AAEB_semi,
    'ddvi_simplified': DDVISimplified, 'ddvi': DDVI, 'ddvi_semi': DDVISemi,
}


def build_model(cfg, training=True):
    return MODELS[cfg.model.name](cfg, training=training)
