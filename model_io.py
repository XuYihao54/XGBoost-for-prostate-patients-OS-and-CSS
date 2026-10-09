"""Frozen encoding, early-stopping limits and probability mapping."""
from pathlib import Path
import json, logging, math
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
LOG=logging.getLogger(__name__)
def load_models():
    try:
        meta=json.loads((HERE/'model_metadata.json').read_text(encoding='utf-8'));models={}
        for e,s in meta.items():
            # Python reads the bytes to avoid Windows non-ASCII path failures.
            b=xgb.Booster();b.load_model(bytearray((HERE/f'{e}-xgboost_model.json').read_bytes()))
            if b.num_features()!=len(s['feature_order']): raise ValueError(f'{e}: feature count mismatch')
            if int(b.attr('best_iteration'))+1!=s['prediction_rounds']: raise ValueError(f'{e}: early-stopping metadata mismatch')
            models[e]=b
        return models,meta
    except (OSError,ValueError,TypeError,KeyError,xgb.core.XGBoostError) as exc:
        LOG.error('Local model loading failed (%s): %s',type(exc).__name__,str(exc).splitlines()[0])
        raise RuntimeError('Cannot load local JSON models. Check both model files and model_metadata.json. See README.md for the verified offline environment.') from None
def platt_scale(raw_probability,intercept,coefficient):
    # glm(y ~ p_raw): do not transform p_raw to logit.
    z=float(intercept)+float(coefficient)*float(raw_probability)
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))
def predict(inputs,endpoint,booster,spec):
    values=[]
    for f in spec['feature_order']:
        v=inputs[f]
        if f in spec['display_levels_1_based']: v=spec['display_levels_1_based'][f].index(v)+1
        else:
            v=float(v);lo,hi=spec['numeric_bounds'][f]
            if not math.isfinite(v) or not lo<=v<=hi: raise ValueError(f'{f}: expected {lo} to {hi}')
        values.append(v)
    d=xgb.DMatrix(np.array([values],dtype=np.float32))
    raw=float(booster.predict(d,iteration_range=(0,spec['prediction_rounds']))[0])
    c=spec['calibrator'];p=platt_scale(raw,c['intercept'],c['coefficient'])
    return dict(endpoint=endpoint,raw_probability=raw,mortality=p,survival=1-p)
