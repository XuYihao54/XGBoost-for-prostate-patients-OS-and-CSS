"""Offline serialization upgrade. No training, tuning or tree removal."""
from pathlib import Path
import hashlib, json
import xgboost as xgb
HERE = Path(__file__).resolve().parent
def main():
    rows = []
    for e in ('OS', 'CSS'):
        source=HERE/f'{e}-xgboost_model.model';target=HERE/f'{e}-xgboost_model.json'
        b=xgb.Booster()
        try: b.load_model(bytearray(source.read_bytes()))
        except xgb.core.XGBoostError:
            raise RuntimeError('Migrate with the locally verified XGBoost 3.0.0 environment.') from None
        before=b.get_dump(dump_format='json');b.save_model(target)
        c=xgb.Booster();c.load_model(bytearray(target.read_bytes()))
        same=before==c.get_dump(dump_format='json') and b.attributes()==c.attributes()
        if not same: raise RuntimeError(f'{e}: model contents changed; migration rejected')
        rows.append(dict(endpoint=e,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
          json_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),features=c.num_features(),
          stored_rounds=c.num_boosted_rounds(),attributes=c.attributes(),all_trees_and_attributes_unchanged=same))
    (HERE/'migration_manifest.json').write_text(json.dumps(dict(xgboost=xgb.__version__,models=rows),indent=2),encoding='utf-8')
    print('Both JSON models saved; all original trees and attributes preserved.')
if __name__=='__main__': main()
