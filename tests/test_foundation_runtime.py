from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import json
from hashlib import sha256
from types import SimpleNamespace

from ml.foundation_runtime import (FoundationUnavailable, PaceSeries, REVISIONS,
                                   TabFMAdapter, TimesFMAdapter, checkpoint_info)

CUTOFF = datetime(2025, 10, 19, tzinfo=timezone.utc)


def rows():
    history = [dict(event_id='2025:18', driver_id=str(i), actual_position=i + 1,
                    outcome_available_at_utc=CUTOFF - timedelta(days=1), grid_position=i + 1)
               for i in range(3)]
    target = [dict(event_id='2025:19', driver_id=str(i), grid_position=i + 1) for i in range(2)]
    return history, target


class Regressor:
    def fit(self, x, y):
        assert list(x.columns) == ['grid_position']
        assert list(y) == [1, 2, 3]

    def predict(self, x):
        return np.array([2.1, 1.4])


def test_tabfm_real_estimator_contract_and_no_label_columns(monkeypatch, tmp_path):
    adapter = TabFMAdapter(tmp_path)
    monkeypatch.setattr(adapter, '_estimator', lambda: Regressor())
    result = adapter.predict(*rows(), feature_columns=['grid_position'],
                             cutoff_utc=CUTOFF, target_event_id='2025:19')
    assert [r['driver_id'] for r in result] == ['1', '0']
    assert [r['predicted_rank'] for r in result] == [1, 2]
    assert all(r['event_id'] == '2025:19' for r in result)
    assert all('win_probability' not in r for r in result)


@pytest.mark.parametrize('bad', ['same_race', 'future_label', 'naive', 'label_feature', 'duplicate'])
def test_tabfm_rejects_leakage_before_loading(monkeypatch, tmp_path, bad):
    adapter = TabFMAdapter(tmp_path)
    monkeypatch.setattr(adapter, '_estimator', lambda: pytest.fail('must validate before loading'))
    history, target = rows()
    features = ['grid_position']
    if bad == 'same_race':
        history[0]['event_id'] = '2025:19'
    elif bad == 'future_label':
        history[0]['outcome_available_at_utc'] = CUTOFF
    elif bad == 'naive':
        history[0]['outcome_available_at_utc'] = CUTOFF.replace(tzinfo=None)
    elif bad == 'label_feature':
        features = ['actual_position']
    else:
        target.append(dict(target[0]))
    with pytest.raises(ValueError):
        adapter.predict(history, target, feature_columns=features, cutoff_utc=CUTOFF,
                        target_event_id='2025:19')


class Forecaster:
    def forecast(self, *, horizon, inputs):
        assert len(inputs) == 2 and all(len(x) >= 8 for x in inputs)
        return np.array([[89., 90., 91.], [87., 88., 89.]]), None


def series():
    return [PaceSeries(str(i), '2025:19', 'FP1', CUTOFF - timedelta(hours=2),
                       tuple(90. + j / 10 for j in range(8))) for i in range(2)]


def test_timesfm_pace_forecast_ranking_is_explicit_heuristic(monkeypatch, tmp_path):
    adapter = TimesFMAdapter(tmp_path)
    monkeypatch.setattr(adapter, '_load', lambda: Forecaster())
    result = adapter.predict(series(), cutoff_utc=CUTOFF, target_event_id='2025:19',
                             driver_ids=['0', '1'], horizon=3)
    assert [r['driver_id'] for r in result] == ['1', '0']
    assert result[0]['prediction_score'] == 88.
    assert result[0]['interpretation'] == 'pace_based_ranking_heuristic'
    assert result[0]['score_units'] == 'seconds'


@pytest.mark.parametrize('bad', ['target_race', 'late', 'missing_driver', 'nan', 'short', 'horizon'])
def test_timesfm_rejects_bad_or_leaking_series_before_loading(monkeypatch, tmp_path, bad):
    adapter = TimesFMAdapter(tmp_path)
    monkeypatch.setattr(adapter, '_load', lambda: pytest.fail('must validate before loading'))
    data = series()
    item = data[0]
    changes = dict(driver_id=item.driver_id, event_id=item.event_id, session_code=item.session_code,
                   completed_at_utc=item.completed_at_utc, values=item.values)
    if bad == 'target_race':
        changes['session_code'] = 'R'
    elif bad == 'late':
        changes['completed_at_utc'] = CUTOFF
    elif bad == 'missing_driver':
        data.pop()
    elif bad == 'nan':
        changes['values'] = (float('nan'),) * 8
    elif bad == 'short':
        changes['values'] = (90.,)
    data[0] = PaceSeries(**changes)
    with pytest.raises(ValueError):
        adapter.predict(data, cutoff_utc=CUTOFF, target_event_id='2025:19', driver_ids=['0', '1'],
                        horizon=0 if bad == 'horizon' else 3)


def test_checkpoint_hash_and_revision_checks_do_not_load_backends(tmp_path, monkeypatch):
    import ml.foundation_runtime as runtime
    monkeypatch.setattr(runtime.importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(runtime.importlib.metadata, 'version', lambda name: 'test-version')
    root = tmp_path / 'timesfm'
    root.mkdir()
    files = []
    for name in ('config.json', 'model.safetensors'):
        contents = b'fixture-only-not-weights'
        (root / name).write_bytes(contents)
        files.append(dict(path=name, bytes=len(contents), sha256=sha256(contents).hexdigest()))
    manifest = dict(model_id='timesfm', revision=REVISIONS['timesfm'], files=files)
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    info = checkpoint_info('timesfm', tmp_path, verify_hashes=True)
    assert info['available'] is True and info['weights_license'] == 'Apache-2.0'
    assert info['package_versions']['torch'] == 'test-version'
    (root / 'model.safetensors').write_bytes(b'corrupted-same-size!!!!')
    assert checkpoint_info('timesfm', tmp_path, verify_hashes=True)['available'] is False


def test_low_memory_stops_before_any_model_loading(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=0))
    with pytest.raises(FoundationUnavailable, match='free RAM'):
        TimesFMAdapter(tmp_path)._load()
