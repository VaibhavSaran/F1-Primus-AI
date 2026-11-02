"""UI-safe foundation candidate readiness; no imports of Torch or model loading."""
from __future__ import annotations
import sys


def foundation_registry(root=None) -> dict[str, dict]:
    """A downloaded checkpoint is a local research candidate, never a promotion."""
    from ml.foundation_runtime import DEFAULT_ROOT, checkpoint_info
    common = dict(production_eligible=False, verified_on='2026-11-01')
    models = {
        'tabfm': dict(
            common, id='tabfm', label='TabFM — local research',
            role='Tabular final-race challenger',
            targets=['finishing_position', 'clean_pace'], direct_race_ranking=True,
            description=('Uses past labelled event-driver rows as context for new race '
                         'rows. Current adapter implements finishing-position regression; '
                         'pace regression is a future target, not an enabled path.'),
            next_step=('Official packaging requires Python 3.11+. This pinned PyTorch-only '
                       'source path is an unsupported Python 3.10 compatibility experiment, '
                       'using FP32 memory-mapped weights. Default weights are licensed '
                       'for non-commercial, non-production research only. Benchmark '
                       'against identical historical whole-race folds before promotion.'),
            source_url='https://github.com/google-research/tabfm',
            model_card_url='https://huggingface.co/google/tabfm-1.0.0-pytorch'),
        'timesfm': dict(
            common, id='timesfm', label='TimesFM 2.5 — pace research',
            role='Auxiliary lap-pace time-series forecast',
            targets=['pace_sequence'], direct_race_ranking=False,
            description=('Forecasts the next five clean lap times from each driver’s '
                         'latest eligible contiguous pre-race stint. Median projected '
                         'pace gives an explicitly uncalibrated race-ranking heuristic.'),
            next_step=('TimesFM 2.5 uses Apache-2.0 weights and has verified CPU inference '
                       'in primusEnv. At least three contiguous clean laps are required; '
                       'short contexts are padded and need held-out evaluation. TimesFM 3 '
                       'is a separate multivariate candidate whose downloaded weights '
                       'have non-commercial, non-production restrictions.'),
            source_url='https://github.com/google-research/timesfm',
            model_card_url='https://huggingface.co/google/timesfm-2.5-200m-pytorch'),
    }
    for model_id, model in models.items():
        info = checkpoint_info(model_id, root if root is not None else DEFAULT_ROOT)
        model.update(available=info['available'],
                     status='local_research_ready' if info['available'] else 'not_configured',
                     reason=('Ready for explicit offline operator evaluation; UI forecasts '
                             'require a matching saved run.' if info['available'] else
                             'Local checkpoint/dependencies unavailable: ' + info['reason']),
                     weights_license=('non-commercial, non-production' if model_id == 'tabfm'
                                      else 'Apache-2.0'),
                     checkpoint=info)
        if model_id=='tabfm' and sys.version_info >= (3,11):
            model['next_step']=('The installed Python satisfies the official Python 3.11+ requirement. '
                'Pinned official PyTorch source and memory-mapped weights are used for local research. '
                'The current official checkpoint is v1.0, not v3.0. Compare on the same held-out races; '
                'pretrained-data overlap and calibrated winner probabilities remain unverified.')
    return models
