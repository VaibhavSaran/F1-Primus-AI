"""Compatibility facade for versioned inference; training is an explicit offline task."""
from flows.prediction_service import generate_forecast
from ml.artifacts import ArtifactStore, ModelArtifact
from ml.training import TrainingConfig, train_model, predict_model, candidate_registry


def run_prediction(round_number=None,race_name=None, *, request=None,snapshot=None,**options):
    if request is None or snapshot is None:
        raise ValueError('Prediction requires explicit request, snapshot and fitted artifact or baseline; train-on-request was retired')
    if round_number is not None and round_number!=request.event.round_number:
        raise ValueError('round_number does not match request')
    return generate_forecast(request,snapshot,**options)
