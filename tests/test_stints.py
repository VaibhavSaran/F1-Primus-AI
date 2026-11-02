import pytest


def observations():
    return [dict(driver_id='A', compound='MEDIUM', stint=1, lap_number=i,
                 tyre_age_laps=i, lap_time_s=90 + .2 * i, clock_ms=i * 90_000,
                 accurate=True, deleted=False, pit_in=False, pit_out=False, track_status='1')
            for i in range(2, 15)]


SOURCE = dict(provider='fixture', source_id='validated-fixture', quality_status='validated',
              clock_basis='session_relative_observed')


def test_fit_is_prefix_only_and_recovers_known_slope():
    from ml.stints import fit_stint_model, forecast_stint
    laps = observations()
    fit = fit_stint_model(laps, clock_ms=900_000, source=SOURCE, seed=9)
    model = fit.models[0]
    assert model.sample_count == 9
    assert model.slope_s_per_lap == pytest.approx(.2)
    assert forecast_stint(model, tyre_age_laps=10, horizon_laps=2).median_s == pytest.approx((92.2, 92.4))
    changed = [dict(lap, lap_time_s=5_000) if lap['clock_ms'] > 900_000 else lap for lap in laps]
    assert fit == fit_stint_model(changed, clock_ms=900_000, source=SOURCE, seed=9)


def test_unknown_quality_and_ineligible_laps_cannot_fit():
    from ml.stints import fit_stint_model
    assert not fit_stint_model(observations(), clock_ms=2_000_000, source={}).models
    for field, value in [('deleted', None), ('deleted', True), ('pit_in', True),
                         ('pit_out', True), ('accurate', False), ('track_status', '4')]:
        fit = fit_stint_model([dict(lap, **{field: value}) for lap in observations()],
                              clock_ms=2_000_000, source=SOURCE)
        assert not fit.models
        assert fit.reasons


def test_driver_compound_stint_are_separate_and_sparse_is_insufficient():
    from ml.stints import fit_stint_model
    laps = observations()
    other = [dict(lap, driver_id='B', stint=2, compound='HARD', lap_time_s=100 + .4 * lap['tyre_age_laps'])
             for lap in laps]
    sparse = [dict(lap, stint=3, tyre_age_laps=1, lap_number=lap['lap_number'] + 20,
                   clock_ms=lap['clock_ms'] + 1_000_000) for lap in laps[:3]]
    fit = fit_stint_model(laps + other + sparse, clock_ms=2_000_000, source=SOURCE)
    assert {(m.driver_id, m.compound, m.stint) for m in fit.models} == {('A', 'MEDIUM', 1), ('B', 'HARD', 2)}
    assert fit.reasons


def test_robust_fit_and_uncertainty_are_reproducible():
    from ml.stints import fit_stint_model, forecast_stint
    laps = observations()
    laps[4]['lap_time_s'] += 20
    fit = fit_stint_model(laps, clock_ms=2_000_000, source=SOURCE, seed=23)
    assert fit.models[0].slope_s_per_lap == pytest.approx(.2)
    assert fit.models[0].sample_count == len(laps) - 1
    forecast = forecast_stint(fit.models[0], tyre_age_laps=15, horizon_laps=3)
    assert forecast.calibrated is False
    assert all(low <= median <= high for low, median, high in zip(forecast.lower_s, forecast.median_s, forecast.upper_s))
    assert fit == fit_stint_model(laps, clock_ms=2_000_000, source=SOURCE, seed=23)


@pytest.mark.parametrize('kwargs', [dict(clock_ms=-1), dict(min_laps=1), dict(bootstrap_samples=2001), dict(seed=True)])
def test_fit_bounds(kwargs):
    from ml.stints import fit_stint_model
    arguments = dict(clock_ms=2_000_000, source=SOURCE)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        fit_stint_model(observations(), **arguments)
