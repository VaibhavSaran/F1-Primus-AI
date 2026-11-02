import pandas as pd


def test_initial_departure_does_not_claim_an_api_gap_or_fabricate_duration():
    from streamlit_app.archive_views import pit_visit_label
    assert pit_visit_label(pd.Series(dict(unmatched_exit=True,unmatched_entry=False,exit_lap=1)))=='Initial pit departure'
    assert pit_visit_label(pd.Series(dict(unmatched_exit=True,unmatched_entry=False,exit_lap=9)))=='Entry not recorded'
    assert pit_visit_label(pd.Series(dict(unmatched_exit=False,unmatched_entry=True,exit_lap=None)))=='Exit not recorded'
