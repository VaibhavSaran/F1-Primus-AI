"""Fixed forecast -> evidence report -> immutable persistence orchestration.

No LLM, API client or tool is instantiated on import. Read-only Ask F1 has its
own guarded service; numeric predictions never depend on model tool ordering.
"""
from __future__ import annotations
from typing import Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from flows.prediction_service import generate_forecast


class ForecastState(TypedDict,total=False):
    request: Any
    snapshot: Any
    options: dict
    result: dict
    report: str


from flows.reporting import forecast_report


def build_forecast_graph():
    def predict(state):
        options={key:value for key,value in state['options'].items() if key!='store'}
        return {'result':generate_forecast(state['request'],state['snapshot'],**options)}
    def explain(state):
        return {'report':forecast_report(state['result'])}
    def persist(state):
        result=dict(state['result'])
        store=state['options'].get('store')
        if store is not None and result['ranking']:
            store.save_forecast(result,is_public=False)
        return {'result':result}
    graph=StateGraph(ForecastState)
    graph.add_node('forecast',predict)
    graph.add_node('evidence_report',explain)
    graph.add_node('persist',persist)
    graph.add_edge(START,'forecast')
    graph.add_edge('forecast','evidence_report')
    graph.add_edge('evidence_report','persist')
    graph.add_edge('persist',END)
    return graph.compile()


def run_pre_race_agent(round_number=None, *, request=None, snapshot=None, **options):
    """Compatibility entrypoint with explicit event/cutoff/snapshot requirements."""
    if request is None or snapshot is None:
        raise ValueError('An explicit PredictionRequest and verified DataSnapshot are required; round-only training was retired')
    if round_number is not None and round_number!=request.event.round_number:
        raise ValueError('round_number does not match explicit request')
    return build_forecast_graph().invoke({'request':request,'snapshot':snapshot,'options':options})['result']
