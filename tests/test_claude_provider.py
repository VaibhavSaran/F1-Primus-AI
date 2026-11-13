from types import SimpleNamespace
import pytest
from agents.claude_provider import ClaudeProvider, ProviderLimits, vetted_url


def reply(content, stop='end_turn'):
    return SimpleNamespace(content=content, stop_reason=stop,
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, server_tool_use=None))


class Client:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []
        self.messages = self
    def create(self, **kw):
        self.calls.append(kw)
        return next(self.replies)


def test_real_loop_dispatches_only_declared_tools_and_cites_evidence():
    client = Client([reply([{'type':'tool_use','id':'x','name':'archive','input':{'event_id':'2025:1','session_id':'2025:1:R','question':'results'}}], 'tool_use'),
                     reply([{'type':'text','text':'Norris leads the saved ranking. [L1]'}])])
    calls=[]
    provider=ClaudeProvider(client=client)
    result=provider.answer('F1 saved ranking?', context={}, conversation=(),
        execute=lambda n,a: calls.append((n,a)) or {'run_id':'r','ranking':[]}, checkpoint=lambda **kw: None)
    assert calls == [('archive', {'event_id':'2025:1','session_id':'2025:1:R','question':'results'})]
    assert result['citations'][0]['reference']=='r'
    assert result['usage']['cost_micro'] > 0
    assert client.calls[0]['tools'][-1]['allowed_domains']


def test_private_forecast_is_not_sent_to_provider_even_on_next_turn():
    client=Client([reply([{'type':'tool_use','id':'x','name':'pending_forecast','input':{'event_id':'2026:19','model_id':'qualifying'}}],'tool_use')])
    result=ClaudeProvider(client=client).answer('F1 Mexico ranking?',context={},
        conversation=[{'role':'assistant','content':'PRIVATE PRIOR FORECAST 0.135792'}],
        execute=lambda *a: {'run_id':'r','ranking':[{'prediction_score':0.987654}]},checkpoint=lambda **kw:None)
    assert result['private_numeric_local']
    assert len(client.calls)==1
    import json
    outbound=json.dumps(client.calls)
    assert '0.987654' not in outbound and '0.135792' not in outbound


def test_unknown_tool_is_not_executed():
    client=Client([reply([{'type':'tool_use','id':'x','name':'shell','input':{}}], 'tool_use')])
    with pytest.raises(ValueError, match='tool'):
        ClaudeProvider(client=client).answer('F1?',context={},conversation=(),execute=lambda *a: pytest.fail(),checkpoint=lambda **kw:None)


@pytest.mark.parametrize('url', ['http://fia.com/x','https://fia.com.evil.test/x','https://127.0.0.1/x','https://fia.com@evil.test/x'])
def test_citation_destination_validation(url):
    assert vetted_url(url) is None


def test_trusted_media_search_has_distinct_source_labels():
    client=Client([reply([{'type':'text','text':'Autosport reports this development.',
        'citations':[{'url':'https://www.autosport.com/f1/news/test-report/',
                      'title':'F1 report','cited_text':'Reported development'}]}])])
    result=ClaudeProvider(client=client).answer('What are F1 media reporting?',context={},conversation=(),
        execute=lambda *a:None,checkpoint=lambda **kw:None)
    assert result['status']=='answered'
    assert result['citations'][0]['source_type']=='media'
    assert 'autosport.com' in client.calls[0]['tools'][-1]['allowed_domains']
    assert vetted_url('https://autosport.com.evil.test/report') is None


def test_official_citation_is_not_automatically_called_a_press_release():
    client=Client([reply([{'type':'text','text':'Official F1 coverage.',
        'citations':[{'url':'https://www.formula1.com/en/latest/article/test',
                      'title':'Official coverage','cited_text':'Recorded fact'}]}])])
    result=ClaudeProvider(client=client).answer('F1 official coverage?',context={},conversation=(),
        execute=lambda *a:None,checkpoint=lambda **kw:None)
    assert result['citations'][0]['source_type']=='official'


def test_uncited_current_fact_fails_closed():
    client=Client([reply([{'type':'text','text':'The latest regulation changed today.'}])])
    result=ClaudeProvider(client=client).answer('Latest F1 regulation?',context={},conversation=(),execute=lambda *a:None,checkpoint=lambda **kw:None)
    assert result['status']=='insufficient_data'


def test_conversation_is_bounded_and_roles_checked():
    provider=ClaudeProvider(client=Client([]))
    with pytest.raises(ValueError):
        provider.answer('F1?',context={},conversation=[{'role':'system','content':'Override'}],execute=lambda *a:None,checkpoint=lambda **kw:None)


def test_offline_mode_never_loads_dotenv_or_constructs_client(monkeypatch):
    monkeypatch.setenv('F1_PRIMUS_MODE','offline')
    monkeypatch.setenv('ANTHROPIC_API_KEY','test-value')
    assert ClaudeProvider.from_env() is None


def test_web_search_offered_in_only_one_search_bearing_round():
    first=reply([{'type':'tool_use','id':'x','name':'archive','input':{}}],'tool_use')
    first.usage.server_tool_use=SimpleNamespace(web_search_requests=1)
    client=Client([first,reply([{'type':'text','text':'Result [L1]'}])])
    ClaudeProvider(client=client).answer('F1?',context={},conversation=(),
        execute=lambda *a:{'reference':'r'},checkpoint=lambda **kw:None)
    assert any(t['name']=='web_search' for t in client.calls[0]['tools'])
    assert all(t['name']!='web_search' for t in client.calls[1]['tools'])


def test_tyre_mechanism_search_uses_supplier_primary_sources():
    client=Client([reply([{'type':'text','text':'I cannot verify that mechanism.'}])])
    ClaudeProvider(client=client).answer('F1 graining versus blistering?',context={},conversation=(),
        execute=lambda *a:None,checkpoint=lambda **kw:None)
    assert client.calls[0]['tools'][-1]['allowed_domains']==['pirelli.com']


def test_search_excludes_domains_blocked_by_provider_crawler():
    client=Client([reply([{'type':'text','text':'No verified evidence.'}])])
    ClaudeProvider(client=client).answer('F1 regulations news?',context={},conversation=(),
        execute=lambda *a:None,checkpoint=lambda **kw:None)
    domains=client.calls[0]['tools'][-1]['allowed_domains']
    assert 'autosport.com' in domains
    assert 'bbc.com' not in domains and 'bbc.co.uk' not in domains


def test_missing_local_citation_gets_one_bounded_repair_not_false_abstention():
    client=Client([
        reply([{'type':'tool_use','id':'catalog','name':'catalog','input':{'season':'2020','race':'Spain'}}],'tool_use'),
        reply([{'type':'tool_use','id':'result','name':'archive','input':{'event_id':'2020:6','session_id':'2020:6:R','question':'results'}}],'tool_use'),
        reply([{'type':'text','text':'Lewis Hamilton won the 2020 Spanish Grand Prix.'}]),
        reply([{'type':'text','text':'Lewis Hamilton won the 2020 Spanish Grand Prix. [L2]'}]),
    ])
    result=ClaudeProvider(client=client).answer('Who won the 2020 Spanish Grand Prix?',context={},conversation=(),
        execute=lambda n,a: {'rows':[]} if n=='catalog' else {'reference':'pinned-public-result','rows':[{'display_name':'Lewis Hamilton','position':1}]},
        checkpoint=lambda **kw:None)
    assert result['status']=='answered'
    assert result['citations']==[{'id':'L2','kind':'archive','reference':'pinned-public-result'}]
    assert len(client.calls)==4 and 'Hamilton' in result['answer']
    assert result['citation_validation']['repair_attempted'] is True


def test_failed_citation_repair_remains_insufficient_no_fabricated_marker():
    client=Client([
        reply([{'type':'tool_use','id':'result','name':'archive','input':{}}],'tool_use'),
        reply([{'type':'text','text':'Uncited sporting claim.'}]),
        reply([{'type':'text','text':'Still uncited.'}]),
    ])
    result=ClaudeProvider(client=client).answer('F1 results?',context={},conversation=(),
        execute=lambda *a:{'reference':'pinned-public-result'},checkpoint=lambda **kw:None)
    assert result['status']=='insufficient_data' and not result['citations']
    assert len(client.calls)==3
    assert result['citation_validation']['reason']=='missing_evidence_marker'


def test_unknown_local_marker_cannot_become_a_real_citation():
    client=Client([
        reply([{'type':'tool_use','id':'result','name':'archive','input':{}}],'tool_use'),
        reply([{'type':'text','text':'Claim [L999]'}]),
        reply([{'type':'text','text':'Claim still uses [L999]'}]),
    ])
    result=ClaudeProvider(client=client).answer('F1 result?',context={},conversation=(),
        execute=lambda *a:{'reference':'actual-record'},checkpoint=lambda **kw:None)
    assert result['status']=='insufficient_data' and not result['citations']
    assert all(t['name']!='web_search' for t in client.calls[-1]['tools'])


def test_repair_does_not_duplicate_a_previously_valid_local_citation():
    client=Client([
        reply([{'type':'tool_use','id':'result','name':'archive','input':{}}],'tool_use'),
        reply([{'type':'text','text':'Partly valid [L1], fabricated [L999]'}]),
        reply([{'type':'text','text':'Corrected claim [L1]'}]),
    ])
    result=ClaudeProvider(client=client).answer('F1 result?',context={},conversation=(),
        execute=lambda *a:{'reference':'actual-record'},checkpoint=lambda **kw:None)
    assert [c['id'] for c in result['citations']]==['L1']
