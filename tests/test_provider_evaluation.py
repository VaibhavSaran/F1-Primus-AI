from agents.provider_evaluation import grade


def test_relevant_words_in_answer_do_not_pass_wrong_source():
    response={'status':'answered','answer':'Graining and blistering [W1]',
              'citations':[dict(id='W1',kind='web',url='https://www.fia.com/safety',title='Safety',cited_text='A safety car was used.')]}
    result=grade(dict(require=('graining','blistering'),source_terms=('tyre','graining')),response)
    assert not result['passed'] and not result['checks']['source_relevance']


def test_untrusted_citation_domain_fails_quality_check():
    response={'status':'answered','answer':'Tyre graining [W1]',
              'citations':[dict(id='W1',kind='web',url='https://example.com/tyres',title='Tyre graining',cited_text='Tyre graining') ]}
    assert not grade(dict(source_terms=('tyre','graining')),response)['passed']


def test_judge_never_receives_forecast_payload():
    from agents.provider_evaluation import judge_public_web_case
    assert judge_public_web_case(None,None,None,{'id':'forecast'},
                                {'status':'answered','answer':'PRIVATE NUMBERS'}) is None


def test_judge_grounding_summary_cannot_hide_unsupported_finding(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agents.provider_evaluation import judge_public_web_case
    from security.access import Principal
    from security.limits import SQLiteLimiter
    monkeypatch.setattr('agents.provider_evaluation.fetch_public_article',lambda url:{'url':url,'status':'retrieved','text':'Public source text'})
    limiter=SQLiteLimiter(tmp_path/'judge.sqlite3')
    limiter.configure_budget(Principal('ops','operator'),daily_micro=500000,monthly_micro=500000)
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=100,output_tokens=100),
            content=[SimpleNamespace(type='tool_use',name='record_judgment',input=dict(answered_question=True,
                source_relevance=True,source_quality=True,year_race_session_match=True,citation_grounding=True,
                findings=['The FIA timetable assertion is uncited.']))])
    provider=SimpleNamespace(model_id='claude-sonnet-5-5',client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    judgment=judge_public_web_case(provider,limiter,Principal('local','ai_user'),
        dict(id='no-sprint',prompt='Public F1 question'),dict(status='answered',answer='Public answer',citations=[]))
    assert not judgment['citation_grounding'] and not judgment['passed']
    assert 'tool_choice' not in calls[0]
    assert calls[0]['thinking']=={'type':'between_tools'}
