import pytest
from security.access import Principal, AccessDenied


def test_durable_chat_is_owner_scoped_and_exports_context_and_sources(tmp_path):
    from agents.chat_history import ChatHistory
    path=tmp_path/'chat.sqlite3'
    alice=Principal('alice','ai_user');bob=Principal('bob','ai_user')
    store=ChatHistory(path)
    chat=store.append_turn(alice,None,'How did Kimi compare?',
        dict(answer='Kimi was third. [L1]',status='answered',context=dict(event_id='2026:7',session_id='2026:7:Q'),
             citations=[dict(id='L1',kind='archive',reference='saved-result'),
                        dict(id='W1',kind='web',url='https://www.formula1.com/en/results',title='Official results')]))
    store=ChatHistory(path)
    assert len(store.list(alice))==1 and store.list(bob)==[]
    with pytest.raises(AccessDenied):store.markdown(bob,chat)
    with pytest.raises(AccessDenied):store.append_turn(bob,chat,'steal',dict(answer='x'))
    text=store.markdown(alice,chat)
    assert all(s in text for s in ('How did Kimi','Kimi was third','2026:7:Q','Internal data','https://www.formula1.com/en/results','UTC'))
    assert store.append_turn(alice,chat,'And weather?',dict(answer='Unknown',status='insufficient_data'))==chat
    assert store.list(alice)[0]['turn_count']==2
