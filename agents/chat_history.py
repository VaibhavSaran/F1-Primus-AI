"""Local owner-scoped conversation history; no provider calls or user paths."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from security.access import AccessDenied, authorize


class ChatHistory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS conversations(
                    chat_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    title TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS chat_turns(
                    id INTEGER PRIMARY KEY, chat_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL, question TEXT NOT NULL, reply_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS chat_turn_chat ON chat_turns(chat_id,id);
            ''')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _owned(db, principal, chat_id):
        if not isinstance(chat_id, str) or not re.fullmatch('[0-9a-f]{32}', chat_id):
            raise ValueError('invalid conversation identifier')
        row = db.execute('SELECT * FROM conversations WHERE chat_id=? AND owner_id=?',
                         (chat_id, principal.identity)).fetchone()
        if row is None:
            raise AccessDenied('conversation_not_available')
        return row

    def append_turn(self, principal, chat_id, question, reply):
        authorize(principal, 'chat')
        if not isinstance(question, str) or not 0 < len(question) <= 4000 or not isinstance(reply.get('answer'), str):
            raise ValueError('invalid conversation turn')
        record = {key: reply.get(key) for key in ('answer', 'status', 'context', 'citations')}
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode()) > 200_000:
            raise ValueError('conversation turn too large')
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as db:
            if chat_id:
                self._owned(db, principal, chat_id)
            else:
                chat_id = uuid4().hex
                title = ' '.join(question.split())[:100]
                db.execute('INSERT INTO conversations VALUES(?,?,?,?,?)',
                           (chat_id, principal.identity, title, now, now))
            db.execute('INSERT INTO chat_turns(chat_id,occurred_at,question,reply_json) VALUES(?,?,?,?)',
                       (chat_id, now, question, encoded))
            db.execute('UPDATE conversations SET updated_at=? WHERE chat_id=?', (now, chat_id))
        return chat_id

    def list(self, principal):
        authorize(principal, 'chat')
        with self._connect() as db:
            return [dict(r) for r in db.execute('''SELECT c.*,count(t.id) turn_count
                FROM conversations c LEFT JOIN chat_turns t USING(chat_id)
                WHERE owner_id=? GROUP BY c.chat_id ORDER BY updated_at DESC LIMIT 500''', (principal.identity,))]

    def markdown(self, principal, chat_id):
        authorize(principal, 'chat')
        from agents.chat import safe_markdown
        from agents.claude_provider import vetted_url
        with self._connect() as db:
            chat = self._owned(db, principal, chat_id)
            turns = db.execute('SELECT * FROM chat_turns WHERE chat_id=? ORDER BY id', (chat_id,)).fetchall()
        lines = ['# Primus AI conversation', '', safe_markdown(chat['title']), '',
                 f"Created (UTC): {chat['created_at']}", '', 'Answers retain their original context and may contain estimates or unavailable evidence.', '']
        for turn in turns:
            reply = json.loads(turn['reply_json'])
            lines += [f"## User · {turn['occurred_at']} UTC", '', safe_markdown(turn['question']), '', '## Primus AI', '', safe_markdown(reply['answer']), '']
            scope = reply.get('context') or {}
            for key in ('event_id', 'session_id', 'driver_id', 'snapshot_id', 'run_id', 'as_of_utc'):
                if scope.get(key):
                    lines += [f"{key.replace('_', ' ').title()}: {safe_markdown(str(scope[key]))}", '']
            for cite in reply.get('citations') or []:
                url = vetted_url(cite.get('url')) if cite.get('kind') == 'web' else None
                if url:
                    lines += [f"- [{safe_markdown(str(cite.get('id', 'W')))}] [{safe_markdown(cite.get('title') or 'Web source')}]({url})"]
                elif cite.get('kind') != 'web':
                    lines += [f"- [{safe_markdown(str(cite.get('id', 'L')))}] Internal data · {safe_markdown(str(cite.get('reference', 'saved evidence')))}"]
            lines.append('')
        return '\n'.join(lines)
