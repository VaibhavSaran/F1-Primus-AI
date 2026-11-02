# Primus AI provider and evidence tools

Primus uses native Anthropic Claude Sonnet 5.5 (`claude-sonnet-5-5`) through the installed Anthropic SDK. The Models API confirmed this identifier on 1 November 2026. The application keeps Streamlit and the existing archive, owner-scoped tracking store and shared SQLite limiter.

Set `ANTHROPIC_API_KEY` in the ignored project `.env` or the server environment. The provider loads the root `.env` without overriding environment settings. Never put a key in a chat or committed file. `F1_PRIMUS_MODE=offline` selects the existing deterministic saved-evidence service and skips dotenv/provider construction; tests set this explicitly. A missing key also selects the deterministic service.

`ChatService(..., provider=ClaudeProvider.from_env())` preserves `ask(...)`, with optional `conversation=()` and the existing optional context identifiers. Without those identifiers, Claude resolves an explicit year/race using the saved catalog. Bound context conflicts ask the user to refine the question or remove additional context. The model cannot silently select a different bound year/event/session/driver/run. Local evidence retains publication/retrieval cutoffs; current web retrieval is separate from historical local knowledge.

The fixed tools read the saved session catalog; results, laps, tyres, weather and sourced incident context; driver/team profiles; saved evaluations; and existing numeric forecasts. There is no shell, arbitrary SQL, arbitrary file reader, URL fetcher, training, archive collector or write action. Archive sessions absent locally do not prove an event was held or cancelled. The forecast tool reads a prepared owner-visible record; chat never fits models or downloads F1 data.

Saved numeric forecasts remain local. Claude can select `saved_run` or `pending_forecast`, but the provider loop immediately returns control to a local renderer without adding their payload to outbound tool results. The renderer takes ranks and scores directly from the saved record, with its run/snapshot/cutoff citation. Previous assistant answers are omitted from outbound conversation to prevent a later turn from transmitting locally rendered predictions. User questions and public historical archive projections can be sent to Claude. Do not paste secrets into questions.

Official Anthropic server web search supplies native citations. Allowed publishers are F1, FIA, Pirelli and an explicit set of official team domains. Citations retain title, URL and native supporting excerpt; unsafe destinations are removed. The model is instructed to match the question's year/race/session and prefer primary sources. A relevant domain alone does not establish claim support. Source text and user text are untrusted data and cannot add capabilities. Answers without usable citations fail closed.

Local tool results supply an explicit citation marker. If Claude omits that marker or invents a local marker despite having returned evidence, the loop allows one citation correction using those existing records. This consumes the existing round/time budget and disables further search; it never attaches a citation to unverified generated prose automatically. A failed correction remains insufficient data. `citation_validation` records the disposition without logging user prose.

The paid service uses `LimitPolicy(tools=8, searches=2, deadline_seconds=90, context_tokens=80000, output_tokens=8000)`. The last two limits use the existing conservative byte-token proxy; Claude independently has a 1,600 native output-token cap, 42,000 outbound message-byte cap, four API rounds and a 35-second request-loop budget. SDK retries are disabled. Server search is offered in only one search-bearing API round, with two uses maximum. Later rounds use bounded local context. Transport timeouts and lease checkpoints do not guarantee upstream cancellation/refunds.

The ledger reserves $3 per request, using standard Sonnet 5.5 pricing ($2/million input tokens, $10/million output tokens) and $0.01/search. This covers one search-bearing call at the model's one-million-token context limit plus three bounded calls and capped outputs. Actual token/search usage settles the reservation; unknown bills retain it for operator reconciliation. Configure the trusted local operator's daily/monthly budget once; request handlers do not reset budgets. The local configuration is $5/day and $30/month. Kill switch, owner quotas, concurrency, expiry and duplicate-request checks continue to apply. Recheck pricing/capabilities before changing the model or enabling new provider modes.

Run the focused offline tests with the project interpreter:

```powershell
& '.\primusEnv\Scripts\python.exe' -m pytest tests/test_chat.py tests/test_claude_provider.py tests/test_archive_chat.py tests/test_primus_evaluation.py tests/test_provider_evaluation.py --basetemp=.scratch/provider-tests
```

The opt-in paid evaluation uses curated public questions, a separate $5 maximum ledger and fresh ignored evidence directory:

```powershell
& '.\primusEnv\Scripts\python.exe' -m agents.provider_evaluation --live --judge-web --output .scratch/provider-live-NEW
```

It checks answers, citation markers, source domains, topic relevance of native excerpts, context mismatch and injection refusals. The local forecast case checks numeric output against the saved tracking record and verifies the local-only path. Tests additionally inspect outbound messages for forecast leakage. `--judge-web` adds a separate public-only critique of answer relevance, source relevance, authority, year/race/session matching and citation grounding; private forecasts are excluded. Explicit unsupported findings override an inconsistent positive grounding summary. Truncated excerpts leave support unknown. Reports retain excerpts for manual semantic review: lexical relevance and model judgments are not proof of entailment or universal safety. Live API failures remain failures; there is no silent model fallback.

Official references: [Models and pricing](https://platform.claude.com/docs/en/models/overview), [web search and domain controls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool).
