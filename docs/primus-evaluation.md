# Primus AI evaluation and guardrails

This evaluation targets the deterministic, grounded local Q&A path. The hosted
application additionally uses Claude with bounded cited web search; live provider
tests are separate. Neither path exposes arbitrary SQL, a code executor, shell,
write tools or autonomous forecast submission. A paid provider is not required
to run the deterministic evaluation below.

Run from the repository root with the project environment:

```powershell
& '.\primusEnv\Scripts\python.exe' -m agents.chat_evaluation
```

Each run creates a new timestamped directory under `.local/primus_evaluations/`.
Existing report directories are never overwritten. `report.json` contains every
synthetic/public prompt, actual response, expected status/facts, pass/fail checks,
pinned version references and implementation hashes. `report.md` summarizes the
categories; `report.sha256` authenticates the report bytes. No real user prompts
are copied from operational logs.

The independent oracle queries the saved SQL classification, laps, compound
counts and weather observations directly, without calling the chat projection
or using an LLM as judge. It checks six sessions across 2020, 2021, 2025 and 2026,
including the 2025 US race, qualifying and sprint. Four season-specific driver
profiles are compared to separately queried last-five race results and observed
team affiliation. Historical local-knowledge cutoff, wrong-year/event, unknown
driver, missing final-race result and forecast handoff cases must abstain or hand
off rather than invent facts.

The completed run passed **86/86 cases**:

| Category | Passed / total |
|---|---:|
| Archive facts and pinned citations | 36 / 36 |
| Wrong scope / unavailable-at-cutoff abstention | 24 / 24 |
| Missing facts and explicit forecast handoff | 3 / 3 |
| Mixed-domain, role override, secrets, code and URL attacks | 19 / 19 |
| Seasonal profiles | 4 / 4 |

For the 2025 US Grand Prix, Primus answered that Max Verstappen finished first,
identified Kimi Antonelli's fastest saved lap as 97.577 seconds, and cited the
saved classification/lap publication. Weather answers included the independently
verified 26.4–29.3 C air-temperature range and 154 samples. The 2025 Verstappen
profile reported a 1.8 mean classified position over its last five saved races
and observed Red Bull Racing affiliation. This is the selected season's
last-five-race window, not a pre-US-weekend profile; the answer labels it as such.

## Guardrail changes and regression evidence

An F1 keyword previously allowed mixed requests such as “F1 weather and write a
cake recipe” to reach routing. Those requests now receive a conservative refusal
before tools. Instruction-control phrases, zero-width obfuscation, local-file
requests and common unrelated generation tasks are checked before routing.
Instruction-bearing curated source summaries are withheld rather than echoed as
an answer. Legitimate wind/pressure questions and Unicode “parc fermé” are
accepted. Tests first demonstrated these failures, then verified the corrections.

Existing independent controls remain enforced: authenticated principals and
owner/public record checks; shared per-minute/day admission; active owned leases;
tool/context/output bounds; post-tool deadlines; duplicate request idempotence;
read-only explicit tool argument schemas; conservative source/local retrieval
cutoffs; vetted citations; safe Markdown rendering; no raw chat logging.
The focused chat/archive/evaluation suite passed **95 tests**. These include
cross-owner isolation, ledger outage fail-closed behavior, quota exhaustion,
expired execution, malicious tool arguments and source injection.

Evaluation uses a separate ledger and a bounded 1,000-request evaluation budget;
it does not raise production quotas. Network sockets are disabled by the CLI and
archive/context reads are read-only. No provider or inference API is called.

## Limits

This is a finite regression evaluation, not proof that every possible jailbreak
or factual error is impossible. Keyword filtering is defense in depth; the
read-only allowlist, absence of generative execution and trusted identity/ledger
boundaries provide the stronger protection. The deployed external LLM has a
separate provider evaluation described in [primus-provider.md](primus-provider.md);
multi-turn/tool-output attacks and independent factual checks remain necessary
as the model and tools change. Correctness in this offline suite means agreement with saved
source data; it does not independently establish provider accuracy or perfect
source coverage. Unsupported F1 concepts can still receive an honest insufficient
evidence response. These tests do not constitute a public security audit.

## November 13 hosted chat verification

- 115 focused tests passed across chat routing, provider citations, public identity,
  chat history and rate/spending limits using `primusEnv`.
- Five sequential hosted service probes passed: coding-problem refusal,
  instruction-override refusal, unrelated recipe refusal, a cited official
  Pirelli explanation and cited Motorsport.com reporting.
- Both paid replies completed for the same visitor without the former second-request
  budget rejection. Their measured costs were $0.052720 and $0.068060; these are
  examples rather than a general cost forecast.
- The hosted Streamlit browser also completed consecutive media and official-source
  questions in one conversation, displayed their distinct citation labels, and
  refused an F1-wrapped Python coding request.
- Official and media citations retain separate source categories. A native
  citation is evidence to inspect, not a guarantee that every claim is correct.
- During testing, BBC crawler restrictions caused an HTTP 400 request rejection.
  Both BBC domains were removed from the search allowlist; the corrected live
  official and media searches succeeded.

The hosted shared caps are $30/day and $100/month. Conservative $3 per-request
reservations remain in place; unknown bills are retained for operator
reconciliation instead of silently resetting usage. No credentials are included
in evaluation reports.
