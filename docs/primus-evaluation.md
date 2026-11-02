# Primus AI evaluation and guardrails

Primus is currently deterministic, grounded local Q&A. It has no external LLM,
URL fetcher, arbitrary SQL, code executor, shell, write tool or autonomous forecast
submission. A paid/generative provider is not required to run this evaluation.

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
boundaries provide the stronger protection. A future external LLM integration
must be separately evaluated with multi-turn/tool-output attacks and independent
factual checks before exposure. Correctness here means agreement with saved
source data; it does not independently establish provider accuracy or perfect
source coverage. Unsupported F1 concepts can still receive an honest insufficient
evidence response. This is local resume-project tooling, not an audited public
production deployment.
