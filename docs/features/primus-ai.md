# Primus AI

Primus AI is the home page and the F1 question-answering interface. The default home route and `/ask` alias call `ask_f1()` in [screens.py](../../streamlit_app/screens.py). No race selection is required for a general question. Give an exact year, event and session when sporting facts depend on them.

## Using the page

Ask a question and inspect its evidence. **Internal data** identifies saved local records; linked web citations identify retrieved sources. Expand **Additional context** to bind a race/session, attach a saved snapshot or forecast, or focus a driver. Binding is optional. Explicit questions that contradict a binding should be refined or unbound rather than silently answered for another race.

**New conversation** clears the active browser conversation and starts a separate durable record. Completed turns save automatically. **Reports → Saved conversations** exports Markdown with original times, context and sources. Conversations predating durable saving may be unavailable.

With a configured provider, Claude Sonnet 5.5 uses bounded catalog, archive, saved-run, pending-forecast, evaluation and profile tools, plus cited web search. Sources must support the specific year/event/session and claim. Current web retrieval is not evidence that a fact was available at a historical cutoff. Without a provider, **Local evidence mode** can answer supported saved-evidence questions, but cannot replace generation or live search.

Trusted identity, owner visibility, cutoff, rate and spending checks apply. There are no arbitrary shell, SQL, file or user-directed network tools. Numeric forecast rankings are rendered locally from real saved runs rather than rewritten by the provider. Previous assistant bodies are omitted from outbound follow-up history. Timeouts and reserved spending remain explicit operational states.

## Example 1: completed 2025 United States Grand Prix

Ask “Who won the United States Grand Prix in 2025?” Resolve the exact event and inspect saved Race classification or a matching cited result. Follow with a teammate comparison using that same recorded field. A retirement status alone does not independently establish an incident's cause. Practice rank and another year's result cannot answer this question.

## Example 2: Mexico 2026, race dated 1 November

Ask “Compare the saved model forecasts for Mexico 2026.” Only FP1, FP2, FP3 and Q are published in the documented local snapshot; R collection failed. Eight prepared models cover 22 drivers and remain unscored. Primus may show those estimates and their frozen cutoff. It must not infer an actual winner from Q order, a predicted first pick or the calendar date. Failed R collection does not establish cancellation or a sporting retirement.

## Implementation and references

Sources: [ChatService](../../agents/chat.py), [Claude provider](../../agents/claude_provider.py), [archive context](../../agents/archive_context.py), [chat history](../../agents/chat_history.py), [access](../../security/access.py), [limits](../../security/limits.py). See [provider operations](../primus-provider.md), [chat design](../archive-grounded-chat.md) and [chat evaluation](../primus-evaluation.md).
