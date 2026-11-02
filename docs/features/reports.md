# Reports

**Reports** (`/reports`) combines saved conversations, selected-session analysis and immutable forecast runs. A separate expander retains legacy narratives. Every artifact keeps its own context.

## Downloads and context

**Saved conversations** lists the current owner's durable chats. Download Markdown or expand the preview. Transcripts retain question/answer times, bound session and sources. New conversation starts another record without deleting earlier ones. Chats predating durable saving may be unrecoverable.

**Selected session analysis** follows the top selectors. Its ZIP contains entries, classification, laps, derived stints/pit visits, track status, race control, weather and coverage, with selected session/version metadata and file hashes in a manifest. Dense position/car streams stay in the library. Unavailable exports warn without substituting data.

**Saved forecast run** is independent and labelled by event/model. It can refer to a different race from the top selectors. Expand Original context & provenance for its own cutoff, mode and evidence status. Finishing order, Evidence and Briefing tabs support driver focus plus JSON, ranking CSV and available Markdown exports. The original JSON retains exact run/input/model references; public documentation need not reproduce private identifiers.

**Legacy Markdown reports** preserves earlier generated previews. They lack the model/input provenance necessary for validated accuracy, and are labelled unvalidated historical narratives. A polished narrative does not establish a measured result.

## Example 1: completed 2020 Spanish Grand Prix

Select Spain 2020 Race and download Selected session analysis. Compare classification with laps/pit visits using the manifest. Then select a United States 2025 saved forecast, if present: its original event and cutoff remain United States 2025 even while the top selectors say Spain. Confirm that distinction before comparing outcomes.

## Example 2: Mexico 2026, race dated 1 November

Choose Mexico Q for an actual qualifying analysis ZIP, then a prepared Mexico forecast to export its 22-driver ranking. Eight prepared models remain unscored. Only FP1/FP2/FP3/Q are published because R collection failed, so no Mexico Race analysis or actual-result comparison is implied. A chat discussing predicted first picks is not an actual classification.

## Implementation

Sources: [Reports UI](../../streamlit_app/screens.py), [analysis bundle](../../flows/analysis_report.py), [chat history](../../agents/chat_history.py), [tracking reads](../../monitoring/queries.py). See [dataset release](../dataset-release.md) for separately curated shareable datasets; a local download is not deployment/publication.
