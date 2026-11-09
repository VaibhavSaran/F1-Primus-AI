# Using Primus F1: a complete walkthrough

Primus F1 combines recorded-session analysis, pre-race research and sourced F1 answers. This walkthrough covers every visible page and each Model Validation tab, with two explicit examples per feature. A scheduled date, qualifying classification and forecast are different evidence.

## Start locally

For the hosted demonstration, open [Primus F1](https://f1primus.vaibhavsaran.com/)
and sign in with the owner's Google account. The following local startup
instructions are useful when demonstrating the source checkout.

From the repository root, use the project environment:

```powershell
& '.\primusEnv\Scripts\python.exe' -m streamlit run streamlit_app/app.py --server.address=127.0.0.1 --server.port=8502
```

Open [Primus F1](http://127.0.0.1:8502/). The entrypoint requires loopback binding; changing a listen flag is not hosted deployment. A fresh clone needs separately prepared local archive/model state. See [environment](../ENVIRONMENT.md) and [architecture](architecture.md).

Primus AI is home; `/ask` remains an alias. Analysis pages share **Season → Event/round → Session**. Choose again after changing season/event because dependent choices reset. Only published saved sessions appear. Archive event names/dates/session inventory are authoritative; do not infer current round numbers from a static calendar or invent Sprint sessions.

## Two evidence contexts

**Completed archive:** Spain 2020 Race supports telemetry/replay; the active study's United States 2025 case supports predictions versus actuals. Historical reconstruction uses retrospective downloads with unverified original publication times. Its scores do not prove forecasts existed before the race.

**Mexico 2026, race dated 1 November:** FP1, FP2, FP3, Qualifying and Race are now
published, as are all five Sao Paulo sessions through November 8. Use Race for
the recorded result and the prepared pre-race snapshot for saved forecasts.
The hosted prepared snapshot has saved TabFM and XGBoost rankings for 22 drivers.
These runs await explicit outcome ingestion and scoring. The local research
archive also retains earlier eight-model comparisons. Preserve the saved input
cutoff when comparing a forecast against the newly collected actual result.

## 1. Primus AI

Ask without selectors or bind Additional context to an exact event/session. Inspect Internal data and web citations. Unbind when changing topics. New conversation starts another durable record; Reports exports completed turns. Without configured provider access, Local evidence mode supports relevant saved-evidence queries but not general generation/live search. Numeric forecast ranks come from saved local runs.

**Example 1 — completed:** ask “Who won the United States Grand Prix in 2025?” Inspect matching Race evidence, then request a teammate comparison from that field. A retirement status alone does not establish an incident cause.

**Example 2 — Mexico:** ask who won the 2026 Mexico race and inspect the internal
race-result evidence. Then ask about saved TabFM and XGBoost forecasts and check
their original input cutoff. Outcome scoring is a separate recorded step.

Read [Primus AI](features/primus-ai.md).

## 2. Race Overview

Cards summarize drivers, laps, weather and stints. Race/Sprint shows recorded classification; Q shows qualifying; practice shows fastest eligible timed laps. Expand participants for event-specific identities.

**Example 1 — completed:** choose Spain 2020 Race and inspect classification/coverage. Switch to FP1: practice times cannot establish Grand Prix finish order. Recorded retirements are sporting outcomes rather than collection failures.

**Example 2 — Mexico:** choose Grand Prix Race to see Max Verstappen's recorded
win. Switch to Qualifying to compare starting-weekend performance, then select
the Race session again when explaining the final classification.

Read [Race Overview](features/race-overview.md).

## 3. Predictions

Select a Saved input snapshot, model and exact saved cutoff/mode. Use saved cutoff & mode restores the request; Use forecast-ready inputs selects prepared evidence. Learned candidates need eligible artifacts; TabFM needs completed matching inference. Generate forecast reopens a matching saved run. The all-model panel reads prepared results without reloading weights on reruns.

**Example 1 — completed:** select the United States 2025 historical pre-Grand-Prix snapshot. Compare qualifying with an eligible saved/fitted challenger, then inspect actuals in Historical Race Studies. Target Race labels must not enter predictors; the reconstruction notice still applies.

**Example 2 — Mexico:** choose Prepared pre-race inputs · forecasts ready, select
TabFM and Generate forecast to open its saved 22-driver ranking. Repeat for
XGBoost and compare first picks. Inspect the original cutoff and export each run.
Check candidate readiness before requesting other models.

Read [Predictions](features/predictions.md) for model inputs and limitations.

## 4. Compare Drivers

Select drivers, clean-lap and compound filters, then telemetry driver/lap. Inspect lap trends, sectors and supplied speed, throttle/brake, engine/gear and DRS channels. Brake is binary rather than pressure; missing samples break lines.

**Example 1 — completed:** compare two Spain 2020 Race drivers. For missing-time behavior, select Spanish FP1, Roy Nissany, lap 2: recorded post-exit samples can remain visible without an official duration. The sample interval is not a certified timed lap.

**Example 2 — Mexico:** compare FP2 or qualifying laps for two participants,
then switch to Race to inspect their race laps. Explain how traffic, tyre age
and conditions affect that comparison.

Read [Compare Drivers](features/compare-drivers.md).

## 5. Race Replay

Select participants and play/pause or seek by time/lap/event. Markers, weather and event highlights follow the clock. Retained timing supports observed Race standings; fallback positions are labelled lap-end checkpoints. Practice/qualifying Session lap completion has Order rather than race position. Missing/stale observations stay unknown.

**Example 1 — completed:** replay Spain 2020 Race, pause at a later event and compare changing standings/weather. Select a subset and preserve absolute recorded ranks/intervals. Final classification must not substitute for earlier running order.

**Example 2 — Mexico:** select Grand Prix Race and play its recorded positions.
Seek to a lap and inspect the observed event markers and classification
checkpoints. Repeat with Sao Paulo to show the newly added November 8 race.

Read [Race Replay](features/race-replay.md).

## 6. Strategy Lab

Inspect observed compounds/stints, visits and one driver's lap time against tyre age. Initial departures can lack prior entry; final returns can lack later exit. Entry-to-exit elapsed time is not stationary service duration. Missing bounds remain unknown.

**Example 1 — completed:** follow a Spain 2020 Race driver's compounds and visits, then inspect pace/tyre age. A post-stop pace change alone does not isolate degradation or prove optimal strategy.

**Example 2 — Mexico:** select Race to inspect recorded stints and pit visits.
Compare one driver's compounds and tyre age across stints. Entry-to-exit duration
includes pit-lane transit and any recorded garage time.

Read [Strategy Lab](features/strategy-lab.md).

## 7. Weather & Context

Inspect Temperature, Wind, Humidity & pressure, Rain, track status and race-control timelines. Rain flags are not depth/probability. Recorded status and independently verified cause are different. Historical weather/reanalysis is not an issued pre-race forecast.

**Example 1 — completed:** inspect Spain 2020 Race conditions/status, then seek Replay to the matching clock. Describe observations while distinguishing causal inference.

**Example 2 — Mexico:** inspect Race weather, which contains ten retained
provider observations, and its race-control messages. Sao Paulo Race has 160
weather observations. Keep retrospective regional ERA5 and measured session
weather identified by source when discussing weather inputs.

Read [Weather & Context](features/weather-context.md).

## 8. Model Validation → Evaluation

Select report/model and 5/10/50/100 nested chronological test races. This changes test-cohort size, not training size. Compare against qualifying. Higher correlation/lower position error are better; each metric has its own denominator and race-bootstrap interval. Ranking-only outputs do not provide calibrated probability scores.

**Example 1 — completed:** inspect United States 2025 in Historical Race Studies, then compare aggregate qualifying/challenger scores. Do not call multi-race metrics that single race's accuracy or reconstruction verified live performance.

**Example 2 — Mexico:** inspect the saved pre-race runs and explain the explicit
outcome-ingestion and scoring step required for this completed race. Keep the
2026 research holdout locked under its existing protocol.

Read [Evaluation](features/evaluation.md).

## 9. Model Validation → Performance Tracking

Select pre-race forecasts or reconstructions, model and outcome filter. Cohorts do not mix. Summaries choose latest cutoff per race/model/mode; older runs remain accessible without inheriting newer scores. A known field mismatch is incomplete evidence. Request health measures service operation rather than model accuracy.

**Example 1 — completed:** inspect a United States 2025 reconstruction's ranking, outcome revision and matching score where available. An earlier run must not inherit the latest score; its historical status stays explicit.

**Example 2 — Mexico:** inspect saved pre-race forecasts awaiting outcome
scoring. The Race publication is now available in the archive. Append the genuine
outcome through the scoring workflow while preserving the frozen predictions.

Read [Performance Tracking](features/performance-tracking.md).

## 10. Model Validation → Historical Race Studies

Choose study/race/model. Inspect the full predicted/actual field and signed error: predicted minus actual, so positive means a better actual finish than predicted. Matched qualifying comparisons matter when candidate cohorts differ.

**Example 1 — completed:** choose the active United States 2025 case. Compare qualifying with a learned candidate, including retired drivers. Read matched counts/intervals before generalizing from one first pick.

**Example 2 — Mexico:** distinguish the saved pre-race ranking from the newly
collected Race classification. Historical Race Studies retains its original
evaluation cohort; a new scored example needs its matching forecast and outcome.

Read [Historical Race Studies](features/historical-race-studies.md).

## 11. Reports

Download conversations, selected-session analysis and forecasts. Top selectors control the session ZIP. Saved forecast run is independent and retains its original event/cutoff. Legacy narratives are unvalidated previews.

**Example 1 — completed:** download Spain 2020 Race analysis, then select a United States 2025 forecast if present. Its event remains United States despite Spain selectors. Confirm context before comparing outcomes.

**Example 2 — Mexico:** export Race analysis and a saved 22-driver forecast.
Compare their event, session and cutoff metadata, then download a completed
conversation to demonstrate how source references are retained.

Read [Reports](features/reports.md).

## 12. Settings & Data

Inspect the library, selected-channel coverage and readiness. Configuration presence is not a successful live call. The page starts no downloads. Replay standings availability comes from version-pinned timing rather than an obsolete ingestion flag.

**Example 1 — completed:** inspect Spain 2020 Race laps, positions, weather and standings coverage, then FP1. Investigate specific gaps without declaring the whole race lost or practice order actual race order.

**Example 2 — Mexico:** confirm all five sessions in the inventory. Repeat for
Sao Paulo, then inspect the preserved partial flag for Rafael Camara's four
parser-generated car samples. Review readiness before choosing a forecast model.

Read [Settings & Data](features/settings-data.md).

## 13. About and credits

Getting started gives a usage sequence and completed/pending examples. Page guide explains all eleven visible pages. Credits attributes providers, libraries, Tom Shaw's replay and Mariana Antaya's prediction research. Attribution does not replace upstream redistribution rights.

**Example 1 — completed:** explain the United States 2025 study as archived-data research and credit the data/model ecosystem. Completion supports retrospective scoring, not proof of live forecasting.

**Example 2 — Mexico:** explain how the hosted project combines recorded race
evidence with saved pre-race rankings. Credit the data providers and reference
projects, then describe the IAP gateway, VM, persistent archive and session reads.

Read [About and credits](features/about-credits.md).

## Continue with operator references

The [documentation index](README.md) preserves archive, collection, recovery, model protocol, provider and runtime guides. [New-race updates](new-race-update-plan.md) is a plan, not an installed scheduler. [Dataset release](dataset-release.md) and [Kaggle release](kaggle-release.md) describe curated publication, not a completed upload. Preserve source/version lineage, frozen forecasts, separate cohorts and honest missing-outcome states as evidence changes.
