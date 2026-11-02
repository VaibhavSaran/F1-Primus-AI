# Using Primus F1: a complete walkthrough

Primus F1 combines recorded-session analysis, pre-race research and sourced F1 answers. This walkthrough covers every visible page and each Model Validation tab, with two explicit examples per feature. A scheduled date, qualifying classification and forecast are different evidence.

## Start locally

From the repository root, use the project environment:

```powershell
& '.\primusEnv\Scripts\python.exe' -m streamlit run streamlit_app/app.py --server.address=127.0.0.1 --server.port=8502
```

Open [Primus F1](http://127.0.0.1:8502/). The entrypoint requires loopback binding; changing a listen flag is not hosted deployment. A fresh clone needs separately prepared local archive/model state. See [environment](../ENVIRONMENT.md) and [architecture](architecture.md).

Primus AI is home; `/ask` remains an alias. Analysis pages share **Season → Event/round → Session**. Choose again after changing season/event because dependent choices reset. Only published saved sessions appear. Archive event names/dates/session inventory are authoritative; do not infer current round numbers from a static calendar or invent Sprint sessions.

## Two evidence contexts

**Completed archive:** Spain 2020 Race supports telemetry/replay; the active study's United States 2025 case supports predictions versus actuals. Historical reconstruction uses retrospective downloads with unverified original publication times. Its scores do not prove forecasts existed before the race.

**Mexico 2026, race dated 1 November:** only FP1, FP2, FP3 and Q are published in the documented local snapshot. R collection failed, so no local actual Race result is available. Eight prepared model forecasts cover 22 drivers and remain unscored. Reading an existing frozen record after the calendar date is not creating a new prospective forecast. Missing R does not establish cancellation, a sporting retirement or zero error. Later publication/scoring must be checked explicitly.

## 1. Primus AI

Ask without selectors or bind Additional context to an exact event/session. Inspect Internal data and web citations. Unbind when changing topics. New conversation starts another durable record; Reports exports completed turns. Without configured provider access, Local evidence mode supports relevant saved-evidence queries but not general generation/live search. Numeric forecast ranks come from saved local runs.

**Example 1 — completed:** ask “Who won the United States Grand Prix in 2025?” Inspect matching Race evidence, then request a teammate comparison from that field. A retirement status alone does not establish an incident cause.

**Example 2 — Mexico:** ask for Mexico 2026 model comparisons. Inspect the eight-model, 22-driver estimates/cutoff. They remain unscored because only FP1/FP2/FP3/Q are saved and R collection failed. Q order and first picks cannot establish the actual winner.

Read [Primus AI](features/primus-ai.md).

## 2. Race Overview

Cards summarize drivers, laps, weather and stints. Race/Sprint shows recorded classification; Q shows qualifying; practice shows fastest eligible timed laps. Expand participants for event-specific identities.

**Example 1 — completed:** choose Spain 2020 Race and inspect classification/coverage. Switch to FP1: practice times cannot establish Grand Prix finish order. Recorded retirements are sporting outcomes rather than collection failures.

**Example 2 — Mexico:** choose Q or FP1/FP2/FP3. Only these sessions are saved because R collection failed. No Race classification can be synthesized from Q or the separate eight-model, 22-driver unscored comparison.

Read [Race Overview](features/race-overview.md).

## 3. Predictions

Select a Saved input snapshot, model and exact saved cutoff/mode. Use saved cutoff & mode restores the request; Use forecast-ready inputs selects prepared evidence. Learned candidates need eligible artifacts; TabFM needs completed matching inference. Generate forecast reopens a matching saved run. The all-model panel reads prepared results without reloading weights on reruns.

**Example 1 — completed:** select the United States 2025 historical pre-Grand-Prix snapshot. Compare qualifying with an eligible saved/fitted challenger, then inspect actuals in Historical Race Studies. Target Race labels must not enter predictors; the reconstruction notice still applies.

**Example 2 — Mexico:** choose Prepared pre-race inputs · forecasts ready, expand Forecast this upcoming Grand Prix with all models, then Forecast with all active models. Read eight first picks and the 22-driver ranking comparison; open/export an individual matching run. FP1/FP2/FP3/Q are saved, R collection failed, and all remain unscored.

Read [Predictions](features/predictions.md) for model inputs and limitations.

## 4. Compare Drivers

Select drivers, clean-lap and compound filters, then telemetry driver/lap. Inspect lap trends, sectors and supplied speed, throttle/brake, engine/gear and DRS channels. Brake is binary rather than pressure; missing samples break lines.

**Example 1 — completed:** compare two Spain 2020 Race drivers. For missing-time behavior, select Spanish FP1, Roy Nissany, lap 2: recorded post-exit samples can remain visible without an official duration. The sample interval is not a certified timed lap.

**Example 2 — Mexico:** compare FP2 or Q participants/laps. Only FP1/FP2/FP3/Q are saved because R collection failed. These observations are not actual Grand Prix pace; eight prepared models with 22 drivers remain unscored separately.

Read [Compare Drivers](features/compare-drivers.md).

## 5. Race Replay

Select participants and play/pause or seek by time/lap/event. Markers, weather and event highlights follow the clock. Retained timing supports observed Race standings; fallback positions are labelled lap-end checkpoints. Practice/qualifying Session lap completion has Order rather than race position. Missing/stale observations stay unknown.

**Example 1 — completed:** replay Spain 2020 Race, pause at a later event and compare changing standings/weather. Select a subset and preserve absolute recorded ranks/intervals. Final classification must not substitute for earlier running order.

**Example 2 — Mexico:** replay available FP1/FP2/FP3/Q. R collection failed, so there is no local Mexico Race replay. Eight prepared 22-driver forecasts remain unscored and cannot become synthetic actual running order.

Read [Race Replay](features/race-replay.md).

## 6. Strategy Lab

Inspect observed compounds/stints, visits and one driver's lap time against tyre age. Initial departures can lack prior entry; final returns can lack later exit. Entry-to-exit elapsed time is not stationary service duration. Missing bounds remain unknown.

**Example 1 — completed:** follow a Spain 2020 Race driver's compounds and visits, then inspect pace/tyre age. A post-stop pace change alone does not isolate degradation or prove optimal strategy.

**Example 2 — Mexico:** inspect FP2 practice stints. Only FP1/FP2/FP3/Q are saved because R collection failed. Practice is not actual Race strategy; eight prepared 22-driver forecasts remain unscored and contain no actual stop evidence.

Read [Strategy Lab](features/strategy-lab.md).

## 7. Weather & Context

Inspect Temperature, Wind, Humidity & pressure, Rain, track status and race-control timelines. Rain flags are not depth/probability. Recorded status and independently verified cause are different. Historical weather/reanalysis is not an issued pre-race forecast.

**Example 1 — completed:** inspect Spain 2020 Race conditions/status, then seek Replay to the matching clock. Describe observations while distinguishing causal inference.

**Example 2 — Mexico:** inspect FP3/Q weather, not supposed Race conditions. FP1/FP2/FP3/Q are saved but R collection failed. Later weather cannot enter frozen eight-model, 22-driver inputs; their forecasts remain unscored.

Read [Weather & Context](features/weather-context.md).

## 8. Model Validation → Evaluation

Select report/model and 5/10/50/100 nested chronological test races. This changes test-cohort size, not training size. Compare against qualifying. Higher correlation/lower position error are better; each metric has its own denominator and race-bootstrap interval. Ranking-only outputs do not provide calibrated probability scores.

**Example 1 — completed:** inspect United States 2025 in Historical Race Studies, then compare aggregate qualifying/challenger scores. Do not call multi-race metrics that single race's accuracy or reconstruction verified live performance.

**Example 2 — Mexico:** eight prepared models with 22 drivers do not constitute scored evaluation. FP1/FP2/FP3/Q are saved, R collection failed and forecasts remain unscored. Keep the 2026 research holdout separate and locked.

Read [Evaluation](features/evaluation.md).

## 9. Model Validation → Performance Tracking

Select pre-race forecasts or reconstructions, model and outcome filter. Cohorts do not mix. Summaries choose latest cutoff per race/model/mode; older runs remain accessible without inheriting newer scores. A known field mismatch is incomplete evidence. Request health measures service operation rather than model accuracy.

**Example 1 — completed:** inspect a United States 2025 reconstruction's ranking, outcome revision and matching score where available. An earlier run must not inherit the latest score; its historical status stays explicit.

**Example 2 — Mexico:** choose Saved pre-race forecasts → Awaiting outcome. Eight models' 22-driver rankings remain unscored: FP1/FP2/FP3/Q exist but R collection failed. Later genuine outcomes require explicit append/scoring without changing frozen runs.

Read [Performance Tracking](features/performance-tracking.md).

## 10. Model Validation → Historical Race Studies

Choose study/race/model. Inspect the full predicted/actual field and signed error: predicted minus actual, so positive means a better actual finish than predicted. Matched qualifying comparisons matter when candidate cohorts differ.

**Example 1 — completed:** choose the active United States 2025 case. Compare qualifying with a learned candidate, including retired drivers. Read matched counts/intervals before generalizing from one first pick.

**Example 2 — Mexico:** do not add Mexico as scored history. FP1/FP2/FP3/Q are saved, R collection failed and eight-model, 22-driver forecasts remain unscored. Q/estimated ranks cannot populate actual-position labels.

Read [Historical Race Studies](features/historical-race-studies.md).

## 11. Reports

Download conversations, selected-session analysis and forecasts. Top selectors control the session ZIP. Saved forecast run is independent and retains its original event/cutoff. Legacy narratives are unvalidated previews.

**Example 1 — completed:** download Spain 2020 Race analysis, then select a United States 2025 forecast if present. Its event remains United States despite Spain selectors. Confirm context before comparing outcomes.

**Example 2 — Mexico:** export real Q analysis and a prepared 22-driver forecast. All eight models remain unscored: FP1/FP2/FP3/Q exist but R collection failed. Neither a ZIP nor conversation export establishes Race results.

Read [Reports](features/reports.md).

## 12. Settings & Data

Inspect the library, selected-channel coverage and readiness. Configuration presence is not a successful live call. The page starts no downloads. Replay standings availability comes from version-pinned timing rather than an obsolete ingestion flag.

**Example 1 — completed:** inspect Spain 2020 Race laps, positions, weather and standings coverage, then FP1. Investigate specific gaps without declaring the whole race lost or practice order actual race order.

**Example 2 — Mexico:** confirm FP1/FP2/FP3/Q inventory and failed/missing R publication. Eight prepared 22-driver forecasts remain unscored. Integration readiness implies neither an actual result nor functioning deployment.

Read [Settings & Data](features/settings-data.md).

## 13. About and credits

Getting started gives a usage sequence and completed/pending examples. Page guide explains all eleven visible pages. Credits attributes providers, libraries, Tom Shaw's replay and Mariana Antaya's prediction research. Attribution does not replace upstream redistribution rights.

**Example 1 — completed:** explain the United States 2025 study as archived-data research and credit the data/model ecosystem. Completion supports retrospective scoring, not proof of live forecasting.

**Example 2 — Mexico:** explain eight prepared 22-driver rankings as unscored. FP1/FP2/FP3/Q are saved and R collection failed. Credits must not imply actual Race evidence or a deployed service.

Read [About and credits](features/about-credits.md).

## Continue with operator references

The [documentation index](README.md) preserves archive, collection, recovery, model protocol, provider and runtime guides. [New-race updates](new-race-update-plan.md) is a plan, not an installed scheduler. [Dataset release](dataset-release.md) and [Kaggle release](kaggle-release.md) describe curated publication, not a completed upload. Preserve source/version lineage, frozen forecasts, separate cohorts and honest missing-outcome states as evidence changes.
