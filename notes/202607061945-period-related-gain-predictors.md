> [!NOTE]
> Drafted by a LLM-based AI tool (Codex/GPT-5).

# Exclude period-related process measures from default gain predictors

Decision: `Predictors.DEFAULT_GAIN` now excludes `Variables.PERIOD_RELATED` (`attend`, `tascore`, `tachang`), matching the existing `DEFAULT_LEVEL` exclusion.

Rationale: the gradient-boosting gain models are reported as predictors of progress. Attendance and teaching-assistant process measures are observed during the gain interval, so including them in the default predictor set makes the ranking partly retrospective and can be mistaken for baseline/actionable prediction. Excluding them keeps the default ranking focused on baseline/state variables. If intervention-process variables are needed, they should be introduced in an explicit sensitivity model or process-focused analysis with that interpretation stated directly.
