"""Training-only discrimination gates; model/templates/scalers are frozen."""
# Selected from 2001–2019 monsoon ordinary-day distances only, using frozen
# training scalers and training-event templates. Ordinary days exclude onset-14
# through onset+7. Cutoff is the largest representable float below the first
# excluded order statistic, so inclusive distance <= cutoff admits <=10%
# of all ordinary days. Missing windows count as no match (243 of 2505).
# Pre-flood hits require a LOEO match at lead 1–3 AND distance <= cutoff.
# Both hit rates are below 25%, so resemblance labels are disabled.
RESEARCH_CLOSE_MATCH_CUTOFF = 4.967996365959005
OPERATIONAL_CLOSE_MATCH_CUTOFF = 3.6858344516098085
RESEARCH_ANALOGUE_DISPLAY_ENABLED = False
OPERATIONAL_ANALOGUE_DISPLAY_ENABLED = False
