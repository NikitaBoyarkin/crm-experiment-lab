#!/usr/bin/env bash
# Phase 1 raw profile for crm-experiment-lab — exact command used for the report.
# Writes reports/eda/profile_raw.md only; touches nothing else.
set -euo pipefail

PROJECT="/Users/nikitaboarkin/Desktop/00 ide/00 portfolio/crm-experiment-lab"
EDA="/Users/nikitaboarkin/Desktop/00 ide/00 portfolio/.eda-tools/.venv/bin/python"
PROF="/Users/nikitaboarkin/Desktop/00 obsidian/obsidian/.claude/skills/eda/scripts/eda_profile.py"

cd "$PROJECT"
"$EDA" "$PROF" \
    --input data/experiment_sample.csv \
    --metric purchased \
    --segments day \
    --time-col day \
    --out reports/eda/profile_raw.md
