# India State Finance Extractor (`statefin`)

Auditable extraction of Indian state-government finance variables (pensions, salaries,
interest, receipts, expenditure, deficits, debt, GSDP, central transfers) from official
PDFs into a **state × financial-year panel**, where every number can be traced to a
document, page, table and row.

Status: **Phase 1** (deterministic single-PDF pipeline) and the **Phase 3 foundation**
(provider-independent LLM row mapping + benchmark harness) are complete. Discovery and
download of documents (Phase 2) is not built yet.

## Principles

* The **table** is the unit of extraction. Deterministic code finds pages, tables, year
  columns, row labels and cell values. An LLM, when enabled, only classifies row *labels*.
  It never supplies a number.
* Raw observations are never altered. Deficits keep their printed sign, together with the
  detected sign convention. Non-crore units are flagged, not converted.
* Conflicting sources are never averaged. Agreement is decided within the printed
  precision. Otherwise the source hierarchy decides, or the cell goes to human review.
* Checks **flag**; they do not fix. A suspicious value goes to the review queue.
* A missing value is better than a fabricated one.
* Everything that depends on document conventions lives in `config/*.yaml`.

## Quick start

```bash
cd india_state_finance_extractor
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"     # or reuse an existing venv
.venv/bin/python extract.py --pdf path/to/report.pdf
```

Useful options:

```bash
python extract.py --pdf a.pdf --pdf b.pdf            # several documents, reconciled together
python extract.py --pdf a.pdf --state "Himachal Pradesh" --doc-type CAG_SFAR   # override detection
python extract.py --pdf a.pdf --force                # ignore caches
python extract.py --pdf a.pdf --llm                  # add LLM row mapping (see below)
python -m pytest                                     # tests (the real-PDF regression runs if ../Docs has the SFAR)
```

`python extract.py --all-states --start-year 1995 --end-year 2025` is reserved for Mode A
(Phase 2) and currently exits with a message.

## Pipeline

```
PDF ─► registry        sha256 → document_id, cached copy, manifest.json with per-stage status
    ─► pdf/text        page text (PyMuPDF); pages without a text layer marked needs_ocr
    ─► candidates      rank pages by distinct target concepts + table signals
    ─► pdf/tables      PyMuPDF find_tables on candidate pages (landscape pages de-rotated in
                       memory); caption, unit, "Source:" note, footnotes
    ─► structure       which cells are year columns / row labels, by geometry:
                       Actuals vs BE/RE/projection columns, multi-row headers, transposed
                       tables, percent-share and "change over" columns, section units
    ─► mapper          whole-label match against variables.yaml → observations + provenance
    ─► [semantic]      optional LLM classification of rows the rules could not map
    ─► reconciliation  per state × year × variable: consistent / resolved_by_priority / conflict
    ─► validation      per-value, accounting identity, ratios, year-on-year, provenance text
    ─► export          extractions.json/csv, review_queue.csv, panel.csv, panel_long.csv
```

Per-document stages (`DOWNLOADED → TEXT_EXTRACTED → CANDIDATES_FOUND → TABLES_EXTRACTED →
VALUES_EXTRACTED → [SEMANTIC_MAPPED] → VALIDATED`) are cached under
`data/processed/<document_id>/`. A stage is reused only if its inputs and the config
sections it depends on are unchanged. Editing `variables.yaml` re-runs mapping without
re-parsing any PDF. A failed document records the failing stage and resumes from there.

## Outputs (`data/output/`)

| file | content |
|---|---|
| `extractions.csv` / `.json` | **raw layer**: one row per observation (state, year, variable, value, unit, raw label, document, hash, page, table caption, row text, confidence + reasons, flags, review status, snapshot path) |
| `review_queue.csv` | observations needing review, missing core variables, unresolved LLM rows; blank `reviewer_decision/value/note` columns |
| `panel.csv` | **analytical layer**: state × year × variable (INR crore), rebuilt from the raw layer on every run |
| `panel_long.csv` | every panel cell with its status, reason and the observation id(s) it came from |
| `candidates.json` | page ranking, detected tables, year columns (with skip reasons), rejected labels |
| `summary.txt` | the terminal summary |
| `llm_label_suggestions.csv` | (with `--llm`) every label the LLM classified vs the rules. Confirmed ones can be promoted to aliases |
| `human_reviews.csv` | every reviewer decision ever made (append-only ledger export) |

Table snapshots (PNG crops of each source table) are written to
`data/processed/<document_id>/snapshots/` for visual checking.

## Live dashboard and human review

```bash
python dashboard.py --open                         # http://127.0.0.1:8765
python extract.py --pdf report.pdf --dashboard     # run in the terminal and watch/review in the browser
```

Every run records its progress in `data/statefin.db` (SQLite), whether you start it from the
dashboard or with `extract.py` in a terminal (`--no-db` opts out). The dashboard streams
that database live (Server-Sent Events), so it shows runs in progress from either place.

| tab | what it shows |
|---|---|
| **Live** | current run (status, elapsed, options), stat tiles (documents, pages, candidate pages, tables, observations, open reviews, LLM calls), one card per document with a stage stepper (Text → Candidates → Tables → Values → LLM → Validated) and a progress meter ("finding tables on candidate page 47 of 54"), the event log (filter by warnings/errors), and a form to start an extraction |
| **Review** | the review queue, most severe first, with filters (open/decided, type, variable, search). Each item shows the value, **the source table with the exact cell highlighted** (toggle to the whole page), row label and printed row, flags with explanations, confidence reasons, all observations of the same state × year × variable, and the decision history |
| **Panel** | coverage grid, years × variables, each cell with status icon + colour (auto-accepted, reviewer-verified, needs review, derived, excluded, no data); hover for source, click to open it for review |
| **Observations** | every observation, searchable; click to open |
| **Documents** | processed documents with stage status and errors; recent runs; CSV export of all review decisions |

**Decisions** (buttons or keys: `a` accept, `r` reject, `c` correct, `j`/`k` next/previous):

* *Accept*: the value is final and enters the panel even if an automatic check flagged it.
* *Reject*: kept in the raw layer as `HUMAN_REJECTED`; takes no part in reconciliation or the panel.
* *Correct*: the panel uses your value. The extracted value stays in `value_as_extracted` and
  `raw_value_text`, and an info flag records who changed what and why. Enter INR crore; deficits negative.
* *This one is correct* (conflicts): accepts one observation and rejects the others of that cell.
* *Map row* (rows the LLM could not decide): assign a variable, or "none". The row's printed
  cells become observations (`mapping_method = human:<you>`).
* *Acknowledge* (missing variables): record that the documents lack it, with a note.
* *Undo*: withdraws a decision.

Decisions are stored **append-only** (the full history stays; the latest decision per item
wins). They are keyed to stable observation ids, so they survive reruns. About 1.5 s after a
decision, the dashboard rebuilds outputs from the cached extraction (≈1 s; no PDF parsing),
so the panel, CSVs and review queue reflect it. All decisions are exported to
`data/output/human_reviews.csv` on every run. Review can start mid-run: results are
published after each document.

The server binds to 127.0.0.1 only and refuses requests with a non-local `Host` header,
POSTs without its `X-Statefin` header, and cross-origin POSTs, so other websites cannot
submit reviews or start runs. The page loads nothing from the internet.

## Configuration

* `config/variables.yaml`: canonical variables, definitions, aliases/patterns (matched
  against the **whole** normalised label), per-alias caveats, sanity ranges. For example,
  "Salaries" alone maps to `salary_wages` at reduced confidence with a note that wages may
  be excluded.
* `config/source_rules.yaml`: document types and **source priority** (1 Finance Accounts,
  2 SFAR, 3 State budget, 4 Finance Commission studies, 5 other), unit phrases, column
  qualifiers (Actuals / BE / RE / projection / provisional), sign-convention markers.
* `config/pipeline.yaml`: thresholds: candidate scoring, confidence factors and tiers
  (≥0.95 exact, 0.80–0.95 strong, 0.60–0.80 ambiguous, <0.60 manual), tolerances, ratio checks.
* `config/states.yaml`: states/UTs with legislature, aliases (Orissa → Odisha), reorganisations.
* `config/llm.yaml`: LLM providers, models and policy.

## Validation (flag-only)

Impossible negatives; unknown / non-crore units; implausible magnitude; financial year
after the report year, or **after the PDF's own creation date** (a 2019 study cannot
contain 2020-21 actuals); declared vs canonical sign convention; revenue identity
`RR − RE = RD`; salary+pension+interest ≤ revenue expenditure; ratio ranges
(salary/RE, pension/RE, interest/RR, GSDP/RR, debt/GSDP, |FD|/RR, …); year-on-year jumps
(> 50 % warning, ≥ 10× possible unit error); mixed units in a series; and **every value's
printed text must occur on its cited page**.

Reconciliation compares deficits in the canonical sign convention (surplus-positive, as
CAG prints them). An unmarked "Gross Fiscal Deficit" of +12,611.05 is recognised as the
deficit-positive presentation of −12,611.05, and is recorded as such without changing it.

## LLM semantic mapping (optional)

The LLM answers one question per row: *which canonical variable does this label denote?*
Its answer is `<variable>`, `NONE` or `UNKNOWN`, with a confidence and reason. The JSON
schema has no field for values, so numbers always come from the parsed cells.

```bash
echo "GEMINI_API_KEY=..." > .env            # git-ignored; see .env.example
python extract.py --pdf report.pdf --llm
python extract.py --pdf report.pdf --llm --llm-provider ollama --llm-model qwen2.5:14b --no-llm-fallback
STATEFIN_LLM_MODEL=gemini-3.5-flash python extract.py --pdf report.pdf --llm
```

* **Providers** (`src/statefin/llm/`): Gemini API, Ollama, any OpenAI-compatible server,
  all behind `LLMProvider.structured_extract(input_data, schema, instructions)`. They use
  plain HTTP from the standard library, so no vendor SDK is needed.
* **Default models**: `gemini-3.5-flash-lite` → fallback `gemini-3.5-flash`. The originally
  specified `gemini-2.5-flash-lite`/`gemini-2.5-flash` return HTTP 404 ("no longer
  available to new users") for new keys. Change the models in `config/llm.yaml`.
* **Selective use**: by default only tables with a row whose label uses target vocabulary
  but that the rules could not map are sent (`--llm-tables all_year_tables` sends every
  year table, which is useful for cross-checking the rules).
* **Two stages**: the fallback model is called only when the primary abstains, is below
  `escalate_below_confidence`, or returns an invalid/hallucinated answer. If the models
  still disagree, the row becomes `LLM_UNRESOLVED_ROW` in the review queue.
* **Guardrails**: unknown row ids or echoed labels that differ from the table are counted
  as hallucinations and discarded. The LLM cannot override the rules' ratio/percentage
  exclusions or unit checks. Rule mappings keep precedence, and LLM disagreement with a
  rule mapping is flagged `llm_disagrees`.
* **Review policy**: until the benchmark shows adequate accuracy, observations that exist
  only because of an LLM mapping are capped at confidence 0.79, so they go to review
  (`accept_llm_mappings_without_review` in `config/llm.yaml`).
* **Reproducibility**: responses are cached by a hash of provider, model, prompt, input
  and schema (`data/llm_cache/`), and each entry stores the full request/response.
* Billing/auth/model errors stop the LLM step once, keep all deterministic results, and
  make the CLI exit with code 3.

## Benchmark

```bash
python benchmark.py draft --pdf report.pdf        # pre-fill cases for its tables (status: draft)
python benchmark.py list
# verify each benchmark/cases/*.json against its snapshot, fix variables/values, set "status": "verified"
python benchmark.py eval --system deterministic
python benchmark.py eval --system hybrid                          # rules + LLM (production mode)
python benchmark.py eval --system llm_only --llm-provider ollama --llm-model gemma3:12b --no-llm-fallback
```

Metrics: variable-mapping accuracy, target-row outcomes (correct / wrong / missed /
abstained), false positives on non-target rows, row identification P/R, observation
P/R/F1, financial-year accuracy, numeric accuracy, unit accuracy, hallucination rate,
unsupported values, abstention rate, calls/escalations/tokens. Reports are saved to
`benchmark/results/`.

`benchmark/cases/` holds **41 draft cases** (331 rows) from the two Himachal Pradesh
documents. They are pre-filled by the rules and **not yet verified**. The evaluator
warns that scores on unverified drafts are circular. Target: 50–100 verified tables
across several states before any prompt/program optimisation.

## Layout

```
config/          variables, source_rules, pipeline, states, llm (YAML)
src/statefin/
  registry.py    content-addressed document cache + stage status
  pipeline.py    orchestration          models.py  records   config.py  YAML loading
  pdf/           text.py  tables.py  ocr.py (detection only)
  extraction/    parsers.py candidates.py metadata.py structure.py mapper.py semantic.py
  llm/           base.py providers.py cache.py settings.py
  reconciliation/sources.py   validation/{anomalies,accounting}.py
  export/        outputs.py panel.py summary.py
  benchmark/     cases.py draft.py evaluate.py
  dashboard/     server.py images.py static/index.html     store.py progress.py review.py
  discovery/ download/        Phase 2 interfaces (not implemented)
tests/           synthetic-PDF fixture + unit, pipeline, LLM (fake provider / fake HTTP) tests
extract.py  benchmark.py  dashboard.py
```

## Known limitations

* OCR is not run. Pages without a text layer are only listed (`needs_ocr`).
* Tables spanning pages *without* a repeated header, and tables with no year axis (e.g.
  a single-year statement whose year is only in the caption), are not interpreted.
* Only INR crore enters the panel. Lakh/thousand values are kept and flagged, not converted.
* GSDP base-year series (2004-05 vs 2011-12) are not distinguished yet. They show up as
  conflicts.
* Optional variables (pensioners, employees) and the pension-policy table are not extracted.
* The dashboard is single-user and runs one extraction at a time; it has no login (it only listens on localhost).
* Tested on two Himachal Pradesh documents only. Other states' layouts will surface new
  cases; add them to the benchmark as they appear.
