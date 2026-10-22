# Telecom Customer Churn Insight Assistant

A small local web application for reviewing fictional telecom customer records. It
summarizes verified evidence, computes churn risk with a fixed rule, recommends a
policy-bounded action when the evidence supports one, and pauses for supervisor
review when records conflict, history is insufficient, or a request exceeds policy.

## Run locally

Requires Python 3.12 or newer.

```powershell
uv sync --extra dev
uv run python app.py
```

Open `http://127.0.0.1:8000` for the browser form. The JSON API is
`POST http://127.0.0.1:8000/insight`; supervisor reviews resume through
`POST http://127.0.0.1:8000/review`.
The responsive agent screen places request fields beside risk, the recommended
action, cited drivers, and any missing information or supervisor-review status.

For pip users, install and run with:

```powershell
python -m pip install -e ".[dev]"
python app.py
```

Run the golden scenario tests:

```powershell
uv run pytest
# or: python -m pytest
```

The default session date is `2026-10-15` so the expired-offer scenario is
deterministic. Change the date in the UI when reviewing another session.

## Workflow and safeguards

The LangGraph workflow has eight nodes: input validation, record retrieval,
conflict resolution, risk scoring, response generation, guard checks, supervisor
review, and finalization. Escalated requests interrupt at the supervisor node.
The browser form can resume that paused run after review; approving a run does
not change the already-guarded recommendation or authorize an out-of-policy offer.

Risk is computed only in code. The model cannot set or change the risk score, risk band, evidence citations, or
recommended action. By default, summary
generation is deterministic and runs without a model service. To use the
assignment's Ollama model for a concise summary, set these PowerShell variables
before starting the server:

```powershell
$env:OLLAMA_MODEL = "gpt-oss:120b-cloud"
$env:OLLAMA_HOST = "http://localhost:11434"
python app.py
```

Only verified driver evidence and the code-computed score and band are sent to
Ollama; customer IDs, agent questions, and stated reasons are not included. If a
configured Ollama service fails or returns invalid summary JSON, the API reports
an error instead of silently substituting a result.

| Scored signal | Weight |
| --- | ---: |
| Three or more network complaints in 30 days | +3 |
| Open or disputed billing issue | +3 |
| Competitor mention or port-out request | +3 |
| Usage down more than 30% month on month | +2 |
| Repeat contact for the same issue | +1 |

Scores 0–2 are low, 3–5 medium, and 6 or more high. Verified records are the
only customer evidence eligible for citation. If two verified records for the
same issue disagree, the latest dated record determines the current status and
both records are surfaced as a conflict. Unverified content is never cited or
scored.

The action catalog is `plan_review`, `network_escalation`, `bill_correction`,
and `callback`. Credits are capped at Rs 500; plan discounts are capped at 20%
for 3 months. An exception is never promised and always pauses for supervisor
review. The festive offer is valid only from 2026-09-25 through 2026-10-08.

## Data and privacy

The sample dataset is in `data/sample_data.json`. It contains fictional records
for five case-study customers plus one learner-added customer, policy documents,
and deliberate cases for conflicts, unverified content, expired offers,
unsupported requests, and missing history.
No customer names or direct contact details are used. Do not enter personal
data into the app. Default deterministic mode stays local and request content
is not written to logs. If Ollama is enabled, only verified driver evidence and
the computed score and band are sent to the configured endpoint; avoid enabling
a cloud model for sensitive records unless approved for that data.

## Response shape

The structured response contains `customer_id`, `summary`, `drivers`, `risk`,
`recommended_action`, `missing_information`, `unsupported`, `data_conflicts`,
`escalate`, and `escalation_reason`. The API also returns `supervisor_status` and
a `review_id` when a LangGraph interrupt is pending. Each driver and action
includes citations checked against verified retrieved records and policy
documents before the response is returned.
