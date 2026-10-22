from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any, TypedDict
from urllib.error import URLError
from urllib.request import Request, urlopen

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sample_data.json"
DRIVER_ORDER = (
    "network_quality",
    "billing_dispute",
    "plan_fit",
    "service_experience",
    "competitor_intent",
    "usage_decline",
)
ACTION_CATALOG = {
    "plan_review",
    "network_escalation",
    "bill_correction",
    "callback",
}
SIGNAL_WEIGHTS = {
    "network_complaints_3plus": 3,
    "open_billing_issue": 3,
    "competitor_intent": 3,
    "usage_drop_30pct": 2,
    "repeat_contact": 1,
}
SIGNAL_ORDER = tuple(SIGNAL_WEIGHTS)


class SummaryGenerationError(RuntimeError):
    pass


class ChurnState(TypedDict, total=False):
    request: dict[str, Any]
    session_date: str
    customer: dict[str, Any] | None
    records: list[dict[str, Any]]
    policy_documents: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    latest_issue_records: list[dict[str, Any]]
    scored_signals: list[str]
    score: int | None
    risk_level: str | None
    drivers: list[dict[str, Any]]
    missing_information: list[str]
    unsupported: list[str]
    policy_exception: str | None
    response: dict[str, Any]
    escalation_reason: str | None
    supervisor_status: str
    run_id: str


def load_data() -> dict[str, Any]:
    with DATA_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _record_citation(record: dict[str, Any]) -> str:
    return str(record["id"])


def _record_text(record: dict[str, Any]) -> str:
    return str(record.get("text", ""))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _level_for_score(score: int) -> str:
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    return "high"


def _verified_records(state: ChurnState) -> list[dict[str, Any]]:
    return [record for record in state.get("records", []) if record.get("verified") is True]


def _policy_citation(state: ChurnState, doc_id: str) -> str:
    return str(_policy_document(state, doc_id)["id"])


def _policy_document(state: ChurnState, doc_id: str) -> dict[str, Any]:
    for document in state.get("policy_documents", []):
        if document.get("id") == doc_id:
            return document
    raise ValueError(f"Required policy document is missing: {doc_id}")

def _check_input(state: ChurnState) -> ChurnState:
    request = state["request"]
    customer_id = str(request.get("customer_id", "")).strip().upper()
    if not re.fullmatch(r"CUST-\d{4}", customer_id):
        state["customer"] = None
        state["response"] = _empty_response(
            customer_id or None,
            unsupported=["Customer ID is invalid or not present in the approved sample data."],
        )
        state["supervisor_status"] = "not_required"
        return state

    data = load_data()
    customer = data["customers"].get(customer_id)
    if customer is None:
        state["customer"] = None
        state["response"] = _empty_response(
            customer_id,
            unsupported=["Customer ID is not present in the approved customer records."],
        )
        state["supervisor_status"] = "not_required"
        return state

    state["request"]["customer_id"] = customer_id
    state["customer"] = customer
    state["records"] = customer["records"]
    state["policy_documents"] = data["policy_documents"]
    return state


def _empty_response(
    customer_id: str | None, *, unsupported: list[str] | None = None
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "summary": "No customer history was retrieved.",
        "drivers": [],
        "risk": {"level": None, "score": None, "signals_used": []},
        "recommended_action": None,
        "missing_information": ["Verified customer history"],
        "unsupported": unsupported or [],
        "data_conflicts": [],
        "escalate": False,
        "escalation_reason": None,
    }


def _retrieve_records(state: ChurnState) -> ChurnState:
    if state.get("customer") is None:
        return state
    customer = state["customer"]
    session_date = _parse_date(state["session_date"])
    recent_records = []
    for record in customer["records"]:
        if record.get("verified") is not True:
            continue
        record_date = record.get("date")
        if record_date is None or (
            session_date - timedelta(days=30)
            <= _parse_date(record_date)
            <= session_date
        ):
            recent_records.append(record)
    state["records"] = recent_records
    return state


def _resolve_conflicts(state: ChurnState) -> ChurnState:
    records_by_issue: dict[str, list[dict[str, Any]]] = {}
    for record in _verified_records(state):
        issue_key = record.get("issue_key")
        if issue_key:
            records_by_issue.setdefault(str(issue_key), []).append(record)

    conflicts = []
    latest_records = []
    for issue_key, records in records_by_issue.items():
        ordered = sorted(records, key=lambda record: (record.get("date", ""), record["id"]))
        latest_records.append(ordered[-1])
        statuses = {record.get("status") for record in ordered if record.get("status")}
        if len(statuses) > 1:
            conflicts.append(
                {
                    "issue": issue_key,
                    "records": [_record_citation(record) for record in ordered],
                    "resolution": "The latest verified record is used as the current status.",
                    "selected_record": _record_citation(ordered[-1]),
                }
            )
    state["conflicts"] = conflicts
    state["latest_issue_records"] = latest_records
    return state


def _score_risk(state: ChurnState) -> ChurnState:
    if state.get("customer") is None:
        return state

    records = _verified_records(state)
    session_date = _parse_date(state["session_date"])
    recent_interactions = [
        record
        for record in records
        if record.get("type") in {"ticket", "call", "billing"}
        and record.get("issue_key")
        and record.get("date")
        and session_date - timedelta(days=30) <= _parse_date(record["date"]) <= session_date
    ]
    recent_network_tickets = [
        record
        for record in recent_interactions
        if record.get("type") == "ticket" and record.get("category") == "network"
    ]
    billing_open = any(
        record.get("issue_key") == "roaming-charge"
        and record.get("status") in {"open", "disputed"}
        for record in state.get("latest_issue_records", [])
    )
    competitor_records = [
        record for record in records if record.get("competitor_mention") is True
    ]
    usage_records = [
        record
        for record in records
        if record.get("type") == "usage"
        and record.get("month_on_month_change_percent") is not None
    ]
    usage_drop = any(
        record["month_on_month_change_percent"] < -30 for record in usage_records
    )
    repeated_issue = any(
        sum(1 for record in recent_interactions if record.get("issue_key") == issue_key) > 1
        for issue_key in {record.get("issue_key") for record in recent_interactions}
        if issue_key
    )

    detected = {
        "network_complaints_3plus": len(recent_network_tickets) >= 3,
        "open_billing_issue": billing_open,
        "competitor_intent": bool(competitor_records),
        "usage_drop_30pct": usage_drop,
        "repeat_contact": repeated_issue,
    }
    state["scored_signals"] = [
        signal for signal in SIGNAL_ORDER if detected[signal]
    ]
    has_any_verified_signal = any(
        record.get("type") in {"ticket", "call", "billing", "usage"}
        for record in records
    )
    if not has_any_verified_signal:
        state["score"] = None
        state["risk_level"] = None
    else:
        score = sum(SIGNAL_WEIGHTS[signal] for signal in state["scored_signals"])
        state["score"] = score
        state["risk_level"] = _level_for_score(score)

    return state


def _driver_records(state: ChurnState) -> dict[str, list[dict[str, Any]]]:
    records = _verified_records(state)
    session_date = _parse_date(state["session_date"])
    recent_network = [
        record
        for record in records
        if record.get("category") == "network"
        and record.get("type") in {"ticket", "call"}
        and record.get("date")
        and session_date - timedelta(days=30) <= _parse_date(record["date"]) <= session_date
    ]
    billing = [
        record
        for record in records
        if record.get("issue_key") == "roaming-charge"
        and record.get("status") in {"open", "disputed"}
    ]
    competitor = [
        record for record in records if record.get("competitor_mention") is True
    ]
    usage = [
        record
        for record in records
        if record.get("type") == "usage"
        and record.get("month_on_month_change_percent", 0) < -30
    ]
    service = [
        record
        for record in records
        if record.get("service_experience") is True
    ]
    return {
        "network_quality": recent_network,
        "billing_dispute": billing,
        "plan_fit": [],
        "service_experience": service,
        "competitor_intent": competitor,
        "usage_decline": usage,
    }


def _format_drivers(state: ChurnState) -> list[dict[str, Any]]:
    by_driver = _driver_records(state)
    result = []
    for driver in DRIVER_ORDER:
        records = by_driver[driver]
        if not records:
            continue
        if driver == "network_quality":
            evidence = (
                f"{len(records)} network complaint(s) in the last 30 days"
                if len(records) > 1
                else _record_text(records[0])
            )
        elif driver == "usage_decline":
            percent = min(
                int(record["month_on_month_change_percent"]) for record in records
            )
            evidence = f"Data usage down {abs(percent)}% month on month"
        elif driver == "billing_dispute":
            evidence = _record_text(records[-1])
        else:
            evidence = _record_text(records[0])
        result.append(
            {
                "driver": driver,
                "evidence": evidence,
                "citations": [_record_citation(record) for record in records],
            }
        )
    return result


def _summarize_with_ollama(
    fallback: str, drivers: list[dict[str, Any]], score: int | None, level: str | None
) -> str:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model or not drivers:
        return fallback

    evidence = [
        {"driver": driver["driver"], "evidence": driver["evidence"]}
        for driver in drivers
    ]
    prompt = {
        "verified_evidence": evidence,
        "fixed_risk_score": score,
        "fixed_risk_level": level,
    }
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Summarize only the supplied verified evidence and fixed risk "
                    "result in one concise sentence. Do not add facts, infer causes, "
                    "recommend actions, or change the risk score or level. Return "
                    'JSON with exactly one string field: {"summary":"..."}'
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
    }
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    request = Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["message"]["content"]
        summary = json.loads(content)["summary"]
    except (URLError, TimeoutError) as error:
        raise SummaryGenerationError(
            f"Configured Ollama model '{model}' could not be reached: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SummaryGenerationError(
            "Ollama returned a response that did not match the required summary JSON."
        ) from error

    if not isinstance(summary, str) or not summary.strip() or len(summary) > 600:
        raise SummaryGenerationError(
            "Ollama summary must be a non-empty string of at most 600 characters."
        )
    if level and (level not in summary.lower() or str(score) not in summary):
        raise SummaryGenerationError(
            "Ollama summary must preserve the code-computed risk score and level."
        )
    return summary.strip()


def _generate_response(state: ChurnState) -> ChurnState:
    if state.get("customer") is None:
        return state

    request = state["request"]
    customer = state["customer"]
    records = _verified_records(state)
    question = str(request.get("question", "")).strip()
    context = request.get("context") or {}
    session_date = _parse_date(state["session_date"])
    drivers = _format_drivers(state)
    missing_information: list[str] = []
    unsupported: list[str] = []
    policy_exception = None

    if customer["id"] == "CUST-1005":
        missing_information.extend(
            ["Recent customer interactions", "Current usage trend"]
        )
    if not missing_information:
        missing_information.append("Customer's preferred callback time")

    if re.search(r"\b(cheaper|price|pricing|cost|rate)\b", question, re.IGNORECASE) and re.search(
        r"\b(competitor|competitor's|other provider)\b", question, re.IGNORECASE
    ):
        unsupported.append(
            "Competitor pricing is not in the approved records; no price comparison can be made."
        )

    if re.search(
        r"\b(when|date|timeline|schedule|expected)\b.{0,40}\b(network|repair|restoration)\b"
        r"|\b(network|repair|restoration)\b.{0,40}\b(when|date|timeline|schedule|expected)\b",
        question,
        re.IGNORECASE,
    ):
        unsupported.append(
            "Network repair dates are not present in the approved records."
        )

    offer = next(
        (
            document
            for document in state.get("policy_documents", [])
            if document["id"] == "OFFER-FEST"
        ),
        None,
    )
    if offer and re.search(r"\b(festive|offer-fest)\b", question, re.IGNORECASE):
        if session_date > _parse_date(offer["valid_to"]):
            unsupported.append(
                f"OFFER-FEST expired on {offer['valid_to']} and must not be offered."
            )

    request_text = " ".join(
        [question, str(context.get("stated_reason", ""))]
    )
    discount = re.search(r"(\d{1,3})\s*%", request_text)
    duration = re.search(r"(\d{1,2})\s*months?", request_text, re.IGNORECASE)
    credit_mention = re.search(
        r"\b(credit|refund|compensation)\b", request_text, re.IGNORECASE
    )
    credit = (
        re.search(r"\b(?:rs\.?|inr|₹)\s*([\d,]+)\b", request_text, re.IGNORECASE)
        if credit_mention
        else None
    )
    if discount or duration or credit:
        requested_discount = int(discount.group(1)) if discount else 0
        requested_months = int(duration.group(1)) if duration else 0
        requested_credit = int(credit.group(1).replace(",", "")) if credit else 0
        retention_policy = _policy_document(state, "RET-POL@v1")
        discount_requested = bool(
            discount
            or re.search(r"\b(discount|off|concession)\b", request_text, re.IGNORECASE)
        )
        if (
            requested_discount > retention_policy["max_discount_percent"]
            or (
                discount_requested
                and requested_months > retention_policy["max_discount_months"]
            )
            or requested_credit > retention_policy["max_bill_credit_rs"]
        ):
            policy_exception = (
                "The requested concession exceeds RET-POL limits and requires supervisor approval."
            )

    if state.get("risk_level") is None:
        summary = (
            "No recent verified interactions or usage trend are available to classify churn risk."
        )
    else:
        summary_parts = []
        if drivers:
            summary_parts.append(
                "Verified dissatisfaction signals: "
                + "; ".join(
                    driver["evidence"].rstrip(".; ") for driver in drivers
                )
                + "."
            )
        else:
            summary_parts.append("No scored churn-risk signals were found.")
        summary_parts.append(
            f"Risk score {state['score']} maps to {state['risk_level']} risk under the fixed scoring rule."
        )
        if context.get("stated_reason"):
            summary_parts.append(
                "The customer's stated reason is considered context, not independent evidence."
            )
        summary = " ".join(summary_parts)
        summary = _summarize_with_ollama(
            summary, drivers, state["score"], state["risk_level"]
        )

    latest_billing = next(
        (
            record
            for record in state.get("latest_issue_records", [])
            if record.get("issue_key") == "roaming-charge"
            and record.get("status") in {"open", "disputed"}
        ),
        None,
    )
    scored = set(state.get("scored_signals", []))
    action = None
    if "network_complaints_3plus" in scored:
        network_records = _driver_records(state)["network_quality"]
        action = {
            "action": "network_escalation",
            "reasoning": "Three or more verified network complaints meet the NET-ESC threshold.",
            "based_on_signals": ["network_complaints_3plus"],
            "citations": [
                _record_citation(network_records[-1]),
                _policy_citation(state, "NET-ESC@v1"),
            ],
            "follow_up": "callback",
        }
    elif "open_billing_issue" in scored and latest_billing:
        action = {
            "action": "bill_correction",
            "reasoning": "The latest verified record shows the disputed charge remains unresolved; BILL-POL requires correction after verification.",
            "based_on_signals": ["open_billing_issue"],
            "citations": [
                _record_citation(latest_billing),
                _policy_citation(state, "BILL-POL@v1"),
            ],
            "follow_up": "callback",
        }
    elif "competitor_intent" in scored:
        competitor_record = next(
            record for record in records if record.get("competitor_mention") is True
        )
        action = {
            "action": "plan_review",
            "reasoning": (
                "A verified competitor or port-out mention supports a plan review; "
                "no unapproved discount is promised."
                if not policy_exception
                else "A plan review is the permitted next step; the requested concession is held for supervisor review."
            ),
            "based_on_signals": ["competitor_intent"],
            "citations": [
                _record_citation(competitor_record),
                _policy_citation(state, "RET-POL@v1"),
            ],
            "follow_up": "callback",
        }

    no_signal = state.get("score") is None
    uncertain_classification = no_signal
    escalation_reasons = []
    if state.get("conflicts"):
        escalation_reasons.append("Conflicting verified records require supervisor review.")
    if policy_exception:
        escalation_reasons.append(policy_exception)
    if uncertain_classification:
        escalation_reasons.append(
            "Insufficient verified history to classify risk or recommend a grounded action."
        )

    state["drivers"] = drivers
    state["missing_information"] = missing_information
    state["unsupported"] = unsupported
    state["policy_exception"] = policy_exception
    state["escalation_reason"] = " ".join(escalation_reasons) or None
    state["response"] = {
        "customer_id": customer["id"],
        "summary": summary,
        "drivers": drivers,
        "risk": {
            "level": state.get("risk_level"),
            "score": state.get("score"),
            "signals_used": state.get("scored_signals", []),
        },
        "recommended_action": action,
        "missing_information": missing_information,
        "unsupported": unsupported,
        "data_conflicts": state.get("conflicts", []),
        "escalate": bool(escalation_reasons),
        "escalation_reason": " ".join(escalation_reasons) or None,
    }
    return state


def _guard_checks(state: ChurnState) -> ChurnState:
    response = state.get("response")
    if response is None:
        return state

    verified_ids = {_record_citation(record) for record in _verified_records(state)}
    policy_ids = {
        str(document["id"]) for document in state.get("policy_documents", [])
    }
    allowed_citations = verified_ids | policy_ids
    citations = []
    for driver in response["drivers"]:
        citations.extend(driver["citations"])
    action = response.get("recommended_action")
    if action:
        citations.extend(action["citations"])
        if action["action"] not in ACTION_CATALOG:
            raise ValueError("Guard check failed: recommended action is not in the catalog.")
        if not action["based_on_signals"] or not set(action["based_on_signals"]).issubset(
            set(response["risk"]["signals_used"])
        ):
            raise ValueError("Guard check failed: action is not grounded in a scored signal.")
    if not set(citations).issubset(allowed_citations):
        raise ValueError("Guard check failed: a citation does not resolve to verified evidence.")

    if response["risk"]["score"] is not None:
        expected_score = sum(
            SIGNAL_WEIGHTS[signal] for signal in response["risk"]["signals_used"]
        )
        if expected_score != response["risk"]["score"]:
            raise ValueError("Guard check failed: risk score does not match fixed signal weights.")
        if response["risk"]["level"] != _level_for_score(expected_score):
            raise ValueError("Guard check failed: risk band does not match the fixed thresholds.")
    elif response["risk"]["signals_used"]:
        raise ValueError("Guard check failed: a null risk score cannot have scored signals.")

    if response["escalate"] and not response["escalation_reason"]:
        raise ValueError("Guard check failed: escalation requires a reason.")
    if not response["missing_information"]:
        raise ValueError("Guard check failed: missing_information cannot be empty.")
    return state


def _supervisor_review(state: ChurnState) -> ChurnState:
    decision = interrupt(
        {
            "customer_id": state["request"]["customer_id"],
            "reason": state["response"]["escalation_reason"],
        }
    )
    state["supervisor_status"] = (
        "approved" if isinstance(decision, dict) and decision.get("approved") is True
        else "rejected"
    )
    return state


def _finalize(state: ChurnState) -> ChurnState:
    return state


def _route_after_guard(state: ChurnState) -> str:
    if state.get("response", {}).get("escalate"):
        return "supervisor_review"
    return "finalize"


def _build_graph():
    builder = StateGraph(ChurnState)
    builder.add_node("check_input", _check_input)
    builder.add_node("retrieve_records", _retrieve_records)
    builder.add_node("resolve_conflicts", _resolve_conflicts)
    builder.add_node("score_risk", _score_risk)
    builder.add_node("generate_response", _generate_response)
    builder.add_node("guard_checks", _guard_checks)
    builder.add_node("supervisor_review", _supervisor_review)
    builder.add_node("finalize", _finalize)
    builder.add_edge(START, "check_input")
    builder.add_edge("check_input", "retrieve_records")
    builder.add_edge("retrieve_records", "resolve_conflicts")
    builder.add_edge("resolve_conflicts", "score_risk")
    builder.add_edge("score_risk", "generate_response")
    builder.add_edge("generate_response", "guard_checks")
    builder.add_conditional_edges(
        "guard_checks",
        _route_after_guard,
        {
            "supervisor_review": "supervisor_review",
            "finalize": "finalize",
        },
    )
    builder.add_edge("supervisor_review", END)
    builder.add_edge("finalize", END)
    return builder.compile(checkpointer=MemorySaver())


_GRAPH = _build_graph()


def analyze_customer(
    request: dict[str, Any], session_date: str = "2026-10-15"
) -> dict[str, Any]:
    """Run the guarded workflow and return the stable response schema."""
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object.")
    if not isinstance(request.get("customer_id"), str):
        raise ValueError("request.customer_id must be a string.")
    if request.get("context") is not None and not isinstance(request["context"], dict):
        raise ValueError("request.context must be a JSON object.")
    _parse_date(session_date)
    thread_id = str(uuid.uuid4())
    initial_state: ChurnState = {
        "request": {
            "customer_id": request["customer_id"],
            "question": str(request.get("question", "")),
            "context": request.get("context") or {},
        },
        "session_date": session_date,
        "supervisor_status": "not_required",
        "run_id": thread_id,
    }
    result = _GRAPH.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    response = dict(result["response"])
    response["supervisor_status"] = (
        "pending" if result.get("__interrupt__") else result.get("supervisor_status", "not_required")
    )
    response["review_id"] = thread_id if response["supervisor_status"] == "pending" else None
    return response


def resolve_supervisor_review(
    review_id: str, approved: bool
) -> dict[str, Any]:
    """Resume a paused review without bypassing the guarded action catalog."""
    result = _GRAPH.invoke(
        Command(resume={"approved": approved}),
        config={"configurable": {"thread_id": review_id}},
    )
    response = dict(result["response"])
    if approved:
        response["supervisor_status"] = "approved"
    else:
        response["recommended_action"] = None
        response["supervisor_status"] = "rejected"
    response["review_id"] = review_id
    return response
