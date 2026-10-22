import pytest

from churn_assistant import analyze_customer, load_data, resolve_supervisor_review


@pytest.fixture(autouse=True)
def disable_optional_ollama(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)


def analyze(customer_id, question="Why might this customer leave, and what should I do?"):
    return analyze_customer(
        {
            "customer_id": customer_id,
            "question": question,
            "context": {"stated_reason": "Calls keep dropping at home"},
        }
    )


def citation_ids(result):
    citations = []
    for driver in result["drivers"]:
        citations.extend(driver["citations"])
    if result["recommended_action"]:
        citations.extend(result["recommended_action"]["citations"])
    return citations


def test_happy_path_uses_fixed_score_and_cited_network_action():
    result = analyze("CUST-1001")
    assert result["risk"] == {
        "level": "high",
        "score": 6,
        "signals_used": [
            "network_complaints_3plus",
            "usage_drop_30pct",
            "repeat_contact",
        ],
    }
    assert [driver["driver"] for driver in result["drivers"]] == [
        "network_quality",
        "usage_decline",
    ]
    assert result["recommended_action"]["action"] == "network_escalation"
    assert "NOTE-77" not in citation_ids(result)
    assert result["escalate"] is False


def test_unknown_customer_is_rejected_without_invented_history():
    result = analyze("CUST-9999")
    assert result["drivers"] == []
    assert result["risk"]["score"] is None
    assert result["recommended_action"] is None
    assert result["unsupported"]


def test_unverified_note_never_changes_score_or_citations():
    result = analyze("CUST-1001")
    assert result["risk"]["score"] == 6
    assert "NOTE-77" not in citation_ids(result)


def test_latest_verified_billing_record_wins_and_conflict_escalates():
    result = analyze("CUST-1002")
    assert result["risk"] == {
        "level": "medium",
        "score": 4,
        "signals_used": ["open_billing_issue", "repeat_contact"],
    }
    assert result["recommended_action"]["action"] == "bill_correction"
    assert result["recommended_action"]["citations"][0] == "CALL-8821"
    assert result["data_conflicts"][0]["selected_record"] == "CALL-8821"
    assert result["escalate"] is True
    assert result["supervisor_status"] == "pending"


def test_competitor_pricing_question_is_not_answered():
    result = analyze("CUST-1004", "Is the competitor's plan cheaper?")
    assert any("Competitor pricing" in item for item in result["unsupported"])


def test_network_repair_dates_are_not_invented():
    result = analyze("CUST-1001", "When will the network repair be completed?")
    assert any("Network repair dates" in item for item in result["unsupported"])


def test_expired_offer_is_not_recommended_or_quoted():
    result = analyze("CUST-1003", "Can I give this customer the festive offer?")
    assert any("expired" in item for item in result["unsupported"])
    assert result["recommended_action"] is None


def test_policy_exception_pauses_for_supervisor_and_can_resume():
    result = analyze(
        "CUST-1004",
        "The customer wants 50% off for 6 months. Can we offer it?",
    )
    assert result["escalate"] is True
    assert result["supervisor_status"] == "pending"
    assert result["recommended_action"]["action"] == "plan_review"
    approved = resolve_supervisor_review(result["review_id"], True)
    assert approved["supervisor_status"] == "approved"
    assert approved["recommended_action"]["action"] == "plan_review"


def test_supervisor_rejection_withholds_the_action():
    result = analyze(
        "CUST-1004",
        "The customer wants 50% off for 6 months. Can we offer it?",
    )
    rejected = resolve_supervisor_review(result["review_id"], False)
    assert rejected["supervisor_status"] == "rejected"
    assert rejected["recommended_action"] is None


def test_bill_credit_over_policy_limit_requires_supervisor():
    result = analyze(
        "CUST-2001",
        "Can I give the customer a Rs 1,000 bill credit?",
    )
    assert result["escalate"] is True
    assert "exceeds RET-POL limits" in result["escalation_reason"]
    assert result["risk"]["level"] == "low"


def test_sparse_history_has_no_invented_drivers_and_escalates():
    result = analyze("CUST-1005")
    assert result["drivers"] == []
    assert result["risk"] == {"level": None, "score": None, "signals_used": []}
    assert result["missing_information"] == [
        "Recent customer interactions",
        "Current usage trend",
    ]
    assert result["escalate"] is True


def test_sample_data_has_record_metadata_and_learner_added_customer():
    data = load_data()
    assert "CUST-2001" in data["customers"]
    for customer_id, customer in data["customers"].items():
        for record in customer["records"]:
            assert record["customer_id"] == customer_id
            assert record["id"]
            assert record["type"]
            assert record["date"]
            assert record["source_system"]
            assert isinstance(record["verified"], bool)
