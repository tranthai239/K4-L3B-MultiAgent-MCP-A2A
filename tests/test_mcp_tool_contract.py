from __future__ import annotations

import asyncio
from typing import Any

from mcp.types import CallToolResult

from student_agent.agents import (
    EntityResolverAgent,
    EntityResult,
    OrderItemAgent,
    OrderResult,
    PaymentAgent,
    PaymentResult,
    PolicyAgent,
    ShipmentAgent,
    ShipmentResult,
    VerifierAgent,
)
from student_agent.mcp_gateway import EvidenceGateway
from student_agent.workflow import solve_case


class RecordingGateway:
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, str]]] = []
        self.responses = responses or {}

    async def call(
        self, tool_name: str, *, case_id: str, **arguments: str
    ) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, arguments))
        return {
            "evidence_ref": f"ev_{'x' * 20}",
            "data": self.responses.get(tool_name, {}),
        }


class RecordingTrace:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, **event: Any) -> dict[str, Any]:
        self.events.append(event)
        return event


class ValidatingContracts:
    def validate_evidence(self, value: Any, label: str) -> None:
        assert value["schema_version"] == "day09-mcp-evidence-v1"
        assert label == "MCP tool get_order"


class MCP22Session:
    async def call_tool(self, tool_name: str, arguments: dict[str, str]) -> CallToolResult:
        return CallToolResult(
            content=[],
            structured_content={
                "schema_version": "day09-mcp-evidence-v1",
                "evidence_ref": f"ev_{'x' * 20}",
                "result_hash": f"sha256:{'0' * 64}",
                "domain": "order",
                "data": {"order_id": arguments["order_id"]},
            },
            is_error=False,
        )


def test_gateway_reads_mcp_22_call_result_fields() -> None:
    gateway = EvidenceGateway(MCP22Session(), ValidatingContracts())

    result = asyncio.run(
        gateway.call("get_order", case_id="L3B_CASE_001", order_id="order-1")
    )

    assert result["data"]["order_id"] == "order-1"


def test_agents_parse_authoritative_list_and_history_shapes() -> None:
    case = {
        "case_id": "L3B_CASE_001",
        "opened_at": "2018-02-01T09:00:00-03:00",
        "candidate_order_ids": ["order-1"],
        "customer_request": {"claimed_order_id": "order-1"},
        "customer_unique_id_hint": "customer-1",
    }
    gateway = RecordingGateway(
        {
            "get_order": {"order_id": "order-1"},
            "get_customer_history": {
                "customer_unique_id": "customer-1",
                "orders": [
                    {
                        "order_id": "order-1",
                        "order_purchase_timestamp": "2018-01-01T09:00:00-03:00",
                        "order_status": "delivered",
                    },
                    {
                        "order_id": "order-1",
                        "order_purchase_timestamp": "2018-03-01T09:00:00-03:00",
                        "order_status": "canceled",
                    },
                    {"order_id": "order-2"},
                ],
            },
            "get_order_items": [
                {"order_item_id": "item-1", "seller_id": "seller-1"}
            ],
            "get_payment_timeline": {
                "payments": [
                    {
                        "payment_sequential": "1",
                        "payment_type": "credit_card",
                        "payment_value": "89.00",
                    }
                ],
                "events": [
                    {
                        "event_type": "captured",
                        "amount_brl": "89.00",
                        "status": "confirmed",
                    }
                ],
            },
        }
    )
    trace = RecordingTrace()

    async def run() -> tuple[Any, Any, Any]:
        entity = await EntityResolverAgent().run(case, gateway, trace)
        order = await OrderItemAgent().run(case, ["order-1"], gateway, trace)
        payment = await PaymentAgent().run(case, ["order-1"], gateway, trace)
        return entity, order, payment

    entity, order, payment = asyncio.run(run())

    assert entity.related_order_ids == ["order-1", "order-2"]
    assert entity.order_data["order_status"] == "delivered"
    assert not any(name == "get_order" for name, _, _ in gateway.calls)
    assert order.item_ids == ["item-1"]
    assert order.seller_ids == ["seller-1"]
    assert payment.captured_total_brl == 89.0


def test_shipment_agent_parses_authoritative_summary_fields() -> None:
    case = {"case_id": "L3B_CASE_001"}
    gateway = RecordingGateway(
        {
            "get_shipment_summary": {
                "order_status": "delivered",
                "delivered_carrier_at": "2018-05-13T09:00:00-03:00",
                "delivered_customer_at": "2018-05-22T09:00:00-03:00",
                "estimated_delivery_at": "2018-05-21T09:00:00-03:00",
                "events": [
                    {
                        "event_type": "delivered_late",
                        "actor": "logistics_provider",
                        "status": "confirmed",
                    }
                ],
            }
        }
    )

    result = asyncio.run(
        ShipmentAgent().run(case, ["order-1"], gateway, RecordingTrace())
    )

    assert result.timeline_complete is True
    assert result.verdict == "logistics_delay"


def test_payment_agent_uses_timeline_events_for_verdict_and_totals() -> None:
    case = {
        "case_id": "L3B_CASE_005",
        "opened_at": "2018-05-05T09:00:00-03:00",
        "customer_request": {"claimed_order_id": "order-1"},
    }
    gateway = RecordingGateway(
        {
            "get_order_payments": [],
            "get_payment_timeline": {
                "payments": [{"payment_value": "999.00"}],
                "events": [
                    {
                        "event_at": "2018-01-01T09:00:00-03:00",
                        "event_type": "captured",
                        "amount_brl": "35.00",
                        "status": "confirmed",
                    },
                    {
                        "event_at": "2018-05-01T09:00:00-03:00",
                        "event_type": "captured",
                        "amount_brl": "89.00",
                        "status": "confirmed",
                    },
                    {
                        "event_at": "2018-06-01T09:00:00-03:00",
                        "event_type": "captured",
                        "amount_brl": "999.00",
                        "status": "confirmed",
                    }
                ],
            },
            "get_refund_timeline": {
                "events": [
                    {
                        "event_type": "refund_requested",
                        "amount_brl": "89.00",
                        "status": "pending",
                    }
                ]
            },
        }
    )

    result = asyncio.run(
        PaymentAgent().run(
            case,
            ["order-1"],
            gateway,
            RecordingTrace(),
            {"order_purchase_timestamp": "2018-04-23T09:00:00-03:00"},
        )
    )

    assert result.verdict == "refund_pending"
    assert result.captured_total_brl == 89.0
    assert result.refunded_total_brl == 0.0
    assert result.refundable_total_brl == 89.0


def test_payment_agent_distinguishes_split_payment_from_duplicate_capture() -> None:
    case = {"case_id": "L3B_CASE_002"}
    split = [
        {
            "payment_sequential": "1",
            "payment_type": "credit_card",
            "payment_value": "44.50",
        },
        {
            "payment_sequential": "2",
            "payment_type": "voucher",
            "payment_value": "44.50",
        },
    ]
    split_gateway = RecordingGateway(
        {
            "get_order_payments": split,
            "get_payment_timeline": {"payments": split, "events": []},
            "get_refund_timeline": {},
        }
    )
    duplicate_gateway = RecordingGateway(
        {
            "get_order_payments": [*split, *split],
            "get_payment_timeline": {"payments": [*split, *split], "events": []},
            "get_refund_timeline": {},
        }
    )

    split_result = asyncio.run(
        PaymentAgent().run(case, ["order-1"], split_gateway, RecordingTrace())
    )
    duplicate_result = asyncio.run(
        PaymentAgent().run(case, ["order-1"], duplicate_gateway, RecordingTrace())
    )

    assert split_result.verdict == "reconciled"
    assert duplicate_result.verdict == "duplicate_capture"


def test_policy_agent_uses_authoritative_policy_rule() -> None:
    case = {
        "case_id": "L3B_CASE_001",
        "policy_version": "EC_POLICY_V2",
        "customer_request": {
            "claims": [
                {"claim_id": "claim-1", "topic": "late_delivery_logistics"},
                {"claim_id": "claim-2", "topic": "requested_full_refund"},
            ]
        },
    }
    policy_rule = {
        "case_status": "action_required",
        "recommended_action": "refund_freight",
        "refund_brl": 16.0,
        "responsible_parties": [
            {"party_type": "logistics_provider", "party_id": None}
        ],
    }
    gateway = RecordingGateway(
        {
            "get_policy": {
                "currency": "BRL",
                "policy_version": "EC_POLICY_V2",
                "rules": {"late_delivery_logistics": policy_rule},
            }
        }
    )
    shipment = ShipmentResult(
        verdict="logistics_delay",
        timeline_complete=True,
        evidence_refs=[f"ev_{'s' * 20}"],
    )
    payment = PaymentResult(
        verdict="reconciled",
        captured_total_brl=105.0,
        refunded_total_brl=0.0,
        refundable_total_brl=105.0,
        evidence_refs=[f"ev_{'p' * 20}"],
    )

    result = asyncio.run(
        PolicyAgent().run(
            case,
            EntityResult(status="resolved", resolved_order_ids=["order-1"]),
            OrderResult(order_ids=["order-1"]),
            shipment,
            payment,
            gateway,
            RecordingTrace(),
        )
    )

    assert result.primary_issue == "late_delivery_logistics"
    assert result.case_status == "action_required"
    assert result.secondary_issues == []
    assert result.responsible_parties == policy_rule["responsible_parties"]
    assert result.financial_resolution == {
        "currency": "BRL",
        "recommended_refund_brl": 16.0,
        "refund_lines": [
            {
                "reason_code": "late_delivery_logistics",
                "amount_brl": 16.0,
                "entity_id": "order-1",
            }
        ],
    }
    assert result.resolution_actions == ["refund_freight"]
    assert result.claim_assessments == [
        {
            "claim_id": "claim-1",
            "verdict": "supported",
            "confidence": 0.8,
            "evidence_refs": shipment.evidence_refs,
        },
        {
            "claim_id": "claim-2",
            "verdict": "partially_supported",
            "confidence": 0.8,
            "evidence_refs": [
                *payment.evidence_refs,
                *result.evidence_refs,
            ],
        },
    ]


def test_policy_primary_issue_requires_matching_evidence() -> None:
    cases = [
        ("valid_split_payment", PaymentResult(verdict="reconciled"), "valid_split_payment"),
        ("payment_mismatch", PaymentResult(verdict="capture_mismatch"), "payment_mismatch"),
        ("duplicate_charge", PaymentResult(verdict="reconciled"), "unsupported_claim"),
        ("refund_pending", PaymentResult(verdict="refund_failed"), "unsupported_claim"),
        ("refund_failed", PaymentResult(verdict="refunded"), "unsupported_claim"),
    ]

    for topic, payment, expected in cases:
        case = {
            "case_id": "L3B_CASE_001",
            "policy_version": "EC_POLICY_V2",
            "customer_request": {
                "claims": [{"claim_id": "claim-1", "topic": topic}]
            },
        }
        result = asyncio.run(
            PolicyAgent().run(
                case,
                EntityResult(status="resolved", resolved_order_ids=["order-1"]),
                OrderResult(order_ids=["order-1"]),
                ShipmentResult(),
                payment,
                RecordingGateway({"get_policy": {"rules": {}}}),
                RecordingTrace(),
            )
        )
        assert result.primary_issue == expected


def test_verifier_calibrates_confidence_for_conflicts() -> None:
    trace = RecordingTrace()
    output = {
        "case_id": "L3B_CASE_001",
        "assessment": {"case_status": "action_required", "confidence": 0.95},
        "affected_entities": {"order_ids": ["order-1"]},
        "entity_resolution": {"resolved_order_ids": ["order-1"]},
        "payment_analysis": {"verdict": "reconciled"},
        "financial_resolution": {"recommended_refund_brl": 16.0},
        "resolution_actions": ["refund_freight"],
        "evidence_refs": [f"ev_{letter * 20}" for letter in "abcde"],
        "data_conflicts": [{"field": "shipment_status"}],
    }

    verified = asyncio.run(
        VerifierAgent().run({"case_id": "L3B_CASE_001"}, output, trace)
    )

    assert verified["assessment"]["confidence"] <= 0.7
    assert trace.events[-1]["event_type"] == "verification_completed"


def test_verifier_repairs_cross_field_inconsistencies() -> None:
    output = {
        "case_id": "L3B_CASE_001",
        "assessment": {
            "primary_issue": "valid_split_payment",
            "case_status": "no_action",
            "confidence": 0.95,
        },
        "affected_entities": {"order_ids": ["wrong-order"]},
        "entity_resolution": {
            "status": "resolved",
            "resolved_order_ids": ["order-1"],
            "confidence": 0.9,
        },
        "payment_analysis": {"verdict": "reconciled"},
        "financial_resolution": {
            "recommended_refund_brl": 16.0,
            "refund_lines": [{"amount_brl": 16.0}],
        },
        "resolution_actions": ["refund_freight"],
        "evidence_refs": [f"ev_{letter * 20}" for letter in "abcde"],
        "data_conflicts": [],
    }

    verified = asyncio.run(
        VerifierAgent().run(
            {"case_id": "L3B_CASE_001"}, output, RecordingTrace()
        )
    )

    assert verified["affected_entities"]["order_ids"] == ["order-1"]
    assert verified["financial_resolution"] == {
        "recommended_refund_brl": 0.0,
        "refund_lines": [],
    }
    assert verified["resolution_actions"] == []
    assert verified["assessment"]["confidence"] <= 0.7


def test_workflow_emits_required_trace_lifecycle_locally() -> None:
    case = {
        "case_id": "L3B_CASE_001",
        "opened_at": "2018-02-01T09:00:00-03:00",
        "candidate_order_ids": ["order-1"],
        "customer_unique_id_hint": "customer-1",
        "policy_version": "EC_POLICY_V2",
        "customer_request": {
            "claimed_order_id": "order-1",
            "claims": [{"claim_id": "claim-1", "topic": "valid_split_payment"}],
        },
    }
    gateway = RecordingGateway(
        {
            "get_customer_history": {
                "customer_unique_id": "customer-1",
                "orders": [
                    {
                        "order_id": "order-1",
                        "order_purchase_timestamp": "2018-01-01T09:00:00-03:00",
                    }
                ],
            },
            "get_order_items": [],
            "get_shipment_summary": {},
            "get_payment_timeline": {
                "payments": [
                    {
                        "payment_sequential": "1",
                        "payment_type": "credit_card",
                        "payment_value": "89.00",
                    }
                ],
                "events": [
                    {
                        "event_type": "captured",
                        "amount_brl": "89.00",
                        "status": "confirmed",
                    }
                ],
            },
            "get_refund_timeline": {"events": []},
            "get_policy": {
                "currency": "BRL",
                "rules": {
                    "valid_split_payment": {
                        "case_status": "no_action",
                        "recommended_action": None,
                        "refund_brl": 0,
                        "responsible_parties": [],
                    }
                },
            },
        }
    )
    trace = RecordingTrace()
    trace.emit(case_id=case["case_id"], event_type="case_received", actor="coordinator")

    asyncio.run(solve_case(case, gateway, trace))
    trace.emit(case_id=case["case_id"], event_type="case_finalized", actor="coordinator")

    event_types = [event["event_type"] for event in trace.events]
    required = [
        "case_received",
        "task_assigned",
        "tool_result_consumed",
        "handoff",
        "policy_decided",
        "verification_completed",
        "case_finalized",
    ]
    positions = [event_types.index(event_type) for event_type in required]
    assert positions == sorted(positions)
    consumed = [
        event for event in trace.events
        if event["event_type"] == "tool_result_consumed"
    ]
    assert consumed
    assert all(event.get("evidence_refs") for event in consumed)


def test_specialists_use_discovered_mcp_tool_names_and_case_scope() -> None:
    case = {"case_id": "L3B_CASE_001"}
    gateway = RecordingGateway()
    trace = RecordingTrace()

    async def run() -> None:
        await ShipmentAgent().run(case, ["order-1"], gateway, trace)
        await PaymentAgent().run(case, ["order-1"], gateway, trace)

    asyncio.run(run())

    assert [(name, case_id) for name, case_id, _ in gateway.calls] == [
        ("get_shipment_summary", "L3B_CASE_001"),
        ("get_payment_timeline", "L3B_CASE_001"),
        ("get_refund_timeline", "L3B_CASE_001"),
    ]
