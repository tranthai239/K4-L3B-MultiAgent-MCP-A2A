"""Specialist agents for L3B multi-agent workflow.

Each agent is a simple async callable with explicit tool permissions.
No framework — pure Python async state machine.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2


async def _safe_call(
    gateway: EvidenceGateway,
    tool_name: str,
    *,
    case_id: str,
    trace: TraceWriter,
    actor: str,
    **kwargs: str,
) -> dict[str, Any] | None:
    """Call MCP tool with retry. Returns None on permanent failure."""
    for attempt in range(1 + _MAX_RETRIES):
        try:
            evidence = await gateway.call(tool_name, case_id=case_id, **kwargs)
            # emit tool_result_consumed
            trace.emit(
                case_id=case_id,
                event_type="tool_result_consumed",
                actor=actor,
                tool_name=tool_name,
                evidence_refs=[evidence["evidence_ref"]],
                attributes={"attempt": attempt + 1} if attempt > 0 else None,
            )
            return evidence
        except RuntimeError:
            return None
        except Exception:  # noqa: BLE001
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(0.5 * (2**attempt))
    return None


# ---------------------------------------------------------------------------
# Agent result containers
# ---------------------------------------------------------------------------

@dataclass
class EntityResult:
    status: str = "not_found"  # resolved | ambiguous | not_found
    resolved_order_ids: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    confidence: float = 0.3
    customer_unique_id: str | None = None
    related_order_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    order_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    order_ids: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    seller_ids: list[str] = field(default_factory=list)
    payment_references: list[str] = field(default_factory=list)
    shipment_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    order_data: dict[str, Any] = field(default_factory=dict)
    items_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ShipmentResult:
    verdict: str = "insufficient_evidence"
    late_seller_ids: list[str] = field(default_factory=list)
    timeline_complete: bool = False
    evidence_refs: list[str] = field(default_factory=list)
    shipment_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaymentResult:
    verdict: str = "insufficient_evidence"
    captured_total_brl: float | None = None
    refunded_total_brl: float | None = None
    refundable_total_brl: float | None = None
    evidence_refs: list[str] = field(default_factory=list)
    payment_data: list[dict[str, Any]] = field(default_factory=list)
    payment_events: list[dict[str, Any]] = field(default_factory=list)
    refund_data: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PolicyResult:
    primary_issue: str = "insufficient_evidence"
    secondary_issues: list[str] = field(default_factory=list)
    case_status: str = "needs_investigation"
    ranked_causes: list[dict[str, Any]] = field(default_factory=list)
    responsible_parties: list[dict[str, Any]] = field(default_factory=list)
    data_conflicts: list[dict[str, Any]] = field(default_factory=list)
    financial_resolution: dict[str, Any] = field(default_factory=dict)
    resolution_actions: list[str] = field(default_factory=list)
    claim_assessments: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Entity Resolver — resolve order from candidates, get customer context
# ---------------------------------------------------------------------------

class EntityResolverAgent:
    """Resolve which candidate_order_id is correct, fetch customer history."""

    ACTOR = "entity-resolver"
    ALLOWED_TOOLS = {"get_order", "get_customer_history"}

    async def run(
        self, case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter,
    ) -> EntityResult:
        case_id = case["case_id"]
        trace.emit(
            case_id=case_id, event_type="task_assigned",
            actor="coordinator", target=self.ACTOR,
        )

        result = EntityResult()
        candidates = case.get("candidate_order_ids", [])
        claimed = case.get("customer_request", {}).get("claimed_order_id")
        valid_orders: list[tuple[str, dict[str, Any]]] = []

        cust_hint = case.get("customer_unique_id_hint")
        if cust_hint:
            ev = await _safe_call(
                gateway, "get_customer_history", case_id=case_id, trace=trace,
                actor=self.ACTOR, customer_unique_id=cust_hint,
            )
            if ev is not None:
                result.evidence_refs.append(ev["evidence_ref"])
                cust_data = ev.get("data", {})
                result.customer_unique_id = cust_data.get("customer_unique_id", cust_hint)
                orders = cust_data.get("orders", [])
                result.related_order_ids = list(dict.fromkeys(
                    item["order_id"] for item in orders if item.get("order_id")
                ))
                opened_at = case.get("opened_at", "")
                snapshots = [
                    item for item in orders
                    if item.get("order_id") in candidates
                    and item.get("order_purchase_timestamp", "") <= opened_at
                ]
                latest = {
                    order_id: max(
                        (item for item in snapshots if item["order_id"] == order_id),
                        key=lambda item: item.get("order_purchase_timestamp", ""),
                    )
                    for order_id in dict.fromkeys(item["order_id"] for item in snapshots)
                }
                valid_orders = list(latest.items())

        if not valid_orders and not result.related_order_ids:
            for order_id in candidates:
                ev = await _safe_call(
                    gateway, "get_order", case_id=case_id, trace=trace,
                    actor=self.ACTOR, order_id=order_id,
                )
                if ev is not None:
                    result.evidence_refs.append(ev["evidence_ref"])
                    data = ev.get("data", {})
                    if data and data.get("order_id"):
                        valid_orders.append((order_id, data))

        resolved = [order_id for order_id, _ in valid_orders]
        if claimed and claimed in resolved:
            result.resolved_order_ids = [claimed]
        else:
            result.resolved_order_ids = resolved
        result.rejected_candidates = [
            order_id for order_id in candidates
            if order_id not in result.resolved_order_ids
        ]
        result.status = (
            "resolved" if len(result.resolved_order_ids) == 1
            else "ambiguous" if result.resolved_order_ids
            else "not_found"
        )
        result.confidence = {
            "resolved": 0.9,
            "ambiguous": 0.5,
            "not_found": 0.3,
        }[result.status]
        if result.resolved_order_ids:
            result.order_data = next(
                data for order_id, data in valid_orders
                if order_id == result.resolved_order_ids[0]
            )

        # Handoff to coordinator
        trace.emit(
            case_id=case_id, event_type="handoff",
            actor=self.ACTOR, target="coordinator",
            decision_code=result.status,
        )
        return result


# ---------------------------------------------------------------------------
# Order/Item Agent
# ---------------------------------------------------------------------------

class OrderItemAgent:
    """Fetch order details, items, products. Extract affected entities."""

    ACTOR = "order-item-agent"
    ALLOWED_TOOLS = {"get_order", "get_order_items", "get_product_context", "get_sellers"}

    async def run(
        self,
        case: dict[str, Any],
        resolved_order_ids: list[str],
        gateway: EvidenceGateway,
        trace: TraceWriter,
        resolved_order_data: dict[str, Any] | None = None,
    ) -> OrderResult:
        case_id = case["case_id"]
        trace.emit(
            case_id=case_id, event_type="task_assigned",
            actor="coordinator", target=self.ACTOR,
        )

        result = OrderResult()
        for order_id in resolved_order_ids:
            result.order_ids.append(order_id)

            result.order_data = resolved_order_data or {}

            # Get items
            ev = await _safe_call(
                gateway, "get_order_items", case_id=case_id, trace=trace,
                actor=self.ACTOR, order_id=order_id,
            )
            if ev is not None:
                result.evidence_refs.append(ev["evidence_ref"])
                items = ev.get("data", [])
                if isinstance(items, list):
                    result.items_data = items
                    for item in items:
                        iid = item.get("item_id") or item.get("order_item_id")
                        if iid and iid not in result.item_ids:
                            result.item_ids.append(iid)
                        sid = item.get("seller_id")
                        if sid and sid not in result.seller_ids:
                            result.seller_ids.append(sid)
                        ship_id = item.get("shipment_id") or item.get("tracking_id")
                        if ship_id and ship_id not in result.shipment_ids:
                            result.shipment_ids.append(ship_id)

        trace.emit(
            case_id=case_id, event_type="handoff",
            actor=self.ACTOR, target="coordinator",
        )
        return result


# ---------------------------------------------------------------------------
# Shipment Agent
# ---------------------------------------------------------------------------

class ShipmentAgent:
    """Analyze shipment timeline, determine delivery verdict."""

    ACTOR = "shipment-agent"
    ALLOWED_TOOLS = {"get_shipment_summary"}

    async def run(
        self, case: dict[str, Any], resolved_order_ids: list[str],
        gateway: EvidenceGateway, trace: TraceWriter,
    ) -> ShipmentResult:
        case_id = case["case_id"]
        trace.emit(
            case_id=case_id, event_type="task_assigned",
            actor="coordinator", target=self.ACTOR,
        )

        result = ShipmentResult()
        for order_id in resolved_order_ids:
            ev = await _safe_call(
                gateway, "get_shipment_summary", case_id=case_id, trace=trace,
                actor=self.ACTOR, order_id=order_id,
            )
            if ev is not None:
                result.evidence_refs.append(ev["evidence_ref"])
                data = ev.get("data", {})
                result.shipment_data = data
                result.timeline_complete = bool(data.get("delivered_customer_at"))
                result.verdict = _analyze_shipment_verdict(data)
                # Identify late sellers
                if result.verdict == "seller_delay":
                    result.late_seller_ids = list(dict.fromkeys(
                        limit["seller_id"]
                        for limit in data.get("shipping_limits", [])
                        if limit.get("seller_id")
                    ))

        trace.emit(
            case_id=case_id, event_type="handoff",
            actor=self.ACTOR, target="coordinator",
        )
        return result


# ---------------------------------------------------------------------------
# Payment Agent
# ---------------------------------------------------------------------------

class PaymentAgent:
    """Analyze payments and refunds, reconcile totals."""

    ACTOR = "payment-agent"
    ALLOWED_TOOLS = {"get_order_payments", "get_payment_timeline", "get_refund_timeline"}

    async def run(
        self,
        case: dict[str, Any],
        resolved_order_ids: list[str],
        gateway: EvidenceGateway,
        trace: TraceWriter,
        resolved_order_data: dict[str, Any] | None = None,
    ) -> PaymentResult:
        case_id = case["case_id"]
        trace.emit(
            case_id=case_id, event_type="task_assigned",
            actor="coordinator", target=self.ACTOR,
        )

        result = PaymentResult()
        for order_id in resolved_order_ids:
            # Payment lifecycle
            ev = await _safe_call(
                gateway, "get_payment_timeline", case_id=case_id, trace=trace,
                actor=self.ACTOR, order_id=order_id,
            )
            if ev is not None:
                result.evidence_refs.append(ev["evidence_ref"])
                timeline = ev.get("data", {})
                result.payment_data = timeline.get("payments", [])
                opened_at = case.get("opened_at", "")
                purchased_at = (resolved_order_data or {}).get(
                    "order_purchase_timestamp", ""
                )
                result.payment_events = [
                    event
                    for event in timeline.get("events", [])
                    if not event.get("event_at")
                    or purchased_at <= event["event_at"] <= opened_at
                ]
                captured = sum(
                    float(event.get("amount_brl", 0))
                    for event in result.payment_events
                    if event.get("event_type") == "captured"
                    and event.get("status") == "confirmed"
                )
                result.captured_total_brl = round(captured, 2) if captured else None

            # Refund details
            ev = await _safe_call(
                gateway, "get_refund_timeline", case_id=case_id, trace=trace,
                actor=self.ACTOR, order_id=order_id,
            )
            if ev is not None:
                result.evidence_refs.append(ev["evidence_ref"])
                data = ev.get("data", {})
                opened_at = case.get("opened_at", "")
                purchased_at = (resolved_order_data or {}).get(
                    "order_purchase_timestamp", ""
                )
                result.refund_data = [
                    event
                    for event in data.get("events", [])
                    if not event.get("event_at")
                    or purchased_at <= event["event_at"] <= opened_at
                ]
                refunded = sum(
                    float(event.get("amount_brl", 0))
                    for event in result.refund_data
                    if event.get("status") == "confirmed"
                )
                result.refunded_total_brl = round(refunded, 2)

        # Determine verdict
        result.verdict = _analyze_payment_verdict(result)
        # Refundable = captured - refunded
        if result.captured_total_brl is not None:
            already = result.refunded_total_brl or 0.0
            result.refundable_total_brl = round(
                max(0, result.captured_total_brl - already), 2,
            )

        trace.emit(
            case_id=case_id, event_type="handoff",
            actor=self.ACTOR, target="coordinator",
        )
        return result


# ---------------------------------------------------------------------------
# Policy Agent — synthesize findings, apply policy, resolve conflicts
# ---------------------------------------------------------------------------

class PolicyAgent:
    """Apply business policy to collected evidence. Determine root cause,
    financial resolution, and resolution actions."""

    ACTOR = "policy-agent"
    ALLOWED_TOOLS = {"get_policy"}

    async def run(
        self,
        case: dict[str, Any],
        entity: EntityResult,
        order: OrderResult,
        shipment: ShipmentResult,
        payment: PaymentResult,
        gateway: EvidenceGateway,
        trace: TraceWriter,
    ) -> PolicyResult:
        case_id = case["case_id"]
        trace.emit(
            case_id=case_id, event_type="task_assigned",
            actor="coordinator", target=self.ACTOR,
        )

        result = PolicyResult()

        # Fetch policy
        policy_version = case.get("policy_version", "EC_POLICY_V2")
        ev = await _safe_call(
            gateway, "get_policy", case_id=case_id, trace=trace,
            actor=self.ACTOR, policy_version=policy_version,
        )
        policy_data: dict[str, Any] = {}
        if ev is not None:
            result.evidence_refs.append(ev["evidence_ref"])
            policy_data = ev.get("data", {})

        claims = case.get("customer_request", {}).get("claims", [])

        # Determine primary issue from claims + evidence
        result.primary_issue = _determine_primary_issue(claims, order, shipment, payment)
        result.secondary_issues = _extract_secondary_issues(claims, result.primary_issue)
        rule = policy_data.get("rules", {}).get(result.primary_issue, {})
        result.case_status = rule.get(
            "case_status", _determine_case_status(result.primary_issue, payment)
        )

        # Root cause analysis
        result.ranked_causes = _build_ranked_causes(result.primary_issue, shipment)
        result.responsible_parties = rule.get("responsible_parties") or (
            _build_responsible_parties(result.primary_issue, shipment, order)
        )

        # Detect data conflicts between sources
        result.data_conflicts = _detect_conflicts(shipment, payment)

        # Financial resolution
        result.financial_resolution = _build_financial_resolution(
            result.primary_issue, payment, order, policy_data, rule,
        )

        # Resolution actions
        result.resolution_actions = _build_resolution_actions(
            result.primary_issue, result.case_status, rule,
        )

        # Claim assessments
        result.claim_assessments = _build_claim_assessments(
            claims, shipment, payment, entity, result,
        )

        trace.emit(
            case_id=case_id, event_type="policy_decided",
            actor=self.ACTOR,
            decision_code=result.primary_issue,
        )
        trace.emit(
            case_id=case_id, event_type="handoff",
            actor=self.ACTOR, target="coordinator",
        )
        return result


# ---------------------------------------------------------------------------
# Verifier Agent — final consistency check before output
# ---------------------------------------------------------------------------

class VerifierAgent:
    """Cross-field consistency, confidence calibration, schema check."""

    ACTOR = "verifier"

    async def run(
        self,
        case: dict[str, Any],
        output: dict[str, Any],
        trace: TraceWriter,
    ) -> dict[str, Any]:
        case_id = case["case_id"]
        trace.emit(
            case_id=case_id, event_type="task_assigned",
            actor="coordinator", target=self.ACTOR,
        )

        issues: list[str] = []

        # Check case_id match
        if output.get("case_id") != case_id:
            output["case_id"] = case_id
            issues.append("case_id_corrected")

        # Check entity resolution consistency
        er = output.get("entity_resolution", {})
        resolved = er.get("resolved_order_ids", [])
        affected = output.get("affected_entities", {}).get("order_ids", [])
        if resolved and set(resolved) != set(affected):
            output.setdefault("affected_entities", {})["order_ids"] = resolved
            issues.append("order_id_mismatch_corrected")

        # Check confidence bounds
        assessment = output.get("assessment", {})
        conf = assessment.get("confidence", 0.5)
        if conf < 0 or conf > 1:
            assessment["confidence"] = max(0, min(1, conf))
            issues.append("confidence_clamped")

        # Check financial consistency
        fr = output.get("financial_resolution", {})
        if (
            assessment.get("case_status") == "no_action"
            and fr.get("recommended_refund_brl", 0) > 0
        ):
            fr["recommended_refund_brl"] = 0.0
            fr["refund_lines"] = []
            assessment["confidence"] = min(assessment.get("confidence", 0.5), 0.7)
            issues.append("no_action_refund_removed")

        # Check status/action consistency
        status = assessment.get("case_status")
        actions = output.get("resolution_actions", [])
        if status == "no_action" and actions:
            output["resolution_actions"] = []
            assessment["confidence"] = min(assessment.get("confidence", 0.5), 0.7)
            issues.append("no_action_actions_removed")

        # Calibrate confidence based on evidence completeness and conflicts
        ev_refs = output.get("evidence_refs", [])
        if len(ev_refs) < 3:
            assessment["confidence"] = min(assessment.get("confidence", 0.5), 0.5)
            issues.append("low_evidence_confidence_capped")
        if output.get("data_conflicts"):
            assessment["confidence"] = min(assessment.get("confidence", 0.5), 0.7)
            issues.append("conflict_confidence_capped")

        trace.emit(
            case_id=case_id, event_type="verification_completed",
            actor=self.ACTOR,
            attributes={"issues_found": len(issues), "issues": ",".join(issues[:5])},
        )
        return output


# ---------------------------------------------------------------------------
# Private analysis helpers
# ---------------------------------------------------------------------------

def _as_list(val: Any) -> list:
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def _analyze_shipment_verdict(data: dict[str, Any]) -> str:
    """Determine shipment verdict from MCP evidence data."""
    if not data:
        return "insufficient_evidence"

    status = str(data.get("status", data.get("order_status", ""))).lower()
    delivered = data.get("delivered_customer_at")
    estimated = data.get("estimated_delivery_at")
    shipped = data.get("delivered_carrier_at")
    events = data.get("events", [])
    carrier_delay = any(
        event.get("event_type") == "delivered_late"
        and event.get("actor") == "logistics_provider"
        and event.get("status") == "confirmed"
        for event in events
    )
    seller_delay = any(
        event.get("event_type") == "delivered_late"
        and event.get("actor") == "seller"
        and event.get("status") == "confirmed"
        for event in events
    )

    if seller_delay:
        return "seller_delay"
    if carrier_delay:
        return "logistics_delay"
    if status in ("canceled", "cancelled"):
        return "returned"
    if not shipped and not delivered:
        return "lost" if status == "lost" else "insufficient_evidence"
    if delivered and estimated and str(delivered) > str(estimated):
        return "logistics_delay"
    if delivered:
        return "on_time"
    return "insufficient_evidence"


def _analyze_payment_verdict(result: PaymentResult) -> str:
    """Determine payment verdict from collected data."""
    if not result.payment_data:
        return "insufficient_evidence"

    has_refunds = bool(result.refund_data)
    refunded = result.refunded_total_brl or 0
    # Duplicate means the same payment identity appears more than once.
    payment_keys = [
        (
            payment.get("payment_sequential"),
            payment.get("payment_type"),
            payment.get("payment_value"),
        )
        for payment in result.payment_data
    ]
    if len(set(payment_keys)) < len(payment_keys):
        return "duplicate_capture"

    # Check refund status
    if has_refunds:
        refund_statuses = [str(event.get("status", "")).lower() for event in result.refund_data]
        if any(status in ("pending", "processing") for status in refund_statuses):
            return "refund_pending"
        if any(status in ("failed", "rejected", "denied") for status in refund_statuses):
            return "refund_failed"
        if refunded > 0:
            return "refunded"

    if any(
        event.get("event_type") == "reconciliation_mismatch"
        and event.get("status") == "open"
        for event in result.payment_events
    ):
        return "capture_mismatch"

    return "reconciled"


_CLAIM_TO_ISSUE: dict[str, str] = {
    "late_delivery_logistics": "late_delivery_logistics",
    "late_delivery_seller": "late_delivery_seller",
    "late_delivery": "late_delivery_logistics",
    "canceled_order_paid": "canceled_order_paid",
    "unavailable_order_paid": "unavailable_order_paid",
    "payment_mismatch": "payment_mismatch",
    "duplicate_charge": "duplicate_charge",
    "refund_pending": "refund_pending",
    "refund_failed": "refund_failed",
    "valid_split_payment": "valid_split_payment",
    "requested_full_refund": "refund_pending",
    "unsupported_claim": "unsupported_claim",
}


def _determine_primary_issue(
    claims: list[dict[str, Any]],
    order: OrderResult,
    shipment: ShipmentResult,
    payment: PaymentResult,
) -> str:
    """Map claims + evidence to a primary_issue enum value."""
    # First claim topic drives the primary issue
    if claims:
        topic = claims[0].get("topic", "")
        mapped = _CLAIM_TO_ISSUE.get(topic)
        if mapped:
            # Validate against evidence
            if mapped in ("late_delivery_logistics", "late_delivery_seller"):
                if shipment.verdict == "seller_delay":
                    return "late_delivery_seller"
                if shipment.verdict in ("logistics_delay", "lost"):
                    return "late_delivery_logistics"
                return "insufficient_evidence"
            if mapped == "canceled_order_paid":
                return (
                    mapped
                    if order.order_data.get("order_status") == "canceled"
                    else "unsupported_claim"
                )
            if mapped == "unavailable_order_paid":
                return (
                    mapped
                    if order.order_data.get("order_status") == "unavailable"
                    else "unsupported_claim"
                )
            expected_payment_verdict = {
                "valid_split_payment": "reconciled",
                "payment_mismatch": "capture_mismatch",
                "duplicate_charge": "duplicate_capture",
                "refund_pending": "refund_pending",
                "refund_failed": "refund_failed",
            }.get(mapped)
            if expected_payment_verdict:
                return (
                    mapped if payment.verdict == expected_payment_verdict
                    else "unsupported_claim"
                )
            return mapped

    # Fallback to evidence-driven
    if payment.verdict == "duplicate_capture":
        return "duplicate_charge"
    if payment.verdict == "refund_pending":
        return "refund_pending"
    if payment.verdict == "refund_failed":
        return "refund_failed"
    if shipment.verdict == "seller_delay":
        return "late_delivery_seller"
    if shipment.verdict == "logistics_delay":
        return "late_delivery_logistics"
    return "insufficient_evidence"


def _extract_secondary_issues(claims: list[dict[str, Any]], primary: str) -> list[str]:
    """Extract secondary issues from claims beyond the primary."""
    issues: list[str] = []
    for claim in claims[1:]:
        topic = claim.get("topic", "")
        if topic == "requested_full_refund":
            continue
        mapped = _CLAIM_TO_ISSUE.get(topic, topic)
        if mapped and mapped != primary and mapped not in issues:
            issues.append(mapped)
    return issues[:10]


def _determine_case_status(primary_issue: str, payment: PaymentResult) -> str:
    if primary_issue in ("insufficient_evidence", "unsupported_claim"):
        return "needs_investigation" if primary_issue == "insufficient_evidence" else "no_action"
    if primary_issue in ("valid_split_payment",):
        return "no_action"
    return "action_required"


_ISSUE_TO_CAUSE: dict[str, str] = {
    "late_delivery_seller": "SELLER_SHIPPING_DELAY",
    "late_delivery_logistics": "LOGISTICS_DELIVERY_DELAY",
    "canceled_order_paid": "ORDER_CANCELED_PAYMENT_NOT_REVERSED",
    "unavailable_order_paid": "PRODUCT_UNAVAILABLE_PAYMENT_CAPTURED",
    "payment_mismatch": "PAYMENT_AMOUNT_DISCREPANCY",
    "duplicate_charge": "DUPLICATE_PAYMENT_CAPTURE",
    "refund_pending": "REFUND_NOT_PROCESSED",
    "refund_failed": "REFUND_PROCESSING_FAILURE",
    "valid_split_payment": "VALID_SPLIT_PAYMENT",
    "unsupported_claim": "UNSUPPORTED_CLAIM_TYPE",
    "insufficient_evidence": "INSUFFICIENT_EVIDENCE",
}


def _build_ranked_causes(primary_issue: str, shipment: ShipmentResult) -> list[dict[str, Any]]:
    cause_code = _ISSUE_TO_CAUSE.get(primary_issue, "UNKNOWN_CAUSE")
    causes = [{"cause_code": cause_code, "rank": 1}]
    # Add secondary cause if shipment and payment both have issues
    if (
        primary_issue.startswith("late_delivery")
        and shipment.verdict == "seller_delay"
        and cause_code != "SELLER_SHIPPING_DELAY"
    ):
        causes.append({"cause_code": "SELLER_SHIPPING_DELAY", "rank": 2})
    return causes[:5]


def _build_responsible_parties(
    primary_issue: str, shipment: ShipmentResult, order: OrderResult,
) -> list[dict[str, Any]]:
    parties: list[dict[str, Any]] = []
    if primary_issue == "late_delivery_seller":
        sid = shipment.late_seller_ids[0] if shipment.late_seller_ids else (
            order.seller_ids[0] if order.seller_ids else None
        )
        parties.append({"party_type": "seller", "party_id": sid})
    elif primary_issue == "late_delivery_logistics":
        parties.append({"party_type": "logistics_provider", "party_id": None})
    elif primary_issue in (
        "duplicate_charge", "payment_mismatch", "refund_pending", "refund_failed",
    ):
        parties.append({"party_type": "payment_provider", "party_id": None})
    elif primary_issue in ("canceled_order_paid", "unavailable_order_paid"):
        parties.append({"party_type": "platform", "party_id": None})
    else:
        parties.append({"party_type": "unknown", "party_id": None})
    return parties[:5]


def _detect_conflicts(shipment: ShipmentResult, payment: PaymentResult) -> list[dict[str, Any]]:
    """Detect data conflicts between evidence sources."""
    conflicts: list[dict[str, Any]] = []
    # Example: if shipment says on_time but claim says late
    if shipment.verdict == "conflicting":
        conflicts.append({
            "field": "shipment_status",
            "sources": ["shipment_tracking", "order_status"],
            "selected_source": "shipment_tracking",
            "resolution_code": "prefer_tracking_data",
        })
    return conflicts[:5]


def _build_financial_resolution(
    primary_issue: str,
    payment: PaymentResult,
    order: OrderResult,
    policy_data: dict[str, Any],
    rule: dict[str, Any],
) -> dict[str, Any]:
    recommended = float(rule.get("refund_brl", 0))
    refund_lines: list[dict[str, Any]] = []
    if recommended > 0:
        refund_lines.append({
            "reason_code": primary_issue,
            "amount_brl": recommended,
            "entity_id": order.order_ids[0] if order.order_ids else None,
        })
    return {
        "currency": policy_data.get("currency", "BRL"),
        "recommended_refund_brl": recommended,
        "refund_lines": refund_lines,
    }


def _build_resolution_actions(
    primary_issue: str, case_status: str, rule: dict[str, Any],
) -> list[str]:
    action = rule.get("recommended_action")
    if action:
        return [action]
    if case_status == "no_action":
        return []
    return ["request_additional_documentation"]


def _build_claim_assessments(
    claims: list[dict[str, Any]],
    shipment: ShipmentResult,
    payment: PaymentResult,
    entity: EntityResult,
    policy: PolicyResult,
) -> list[dict[str, Any]]:
    """Assess each customer claim against supporting evidence only."""
    assessments: list[dict[str, Any]] = []

    for claim in claims[:5]:
        claim_id = claim.get("claim_id", "unknown")
        topic = claim.get("topic", "")
        verdict = "insufficient_evidence"
        confidence = 0.3
        claim_refs: list[str] = []

        if topic in ("late_delivery_logistics", "late_delivery", "late_delivery_seller"):
            if shipment.verdict in ("logistics_delay", "seller_delay", "lost"):
                verdict = "supported"
                confidence = 0.8
                claim_refs = shipment.evidence_refs[:10]
            elif shipment.verdict == "on_time":
                verdict = "unsupported"
                confidence = 0.8
                claim_refs = shipment.evidence_refs[:10]

        elif topic == "requested_full_refund":
            recommended = policy.financial_resolution.get(
                "recommended_refund_brl", 0
            )
            captured = payment.captured_total_brl or 0
            if payment.verdict == "refunded":
                verdict = "unsupported"
                confidence = 0.8
            elif recommended > 0:
                verdict = (
                    "supported" if captured > 0 and recommended >= captured
                    else "partially_supported"
                )
                confidence = 0.8
            elif policy.case_status == "no_action":
                verdict = "unsupported"
                confidence = 0.8
            claim_refs = list(dict.fromkeys(
                payment.evidence_refs + policy.evidence_refs
            ))[:10]

        elif topic == "refund_pending":
            if payment.verdict == "refund_pending":
                verdict = "supported"
                confidence = 0.8
            elif payment.verdict in ("refund_failed", "refunded", "reconciled"):
                verdict = "unsupported"
                confidence = 0.8
            claim_refs = payment.evidence_refs[:10]

        elif topic in ("duplicate_charge", "payment_mismatch"):
            if payment.verdict == "duplicate_capture":
                verdict = "supported"
                confidence = 0.8
                claim_refs = payment.evidence_refs[:10]

        elif topic in ("canceled_order_paid", "unavailable_order_paid"):
            verdict = "supported"
            confidence = 0.6

        assessments.append({
            "claim_id": claim_id,
            "verdict": verdict,
            "confidence": confidence,
            "evidence_refs": claim_refs,
        })

    return assessments
