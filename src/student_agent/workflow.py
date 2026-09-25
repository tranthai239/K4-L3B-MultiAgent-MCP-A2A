"""L3B Coordinator — multi-agent workflow orchestration.

Architecture:
  Coordinator → EntityResolver
             → [Order/Item, Payment, Shipment] (parallel)
             → Policy (sequential, needs all specialist results)
             → Verifier
             → Output
"""
from __future__ import annotations

import asyncio
from typing import Any

from .agents import (
    EntityResolverAgent,
    OrderItemAgent,
    PaymentAgent,
    PolicyAgent,
    ShipmentAgent,
    VerifierAgent,
)
from .mcp_gateway import EvidenceGateway
from .trace import TraceWriter


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter,
) -> dict[str, Any]:
    """Coordinator: orchestrate specialist agents, assemble L3B output."""
    case_id = case["case_id"]

    # --- Phase 1: Entity Resolution ---
    entity_agent = EntityResolverAgent()
    entity = await entity_agent.run(case, gateway, trace)

    resolved_ids = entity.resolved_order_ids or []
    # Fallback: use claimed_order_id if entity resolution failed
    if not resolved_ids:
        claimed = case.get("customer_request", {}).get("claimed_order_id")
        if claimed:
            resolved_ids = [claimed]

    # --- Phase 2: Parallel specialist investigation ---
    order_agent = OrderItemAgent()
    shipment_agent = ShipmentAgent()
    payment_agent = PaymentAgent()

    order_result, shipment_result, payment_result = await asyncio.gather(
        order_agent.run(case, resolved_ids, gateway, trace, entity.order_data),
        shipment_agent.run(case, resolved_ids, gateway, trace),
        payment_agent.run(case, resolved_ids, gateway, trace, entity.order_data),
    )

    # Merge payment references from payment evidence into order result
    for p in payment_result.payment_data:
        ref = p.get("payment_sequential") or p.get("payment_type")
        oid = order_result.order_ids[0] if order_result.order_ids else ""
        if ref:
            ref_str = f"{oid}_{ref}" if oid else str(ref)
            if ref_str not in order_result.payment_references:
                order_result.payment_references.append(ref_str)

    # --- Phase 3: Policy analysis (needs all specialist results) ---
    policy_agent = PolicyAgent()
    policy = await policy_agent.run(
        case, entity, order_result, shipment_result, payment_result,
        gateway, trace,
    )

    # --- Assemble output ---
    # Collect all evidence refs (deduplicated, ordered)
    all_refs = list(dict.fromkeys(
        entity.evidence_refs
        + order_result.evidence_refs
        + shipment_result.evidence_refs
        + payment_result.evidence_refs
        + policy.evidence_refs
    ))

    output: dict[str, Any] = {
        "schema_version": "day09-l3b-output-v2",
        "case_id": case_id,
        "assessment": {
            "primary_issue": policy.primary_issue,
            "secondary_issues": policy.secondary_issues,
            "case_status": policy.case_status,
            "confidence": entity.confidence * 0.9,  # tempered by entity resolution
        },
        "affected_entities": {
            "order_ids": order_result.order_ids,
            "item_ids": order_result.item_ids,
            "seller_ids": order_result.seller_ids,
            "payment_references": order_result.payment_references,
            "shipment_ids": order_result.shipment_ids,
        },
        "entity_resolution": {
            "status": entity.status,
            "resolved_order_ids": entity.resolved_order_ids,
            "rejected_candidates": entity.rejected_candidates,
            "confidence": entity.confidence,
        },
        "customer_context": {
            "customer_unique_id": entity.customer_unique_id,
            "related_order_ids": entity.related_order_ids,
        },
        "shipment_analysis": {
            "verdict": shipment_result.verdict,
            "late_seller_ids": shipment_result.late_seller_ids,
            "timeline_complete": shipment_result.timeline_complete,
        },
        "payment_analysis": {
            "verdict": payment_result.verdict,
            "captured_total_brl": payment_result.captured_total_brl,
            "refunded_total_brl": payment_result.refunded_total_brl,
            "refundable_total_brl": payment_result.refundable_total_brl,
        },
        "root_cause_analysis": {
            "ranked_causes": policy.ranked_causes,
            "responsible_parties": policy.responsible_parties,
        },
        "evidence_refs": all_refs[:30],
        "data_conflicts": policy.data_conflicts,
        "financial_resolution": policy.financial_resolution,
        "resolution_actions": policy.resolution_actions[:8],
    }

    # Add optional claim_assessments if claims exist
    if policy.claim_assessments:
        output["claim_assessments"] = policy.claim_assessments[:5]

    # --- Phase 4: Verification ---
    verifier = VerifierAgent()
    output = await verifier.run(case, output, trace)

    return output
