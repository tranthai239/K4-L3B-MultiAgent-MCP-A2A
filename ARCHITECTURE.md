# L3B Architecture Record

## 1. System overview

```text
Input (case JSON)
  │
  ▼
┌──────────────────────────┐
│   Coordinator / Router   │  (workflow.py:solve_case)
└─────────────┬────────────┘
              │ Phase 1: Entity Resolution
              ▼
     ┌──────────────────┐
     │ Entity Resolver   │  resolve order from candidates + customer history
     └────────┬─────────┘
              │ Handoff (resolved_order_ids)
              │ Phase 2: Parallel Investigation
  ┌───────────┼───────────────┐
  ▼           ▼               ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│Order/  │ │ Payment  │ │ Shipment │
│Item    │ │ Agent    │ │ Agent    │
└───┬────┘ └────┬─────┘ └────┬─────┘
    │           │             │
    └───────────┼─────────────┘
                │ Phase 3: Policy (sequential)
                ▼
       ┌──────────────────┐
       │   Policy Agent   │  synthesize findings, apply EC_POLICY_V2
       └────────┬─────────┘
                │ Phase 4: Verification
                ▼
       ┌──────────────────┐
       │  Verifier Agent  │  cross-field consistency, confidence calibration
       └────────┬─────────┘
                │ Validated Output
                ▼
           L3B JSON Output
```

All MCP calls go through `EvidenceGateway`. All observable events go through `TraceWriter`.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| Coordinator | case JSON | Orchestrate phases, assemble output | none (delegates) | Final L3B output |
| Entity Resolver | case.candidate_order_ids, customer_unique_id_hint | Resolve correct order, fetch customer history | `get_customer_history`, fallback `get_order` | EntityResult → coordinator |
| Order/Item Agent | resolved_order_ids | Fetch items and extract entities | `get_order_items` | OrderResult → coordinator |
| Shipment Agent | resolved_order_ids | Analyze delivery timeline | `get_shipment_summary` | ShipmentResult → coordinator |
| Payment Agent | resolved_order_ids | Reconcile payments/refunds | `get_payment_timeline`, `get_refund_timeline` | PaymentResult → coordinator |
| Policy Agent | All specialist results | Apply business rules, determine root cause, financial resolution | `get_policy` | PolicyResult → coordinator |
| Verifier | Assembled output | Cross-field consistency, confidence calibration | none (read-only) | Corrected output → coordinator |

Least privilege: each agent can only call its allowed MCP tools. Verifier has no MCP access.

## 3. Entity resolution và A2A protocol

### Entity resolution
1. Call `get_customer_history` once and match candidate snapshots at or before `opened_at`.
2. If history has no order data, call `get_order` for each candidate as fallback.
3. Prefer `claimed_order_id` if it resolves successfully.
4. Single resolved → status `resolved`, confidence 0.9.
5. Multiple resolved → status `ambiguous`, confidence 0.5.
6. None resolved → status `not_found`, confidence 0.3.

### Customer context
- Use `customer_unique_id_hint` → `get_customer_history`.
- Extract `related_order_ids` for cross-reference.

### A2A protocol
- **Message envelope**: Python dataclasses (`EntityResult`, `OrderResult`, etc.) passed directly between agents via coordinator.
- **Correlation**: All operations keyed by `case_id`. Evidence never crosses cases.
- **Handoff**: Coordinator passes resolved_order_ids to specialists. Each specialist emits `handoff` trace event on completion.
- **No cycles**: Strictly sequential phases — Entity → Specialists (parallel) → Policy → Verifier. No back-loops.
- **Timeout**: MCP calls have 300s timeout (from gateway config). Agent-level timeout is bounded by retry budget.

## 4. Evidence và conflict lifecycle

1. **MCP call** → `EvidenceGateway.call()` → validated against `mcp-evidence-response-v1.schema.json`.
2. **evidence_ref** stored as-is from MCP response. Never fabricated or modified.
3. **tool_result_consumed** trace event emitted immediately after successful MCP call.
4. **Evidence refs accumulated** per agent, deduplicated at output assembly.
5. **Conflict detection**: Policy Agent compares verdicts across sources. Conflicts recorded in `data_conflicts[]` with `selected_source` and `resolution_code`.
6. **Source precedence**: shipment_tracking > order_status > customer_claim.
7. **No cross-case evidence**: Each `solve_case` invocation is independent. No state between calls.

## 5. Failure and efficiency policy

| Failure | Retry budget | Fallback | Trace event/code |
| --- | ---: | --- | --- |
| MCP timeout | 2 retries, exponential backoff (0.5s, 1s) | Return None, use `insufficient_evidence` | No consumption event without evidence |
| Entity not found/ambiguous | 0 (single attempt per candidate) | Use `claimed_order_id` as fallback | `handoff` with `decision_code=not_found` |
| Source conflict | 0 | Record in `data_conflicts`, prefer tracking data | `policy_decided` with conflict noted |
| Invalid specialist result | 0 | Skip specialist, continue with available data | No extra event; verifier catches gaps |

### Efficiency strategy
- **No duplicate calls**: Each order_id queried once per tool per agent.
- **Parallel specialists**: Order/Item, Shipment, Payment run concurrently via `asyncio.gather`.
- **Minimal tools per agent**: Least privilege prevents unnecessary calls.
- **Entity resolver gates specialists**: If no order resolves, specialists still run with `claimed_order_id` but budget is minimal.

## 6. Verification invariants

Before finalize, Verifier checks:

1. **Schema**: `case_id` matches input case.
2. **Entity scope**: `resolved_order_ids` ⊆ `affected_entities.order_ids`.
3. **Rejected candidates**: All non-resolved candidates listed.
4. **Evidence ownership**: All `evidence_refs` obtained from MCP in this case's scope.
5. **Confidence bounds**: 0 ≤ confidence ≤ 1, clamped if violated.
6. **Financial consistency**: `no_action` forces zero refund and empty refund lines.
7. **Status/action consistency**: `no_action` → no resolution_actions.
8. **Evidence completeness**: < 3 evidence refs → confidence capped at 0.5.
9. **Claim linkage**: Each claim_assessment has evidence_refs from relevant specialist.

## 7. Reproducibility

- **Framework**: Pure Python async state machine (no LangGraph/CrewAI dependency).
- **Python**: ≥ 3.11
- **Dependencies**: pinned in `pyproject.toml` — httpx2, jsonschema, mcp, python-dotenv.
- **Concurrency**: `asyncio.gather` for parallel specialists (3 concurrent).
- **Random seed**: None used. `secrets.token_urlsafe` for event_id only (not deterministic, not needed).
- **Run command**: `day09 run`
- **Validate**: `day09 validate`
- **Package**: `day09 package --output dist/submission.zip`
- **Resource limits**: MCP timeout 300s, retry budget 2 per call, max 30 evidence_refs per output.
