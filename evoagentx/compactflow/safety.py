"""Auditable Appendix B.2 Eq. (38) counts, distinct from all-call incidences."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .runtime import _MISSING, _path_get
from .schema import CallState, TraceKind

CATEGORIES = ("argument_mismatch", "duplicate_dispatch", "effect_order", "capacity_overload")


@dataclass(frozen=True)
class CallAudit:
    call_id: str
    early: bool
    failures: tuple[str, ...] = ()
    assessed: bool = True


def summarize_safety(calls: Iterable[CallAudit]) -> dict:
    """Count each early logical call once; priority follows CATEGORIES.

    Counts from different executions must be summarized separately, then pooled
    by summing numerators/denominators. Reusing a call ID here is an input error.
    """
    calls = list(calls)
    if len({c.call_id for c in calls}) != len(calls):
        raise ValueError("duplicate audit identity; supply one audit per logical call")
    if any(set(c.failures) - set(CATEGORIES) for c in calls):
        raise ValueError("unknown contract-failure category")
    early = [c for c in calls if c.early]
    counts = dict.fromkeys(CATEGORIES, 0)
    primary = {}
    for call in early:
        category = next((key for key in CATEGORIES if key in call.failures), None)
        if category:
            counts[category] += 1
            primary[call.call_id] = category
    unassessed = sum(not c.assessed for c in early)
    total = sum(counts.values())
    rates = {key: counts[key] / len(early) if early and not unassessed else None for key in CATEGORIES}
    rates["aggregate"] = total / len(early) if early and not unassessed else None
    return {"definition": "Appendix B.2 Eq.38", "denominator_name": "early_logical_calls",
            "denominator": len(early), "all_logical_calls": len(calls), "unassessed_early_calls": unassessed,
            "counts": {**counts, "aggregate": total}, "rates": rates, "rate_unit": "fraction",
            "primary_category_priority": list(CATEGORIES), "primary_failures": primary,
            "zero_denominator": "undefined", "assessment_complete": unassessed == 0,
            "all_call_failure_incidents": sum(len(set(c.failures)) for c in calls)}


def audit_execution(graph, result, *, reference_arguments: dict | None = None) -> dict:
    """Check actual starts against final/reference inputs, effects and capacity.

    Refused dispatch attempts are not actual duplicate dispatches. An unavailable
    final/reference argument leaves that early call unassessed, never safe by fiat.
    """
    starts = [e for e in result.trace if e.kind is TraceKind.START]
    counts = Counter(e.call_id for e in starts)
    failures = {cid: set() for cid in counts}
    early, assessed = {}, {}
    completed, effects, active = set(), set(), Counter()
    for event in result.trace:
        cid = event.call_id
        if event.kind is TraceKind.START:
            data = graph.incoming_data(cid)
            early[cid] = early.get(cid, False) or any(d.producer is not None and d.producer not in completed for d in data)
            if counts[cid] > 1:
                failures[cid].add("duplicate_dispatch")
            for dependency in graph.incoming_effects(cid):
                if dependency.producer not in completed and not (dependency.allow_partial and (dependency.producer, dependency.effect) in effects):
                    failures[cid].add("effect_order")
            active[cid] += 1
            for resource, capacity in graph.resource_capacity.items():
                if sum(graph.call_map[c].resources.get(resource) * n for c, n in active.items()) > capacity + 1e-12:
                    failures[cid].add("capacity_overload")
        elif event.kind in {TraceKind.COMPLETE, TraceKind.FAILURE, TraceKind.CANCEL}:
            if active[cid]:
                active[cid] -= 1
            if event.kind is TraceKind.COMPLETE:
                completed.add(cid)
        effects.update((cid, effect) for effect in event.detail.get("effects", ()))
    for cid in counts:
        arguments = result.call_traces[cid].arguments
        assessed[cid] = True
        if reference_arguments is not None and cid in reference_arguments:
            if arguments != reference_arguments[cid]:
                failures[cid].add("argument_mismatch")
        else:
            for d in graph.incoming_data(cid):
                if d.producer is None:
                    continue
                if result.states[d.producer] is not CallState.COMPLETED:
                    if early[cid]:
                        assessed[cid] = False
                    continue
                expected = _path_get(result.outputs[d.producer], d.source_path, _MISSING)
                observed = _path_get(arguments, d.target_path, _MISSING)
                if expected != observed or (d.required and observed is _MISSING):
                    failures[cid].add("argument_mismatch")
    return summarize_safety(CallAudit(cid, early[cid], tuple(failures[cid]), assessed[cid]) for cid in counts)
