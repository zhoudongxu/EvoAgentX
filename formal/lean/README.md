# CompactFlow conditional safety core (Lean 4)

Pinned compiler: **Lean 4.19.0**; no Mathlib or third-party Lean packages.
Source of compiler: https://github.com/leanprover/lean4/releases/tag/v4.19.0
Language reference: https://lean-lang.org/doc/reference/4.19.0/

Run from this directory with the pinned toolchain on PATH:

```bash
./check.sh
```

On the prepared A100 host:

```bash
cd /data/Optx/CompactFlow/formal/lean
PATH=/data/Optx/tools/lean-4.19.0-linux/bin:$PATH ./check.sh
```

`CompactFlow.lean` defines an operational transition system and proves invariants
by induction over its steps. `Examples.lean` constructs a full six-event run in
which a consumer actually starts before its producer completes. It also proves
that a deliberately omitted real dependency cannot satisfy the exact-footprint
premise. `Check.lean` prints the transitive axioms of the exported theorems;
`check.sh` rejects incomplete proof placeholders and project-specific axioms.
Only standard Lean foundational principles appear (propext and Quot.sound).

## Proven statements

- `reachable_invariant`: every reachable state preserves materialized values,
  argument snapshots, evaluated results, dispatch uniqueness and resource bounds.
- `argument_agreement`: every read in a dispatched call's declared footprint has
  the same value as the fixed complete-dependency reference execution.
- `at_most_once`: the dispatch history contains no duplicate logical call.
- `capacity_safety`: the sum of actual active-call demands stays within each
  dimension of the resource-capacity vector (not merely a separate counter).
- `effect_order`: each dispatch occurs only after its declared effect predecessors
  appear in the completion history.
- `early_dispatch_requires_label`: reading a field of an unfinished producer
  requires an early-safe consumer label.
- `run_publication_immutable`: a committed value never changes in any later state.
- `field_observational_equivalence`: two legal runs that publish the same observed
  field agree on its value, hence agree on all materialized sink fields.

## Explicit hypotheses and exact scope

`Model.eval` fixes the call response as a function of its arguments: this is the
fixed-external-interaction condition, not a theorem about live stochastic models
or changing websites. `Model.exact` requires that evaluation cannot distinguish
argument environments that agree on the declared footprint. This is a substantive
contract assumption; neither JSON schema checks nor this proof infer or measure
its semantic correctness. `Reference` gives the equations satisfied by the
complete-dependency result; it does not assume the early run's result is equal.
The constructive example establishes that these premises admit a nontrivial run.

Publication commits one whole immutable field. Completion follows publication
of declared outputs. A terminal adapter's atomic publish-and-complete is modeled
as these consecutive logical transitions. Raw token prefixes and mutable output
revisions are outside this fragment. Availability, exact reads, early eligibility,
effect order and resource fit are explicit transition premises; the proof states
what follows when they hold, not that an arbitrary adapter enforces them.

This is a mechanized **conditional safety core for successful single-attempt
executions**, not a verification/refinement proof of the Python implementation or
of the entire paper Theorem 1. It does not prove fair-dispatch liveness, termination,
retries, cancellation/descendant invalidation, compensation or replay-miss handling.
It proves equality of field observations; it does not prove equivalence of arbitrary
side-effectful world states or commutation of undeclared effects. Those obligations
and empirical violation counts must not be claimed as discharged by this artifact.

## Implementation correspondence (not a proved refinement)

| Formal object | Python implementation boundary |
|---|---|
| footprint / exact | typed input bindings compiled in paper_workflow.py and compiler.py; semantic completeness remains a premise |
| publish / immutable store | Partial/Complete events, stable-field checks and immutable commits in runtime.py, paper_workflow.py and gaia.py |
| dispatch / earlySafe | compiled readiness guards and registered per-tool early-safe contracts |
| effectBefore | GFRG effect dependencies, including conservative GAIA task-asset barriers |
| demand / capacity / active | runtime resource admission, reservation and release |
| fixed eval | argument-keyed ReplayBundle responses under a fixed captured trace |

The formal artifact is added under `formal/lean`, outside the code/config directories
hashed by the running experiment. No experiment source, prompt, manifest, frozen
library or result has been changed by adding this proof. This artifact cannot explain
or replace measurements of the paper's 0.63%/0.0% violation claims.
