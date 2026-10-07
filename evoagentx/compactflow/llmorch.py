"""Independent reproduction of LLMOrch Sections III-B/C (arXiv:2504.14872v2).

Fixed-call offline study only: logical processor slots reproduce coordination,
not MPI process overhead or CPU workload scaling. No planner/repair/network.
The scheduler waits for ALL predecessors (the prose definition in III-C),
including cross-rank joins; it does not interpret Algorithm 2 as a level barrier.
"""
from __future__ import annotations
import asyncio
from collections import deque

IMPLEMENTATION = "llmorch_paper_reimplementation_v1"


async def coordinate(order, predecessors, demands, capacity, kinds, processors, invoke):
    """FIFO ready batches, I/O priority, exclusive compute slots, bounded resources.

    `invoke` returns only on full completion. Task exceptions stop the run;
    all running tasks are joined on cancellation. Memory is per invocation.
    """
    order = tuple(order)
    ids = set(order)
    if not ids or len(ids) != len(order) or type(processors) is not int or processors < 1:
        raise ValueError("nonempty unique calls and positive processor count required")
    if set(kinds) != ids or set(kinds.values()) - {"compute", "inout"}:
        raise ValueError("explicit compute/inout classification required for every call")
    rank = {}
    for cid in order:
        deps = predecessors[cid]
        if not deps <= set(rank):
            raise ValueError("cyclic, unknown or non-topological predecessor")
        rank[cid] = 1 + max((rank[p] for p in deps), default=0)
        if any(v < 0 or v > capacity.get(k, 0) for k, v in demands[cid].items()):
            raise ValueError("call demand exceeds shared capacity")
    done, submitted = set(), set()
    batches = deque()
    running, active, occupied = {}, dict.fromkeys(capacity, 0), set()
    assignments = {}

    def submit_ready():
        ready = [c for c in order if c not in submitted and predecessors[c] <= done]
        if ready:
            submitted.update(ready)
            batches.append(sorted(ready, key=lambda c: kinds[c] == "compute"))

    submit_ready()
    try:
        while len(done) < len(order):
            while batches:
                batch, remaining = batches[0], []
                for cid in batch:
                    free = next((i for i in range(processors) if i not in occupied), None)
                    fits = all(active[k] + demands[cid].get(k, 0) <= cap for k, cap in capacity.items())
                    if not fits or (kinds[cid] == "compute" and free is None):
                        remaining.append(cid)
                        continue
                    slot = free if kinds[cid] == "compute" else None
                    if slot is not None:
                        occupied.add(slot)
                    for k in active:
                        active[k] += demands[cid].get(k, 0)
                    assignments[cid] = {"rank": rank[cid], "kind": kinds[cid], "processor": slot}
                    running[asyncio.create_task(invoke(cid))] = (cid, slot)
                batches.popleft()
                if remaining:
                    batches.appendleft(remaining)
                    break  # Algorithm 3: re-coordinate remaining calls first.
            if not running:
                raise ValueError("LLMOrch coordinator stalled")
            completed, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(completed, key=lambda t: order.index(running[t][0])):
                cid, slot = running.pop(task)
                task.result()
                done.add(cid)
                if slot is not None:
                    occupied.remove(slot)
                for k in active:
                    active[k] -= demands[cid].get(k, 0)
            submit_ready()
    finally:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
    return assignments
