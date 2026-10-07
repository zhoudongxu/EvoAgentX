"""Pinned native schedulers for the fixed-workflow, offline execution study.

LLMCompiler uses its upstream TaskFetchingUnit, not a renamed CompactFlow mode.
The bridge resolves typed arguments, bounds resources and records common audit
traces; these boundary operations are disclosed in the upstream provenance.
"""
from __future__ import annotations
import asyncio
import copy
import hashlib
import json
import time
from pathlib import Path
from .schema import (CallState, CallTrace, Complete, Failure, Partial, TraceEvent,
                     TraceKind, ExecutionMetrics, ExecutionResult)
from .runtime import _MISSING, _path_get, _path_set, _deep_merge, _validate_json_instance


def upstream_identity():
    folder=Path(__file__).parent/'_vendor/llmcompiler'
    identity=json.loads((folder/'UPSTREAM.json').read_text())
    if hashlib.sha256((folder/'task_fetching_unit.py').read_bytes()).hexdigest()!=identity['sha256_vendored']:
        raise ValueError('pinned LLMCompiler scheduler checksum mismatch')
    return identity


def execution_coverage():
    return {'llmcompiler':{'status':'available','identity':upstream_identity(),
                          'scope':'native fixed-workflow scheduler with shared typed call/resource boundary'},
            'llmorch':llmorch_identity()}


def llmorch_identity():
    from .llmorch import IMPLEMENTATION
    return {'status':'available', 'implementation':IMPLEMENTATION,
            'paper':'https://arxiv.org/html/2504.14872v2',
            'author_download':'https://www.hostize.com/v/c3oLTBMUwn',
            'author_download_observation':'HTTP 404 on 2026-10-06',
            'source_sha256':hashlib.sha256((Path(__file__).parent/'llmorch.py').read_bytes()).hexdigest(),
            'scope':'Sections III-B/C core scheduler; fixed workflow, full-result dependencies; logical processor slots in timed offline replay',
            'excluded':'query translation, MPI transport, automatic repair/retry/recovery; no claim of CPU scaling reproduction',
            'adaptation':'all predecessor completions required per III-C prose; shared effect-order and capacity boundary; remote service calls classified inout'}


async def replay_llmcompiler(graph, public, *, sinks, call_timeout, workflow_timeout):
    return await _replay_external(graph, public, sinks=sinks, call_timeout=call_timeout,
                                  workflow_timeout=workflow_timeout, backend='llmcompiler')


async def replay_llmorch(graph, public, *, sinks, call_timeout, workflow_timeout,
                         processors=4, kinds=None):
    return await _replay_external(graph, public, sinks=sinks, call_timeout=call_timeout,
                                  workflow_timeout=workflow_timeout, backend='llmorch',
                                  processors=processors, kinds=kinds)


async def _replay_external(graph, public, *, sinks, call_timeout, workflow_timeout,
                            backend, processors=4, kinds=None):
    ids={cid:i+1 for i,cid in enumerate(graph.topological_order)}
    capacity=dict(graph.resource_capacity.amounts)
    active={k:0 for k in capacity};condition=asyncio.Condition()
    trace=[];outputs={};traces={c.id:CallTrace(c.id) for c in graph.calls}
    states={c.id:CallState.WAITING for c in graph.calls};workers=set()
    metrics=ExecutionMetrics(started_at=time.perf_counter())
    def event(kind,cid,detail=None):
        trace.append(TraceEvent(len(trace),time.perf_counter(),kind,cid,detail or {}))
    async def invoke(call):
        workers.add(asyncio.current_task())
        acquired=False;ct=traces[call.id]
        try:
            dependencies=graph.incoming_data(call.id)
            predecessors={d.producer for d in dependencies if d.producer is not None}
            predecessors.update(d.producer for d in graph.incoming_effects(call.id))
            if any(states[p]!=CallState.COMPLETED for p in predecessors):
                raise ValueError('failed predecessor')
            arguments={}
            for d in dependencies:
                value=_path_get(public if d.producer is None else outputs[d.producer],d.source_path,_MISSING)
                if value is _MISSING:
                    if d.required:raise ValueError('missing required replay input: '+d.source_path)
                else:_path_set(arguments,d.target_path,value)
            _validate_json_instance(call.input_schema,arguments,label='native baseline inputs')
            ct.ready_at=time.perf_counter();event(TraceKind.READY,call.id)
            async with condition:
                await condition.wait_for(lambda:all(active[k]+call.resources.get(k)<=capacity[k] for k in capacity))
                for k in capacity:
                    active[k]+=call.resources.get(k)
                    metrics.peak_resources[k]=max(metrics.peak_resources.get(k,0),active[k])
                acquired=True
            ct.arguments=arguments;ct.started_at=time.perf_counter();ct.state=states[call.id]=CallState.RUNNING
            event(TraceKind.START,call.id)
            async def consume():
                value={};terminal=False
                async for item in call.target(**arguments):
                    if isinstance(item,Failure):raise RuntimeError(str(item.error))
                    value=_deep_merge(value,item.data)
                    stamp=time.perf_counter()
                    if ct.first_output_at is None:ct.first_output_at=stamp
                    if metrics.first_internal_output_at is None:metrics.first_internal_output_at=stamp
                    if call.id in sinks and metrics.first_output_at is None:metrics.first_output_at=stamp
                    event(TraceKind.PARTIAL if isinstance(item,Partial) else TraceKind.COMPLETE,
                          call.id,{'effects':list(item.effects)})
                    if isinstance(item,Complete):terminal=True;break
                if not terminal:raise ValueError('incomplete replay stream')
                _validate_json_instance(call.output_schema,value,label='native baseline outputs')
                return value
            outputs[call.id]=await asyncio.wait_for(consume(),call_timeout)
            ct.output=outputs[call.id];ct.state=states[call.id]=CallState.COMPLETED
            # Same first-valid-answer latency boundary as CompactFlow.
            if metrics.output_ready_at is None and all(states[s]==CallState.COMPLETED for s in sinks):
                metrics.output_ready_at=time.perf_counter()
            return outputs[call.id]
        except asyncio.CancelledError:
            ct.error='cancelled native baseline replay';ct.state=states[call.id]=CallState.FAILED
            event(TraceKind.CANCEL,call.id)
            raise
        except Exception as exc:
            ct.error=type(exc).__name__+': '+str(exc);ct.state=states[call.id]=CallState.FAILED
            event(TraceKind.FAILURE,call.id,{'error':ct.error})
            return None  # Upstream still terminates this task; descendants fail closed.
        finally:
            ct.ended_at=time.perf_counter()
            if acquired:
                async with condition:
                    for k in capacity:active[k]-=call.resources.get(k)
                    condition.notify_all()
    if backend == 'llmcompiler':
        from ._vendor.llmcompiler.task_fetching_unit import Task, TaskFetchingUnit
        upstream_identity()
        unit=TaskFetchingUnit()
        tasks={}
        for cid in graph.topological_order:
            call=graph.call_map[cid]
            deps={d.producer for d in graph.incoming_data(cid) if d.producer is not None}
            # Explicit effect ordering is supplied as a conservative completion edge.
            deps.update(d.producer for d in graph.incoming_effects(cid))
            async def tool(c=call):return await invoke(c)
            tasks[ids[cid]]=Task(ids[cid],cid,tool,[],[ids[d] for d in sorted(deps)])
        unit.set_tasks(tasks)
    async def schedule():
        if backend == 'llmcompiler':
            return await unit.schedule()
        from .llmorch import coordinate
        predecessors={cid:{d.producer for d in graph.incoming_data(cid) if d.producer is not None}
                          | {d.producer for d in graph.incoming_effects(cid)} for cid in ids}
        async def call(cid):return await invoke(graph.call_map[cid])
        return await coordinate(graph.topological_order, predecessors,
            {cid:dict(graph.call_map[cid].resources.amounts) for cid in ids}, capacity,
            kinds, processors, call)
    try:await asyncio.wait_for(schedule(),workflow_timeout)
    finally:
        pending=[w for w in workers if not w.done()]
        for w in pending:w.cancel()
        if pending:await asyncio.gather(*pending,return_exceptions=True)
        metrics.ended_at=time.perf_counter()
    return ExecutionResult(outputs,states,tuple(trace),traces,metrics)
