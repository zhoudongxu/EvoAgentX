"""A later cleanup must not overwrite the first valid answer timestamp."""
import asyncio
import pytest
from evoagentx.compactflow.compiler import GFRGCompiler
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.schema import CallSpec, Complete, DataDependency, Failure, Partial, StreamContract
from evoagentx.compactflow.execution_baselines import replay_llmcompiler, replay_llmorch


def fields(*names):
    return {'type':'object','properties':{n:{'type':'string'} for n in names},
            'required':list(names),'additionalProperties':False}


@pytest.mark.parametrize('method', ['guarded', 'llmcompiler', 'llmorch'])
def test_late_other_call_does_not_move_answer_timestamp(method):
    async def run():
        sink_emitted = asyncio.Event()
        async def sink():
            sink_emitted.set()
            yield Complete({'answer':'ok'})
        async def cleanup():
            await asyncio.wait_for(sink_emitted.wait(), 1)
            await asyncio.sleep(.03)
            yield Complete({'done':True})
        graph = GFRGCompiler({'external_call':2}).compile([
            CallSpec('sink',sink,resources={'external_call':1}),
            CallSpec('cleanup',cleanup,resources={'external_call':1})], [])
        if method == 'guarded':
            result = await CompactFlowRuntime(graph, mode='guarded', sink_ids=('sink',)).execute({})
        else:
            runner = replay_llmcompiler if method == 'llmcompiler' else replay_llmorch
            kwargs = {} if method == 'llmcompiler' else {'processors':2, 'kinds':{'sink':'inout','cleanup':'inout'}}
            result = await runner(graph, {}, sinks=('sink',), call_timeout=1, workflow_timeout=2, **kwargs)
        assert not result.errors
        assert result.metrics.output_ready_at <= result.call_traces['sink'].ended_at
        assert result.metrics.output_ready_at < result.call_traces['cleanup'].ended_at
        assert result.metrics.latency < result.metrics.ended_at-result.metrics.started_at
    asyncio.run(run())


@pytest.mark.parametrize('late_failure', [False, True])
def test_early_answer_boundary_and_late_contract_failure(late_failure):
    async def run():
        sink_emitted = asyncio.Event()
        async def producer():
            yield Partial({'ready':'x'}, ('ready',))
            await asyncio.wait_for(sink_emitted.wait(), 1)
            await asyncio.sleep(.03)
            if late_failure:
                yield Failure('producer failed after early answer')
            else:
                yield Complete({'ready':'x','tail':'done'})
        async def sink(x):
            sink_emitted.set()
            yield Complete({'answer':x})
        graph = GFRGCompiler({'external_call':2}).compile([
            CallSpec('producer',producer,output_schema=fields('ready','tail'),resources={'external_call':1}, stream_contract=StreamContract(stable_fields=('ready',))),
            CallSpec('sink',sink,input_schema=fields('x'),output_schema=fields('answer'),early_safe=True,resources={'external_call':1})],
            [DataDependency('producer','sink','ready','x')])
        result = await CompactFlowRuntime(graph, mode='guarded', sink_ids=('sink',)).execute({})
        if late_failure:
            assert result.errors and result.metrics.output_ready_at is None
            assert not result.outputs['sink']
            assert result.metrics.latency == result.metrics.ended_at-result.metrics.started_at
        else:
            assert not result.errors
            assert result.metrics.output_ready_at == result.call_traces['sink'].ended_at
            assert result.metrics.output_ready_at < result.call_traces['producer'].ended_at
    asyncio.run(run())
