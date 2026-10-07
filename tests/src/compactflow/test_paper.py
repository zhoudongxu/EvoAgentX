import asyncio
import copy
import json
from pathlib import Path
import pytest
from test_evolution import make_tasks, pilot
from test_baselines import FakePolicyClient, FakeModelClient
from test_gaia import FakeGaiaClient, FakeBackend, ready
from evoagentx.compactflow.benchmarks import assign_exact_partitions
from evoagentx.compactflow.paper_studies import source_only_pilot, interventions, construction_variants, replay_cases, replay_one, STUDIES
from evoagentx.compactflow.paper_phase import assert_target_allowed, file_digest
from evoagentx.compactflow.baseline_native import ModelSession
from evoagentx.compactflow.paper_workflow import compile_spec
from evoagentx.compactflow.replay import ReplayBundle, graph_to_dict
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.llm import HashEmbedder
from evoagentx.compactflow.execution_baselines import replay_llmcompiler, upstream_identity


def test_pilot_only_formal_source_and_whole_families():
    tasks=make_tasks(150,'GAIA')
    assign_exact_partitions(tasks,seed=42,validation_folds=5)
    selected=source_only_pilot(tasks,pilot(),['GAIA'])
    assert [sum(t.split==s for t in selected) for s in ['source','validation','target']]==[9,3,3]
    source={t.task_id for t in tasks if t.split=='source'}
    assert {t.task_id for t in selected}<=source
    assert all(t.metadata['formal_split']=='source' for t in selected)
    again=source_only_pilot(list(reversed(tasks)),pilot(),['GAIA'])
    assert {(t.task_id,t.split) for t in again}=={(t.task_id,t.split) for t in selected}
    for t in tasks:
        if t.split=='source':t.family_id='one inseparable family'
    with pytest.raises(ValueError,match='without splitting'):
        source_only_pilot(tasks,pilot(),['GAIA'])


def test_global_target_gate_hashes_all_frozen_variants(tmp_path):
    config={'paper_context':{'root':str(tmp_path),'run_identity':'test'}}
    assert_target_allowed(config,'source')
    with pytest.raises(ValueError):assert_target_allowed(config,'target')
    p=tmp_path/'policy.json';p.write_text('{}')
    (tmp_path/'target_gate.lock.json').write_text(json.dumps({'run_identity':'test','artifacts':{'policy.json':file_digest(p)}}))
    assert_target_allowed(config,'target')
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):assert_target_allowed(config,'target')


def test_model_session_stream_arrives_before_completion_and_resumes(tmp_path):
    class Stream(FakeModelClient):
        calls=0
        async def stream(self,*args,**kwargs):
            type(self).calls+=1
            yield 'early'
            await asyncio.sleep(.03)
            await super().text(*args,**kwargs)
            yield 'late'
    async def run():
        session=ModelSession(pilot(),42,tmp_path,client_factory=Stream)
        stream=session.stream([{'role':'user','content':'test'}],'executor','call')
        assert await anext(stream)=='early'
        assert session.records==[]
        assert [c async for c in stream]==['late']
        assert session.accounting()['total_tokens']>0
        replay=ModelSession(pilot(),42,tmp_path,client_factory=Stream)
        assert [c async for c in replay.stream([{'role':'user','content':'test'}],'executor','call')]==['early','late']
        assert replay.accounting()['physical_tokens']==0
        assert Stream.calls==1
    asyncio.run(run())


def test_cancelled_stream_is_not_reinferred(tmp_path):
    class Stream(FakeModelClient):
        async def stream(self,*args,**kwargs):
            yield 'early'
            raise asyncio.CancelledError()
    async def run():
        session=ModelSession(pilot(),42,tmp_path,client_factory=Stream)
        with pytest.raises(asyncio.CancelledError):
            async for _ in session.stream([{'role':'user','content':'x'}],'executor','x'):pass
        assert not session.accounting()['usage_complete']
        from evoagentx.compactflow.baselines import BaselineUnavailable
        with pytest.raises(BaselineUnavailable,match='incomplete stream'):
            async for _ in ModelSession(pilot(),42,tmp_path,client_factory=Stream).stream([{'role':'user','content':'x'}],'executor','x'):pass
    asyncio.run(run())


def typed_fixture():
    return {'nodes':[
        {'id':'first','tool':'llm','instruction':'Reason','inputs':{'problem':'$input.question'},'outputs':['answer']},
        {'id':'second','tool':'llm','instruction':'Check','inputs':{'previous':'first.answer'},'outputs':['answer']},
        {'id':'final','tool':'llm','instruction':'Conclude','inputs':{'a':'first.answer','b':'second.answer'},'outputs':['answer']}],
        'sinks':['final'],'applied_policies':[],'unapplied_policies':[]}


def test_all_four_typed_interventions_and_parameter_sweeps():
    proposals,reasons=interventions(typed_fixture(),{'benchmark':'MATH','question':'test'},12)
    # This graph has fan-out; use a chain for the fusion precondition.
    chain=typed_fixture();chain['nodes'][-1]['inputs']={'b':'second.answer'}
    others,_=interventions(chain,{'benchmark':'MATH','question':'test'},12)
    assert {p['kind'] for p in proposals+others}=={'node_bypass','edge_removal','node_fusion','early_stopping'}
    variants=construction_variants(pilot(),STUDIES)
    assert variants['without_structural_match']['config']['construction']['retrieval_weights']['structural']==0
    for v in variants.values():
        if v['kind']=='sensitivity':
            field={'quality_tolerance':'quality_tolerance','retrieval_k0':'semantic_top_k0','selected_k':'max_policies'}[v['axis']]
            assert v['config']['construction'][field]==v['value']
    assert len([c for c in replay_cases(pilot(),STUDIES) if c['study']=='execution_main'])==6


def test_native_llmcompiler_uses_upstream_and_exact_offline_arguments(monkeypatch):
    from evoagentx.compactflow._vendor.llmcompiler.task_fetching_unit import TaskFetchingUnit
    called=[];schedule=TaskFetchingUnit.schedule
    async def observed(self):called.append(True);return await schedule(self)
    monkeypatch.setattr(TaskFetchingUnit,'schedule',observed)
    async def run():
        from evoagentx.compactflow.baseline_controls import SessionClient
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            client=SessionClient(ModelSession(pilot(),42,Path(folder),client_factory=FakePolicyClient))
            bundle=ReplayBundle('native-test');public={'benchmark':'MATH','question':'test'}
            graph=compile_spec(typed_fixture(),public,client,seed=42,capacity=4,recorder=bundle)
            captured=await CompactFlowRuntime(graph,mode='complete',sink_ids=('final',)).execute(public)
            record={'graph':graph_to_dict(graph,['final']),'replay':{'workflow_id':bundle.workflow_id,'records':bundle.records},'arguments':captured.arguments,'answer':'1','evaluation':{'quality':1}}
            import socket
            monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('replay attempted network'))
            result=await replay_one(record,public,pilot(),{'name':'llmcompiler','backend':'llmcompiler','mode':'complete'})
            assert result['status']=='complete',result
            assert result['outputs_equal_capture']
            assert result['safety']['all_call_failure_incidents']==0
            assert called and len(upstream_identity()['revision'])==40
            record['replay']['records']={}
            failed=await replay_one(record,public,pilot(),{'name':'llmcompiler','backend':'llmcompiler','mode':'complete'})
            assert failed['status']=='incomplete'
    asyncio.run(run())


class PaperFake(FakeGaiaClient):
    async def text(self,system,prompt,*,component,seed):
        try:data=json.loads(prompt)
        except ValueError:data={}
        if component=='executor' and 'observations' not in data:
            if not data:return await FakeModelClient.text(self,system,prompt,component=component,seed=seed)
            client=FakePolicyClient()
            value=await client.text(system,prompt,component=component,seed=seed)
            self.records.extend(client.records)
            return value
        if component=='distiller':
            await FakeModelClient.text(self,system,prompt,component=component,seed=seed)
            return json.dumps({'candidate':None})
        if component=='planner' and data.get('task') and data['task'].get('benchmark')!='GAIA':
            client=FakePolicyClient()
            value=await client.text(system,prompt,component=component,seed=seed)
            self.records.extend(client.records)
            return value
        return await super().text(system,prompt,component=component,seed=seed)
    async def stream(self,*args,**kwargs):yield await self.text(*args,**kwargs)


def test_four_benchmark_full_paper_orchestration_and_resume(tmp_path,monkeypatch):
    from evoagentx.compactflow.paper import PaperExperimentRunner
    from evoagentx.compactflow.baselines import AFlowAdapter,EvoAgentXAdapter
    from evoagentx.compactflow.baseline_controls import PolicyBaselineAdapter
    from evoagentx.compactflow.gaia import attach_assets
    import evoagentx.compactflow.gaia as gaia
    import evoagentx.compactflow.gaia_tools as backends
    from examples.compactflow import run_evolution as cli
    monkeypatch.setattr(gaia,'capability_report',ready)
    monkeypatch.setattr(backends,'GaiaBackends',FakeBackend)
    monkeypatch.setattr(cli,'evaluate',lambda *a,**k:{'quality':1.0})
    import evoagentx.compactflow.baselines as bm
    monkeypatch.setattr(bm,'evaluate',lambda *a,**k:{'quality':1.0})
    c=pilot();c['evaluation']['generation_seeds']=[42]
    raw=tmp_path/'raw';raw.mkdir();(raw/'fixture.txt').write_text('one')
    c['tools']['gaia']['asset_root']=str(raw)
    tasks=[]
    for benchmark in ('MBPP','HotpotQA','MATH','GAIA'):
        formal=make_tasks(150,benchmark)
        assign_exact_partitions(formal,seed=42,validation_folds=5)
        cohort=source_only_pilot(formal,c,[benchmark])
        if benchmark=='GAIA':
            for t in cohort:t.attachments=['fixture.txt']
            attach_assets(cohort,raw)
        tasks+=cohort
    def adapters(methods,evo):
        return [AFlowAdapter(client_factory=PaperFake),EvoAgentXAdapter(client_factory=PaperFake)]+[
            PolicyBaselineAdapter(m,evolution_dir=evo,client_factory=PaperFake,embedder=HashEmbedder()) for m in methods if m not in {'aflow','evoagentx'}]
    def runner():return PaperExperimentRunner(tasks,config=c,output=tmp_path/'out',data_identity={},formal_manifest_identity='formal-test',dataset_coverage={},client_factory=PaperFake,embedder=HashEmbedder(),adapters_factory=adapters)
    summary=asyncio.run(runner().run())
    assert summary['selected_run_status']=='complete',summary['checks']
    assert summary['main_records']==72
    assert summary['publication_status']=='incomplete'
    count=len(FakeModelClient.requests)
    again=asyncio.run(runner().run(resume=True))
    assert len(FakeModelClient.requests)==count and again==summary
    assert list((tmp_path/'out/figures').glob('*.pdf'))


def test_http_cancellation_collects_late_actual_usage(monkeypatch,tmp_path):
    import threading,time,requests
    from evoagentx.compactflow.llm import ModelClient
    first=threading.Event()
    class Response:
        status_code=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def raise_for_status(self):pass
        def iter_lines(self):
            first.set()
            yield b'data: {"choices":[{"delta":{"content":"early"}}]}'
            time.sleep(.04)
            yield b'data: {"usage":{"prompt_tokens":7,"completion_tokens":3},"choices":[]}'
            yield b'data: [DONE]'
    monkeypatch.setattr(requests,'post',lambda *a,**k:Response())
    async def run():
        session=ModelSession(pilot(),42,tmp_path,client_factory=ModelClient)
        job=asyncio.create_task(session.text([{'role':'user','content':'test'}],'executor','cancel'))
        while not first.is_set():await asyncio.sleep(.001)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):await job
        assert session.accounting()['usage_complete']
        assert session.accounting()['total_tokens']==10
        saved=json.loads(next(tmp_path.glob('*.json')).read_text())
        assert saved['terminal_status']=='cancelled' and saved['error']
        assert saved['records'][0]['late_usage_after_cancellation']
    asyncio.run(run())


def test_media_content_failure_is_audited_tool_error(tmp_path,monkeypatch):
    from test_gaia import gaia_config,gaia_tasks
    import evoagentx.compactflow.gaia_tools as backend_module
    from evoagentx.compactflow.gaia import GaiaToolSession
    import yt_dlp
    config=gaia_config(tmp_path);task=gaia_tasks(tmp_path)[0]
    monkeypatch.setattr(backend_module,'public_url',lambda url:url)
    class Downloader:
        def __init__(self,options):
            assert '+bestaudio' in options['format']
            assert options['match_filter']({'duration':3601})
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def extract_info(self,url,download):
            assert download is True
            raise yt_dlp.utils.DownloadError('Requested format is not available')
    monkeypatch.setattr(yt_dlp,'YoutubeDL',Downloader)
    async def run():
        session=GaiaToolSession(config,task.public_input(),output=tmp_path/'output',workspace=tmp_path/'calls',seed=42)
        result=await session.invoke('download',{'url':'https://example.org/video','media':True})
        assert 'resource unavailable' in result['error']
        assert not session.incomplete
        assert session.accounting()['usage_complete']
        rows=[json.loads(l) for l in (tmp_path/'output/tool_records.jsonl').read_text().splitlines()]
        assert rows[-1]['status']=='tool_error'
    asyncio.run(run())


def test_evolution_freeze_does_not_run_target(tmp_path):
    from test_evolution import runner
    value=runner(tmp_path)
    asyncio.run(value.run(stage='freeze'))
    assert not value._target_records
    assert json.loads((tmp_path/'checkpoint.json').read_text())['stage']=='frozen'
    asyncio.run(runner(tmp_path).run(stage='target',resume=True))
    assert json.loads((tmp_path/'checkpoint.json').read_text())['stage']=='complete'
