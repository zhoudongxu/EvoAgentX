import asyncio
import copy
import json
from pathlib import Path
import pytest
from evoagentx.compactflow.llmorch import coordinate
from evoagentx.compactflow.latency import table5, public_from_record, LatencyReplayRunner
from evoagentx.compactflow.paper_studies import replay_one, replay_cases
from evoagentx.compactflow.paper_workflow import compile_spec
from evoagentx.compactflow.replay import ReplayBundle, graph_to_dict, digest
from evoagentx.compactflow.runtime import CompactFlowRuntime
from test_paper import typed_fixture
from test_baselines import FakePolicyClient
from test_evolution import pilot


def test_core_io_priority_resources_and_cross_rank_join():
    async def run():
        events=[];active=0;peak=0;done=set()
        order=('slow','quick','io','child','join')
        deps={'slow':set(),'quick':set(),'io':set(),'child':{'quick'},'join':{'slow','child'}}
        kinds={c:'compute' for c in order};kinds['io']='inout'
        async def invoke(cid):
            nonlocal active,peak
            assert deps[cid] <= done
            events.append(('start',cid));active+=1;peak=max(peak,active)
            await asyncio.sleep(.025 if cid=='slow' else .001)
            active-=1;done.add(cid);events.append(('end',cid))
        assignments=await coordinate(order,deps,{c:{'external_call':1} for c in order},
                                     {'external_call':2},kinds,1,invoke)
        assert events[0]==('start','io')
        assert peak==2 and len(done)==5
        assert assignments['io']['processor'] is None
        assert {assignments[c]['processor'] for c in order if c!='io'}=={0}
        assert assignments['join']['rank']==3
    asyncio.run(run())


def test_no_rank_barrier_and_distinct_compute_processors():
    async def run():
        end=set();overlapped=[]
        async def invoke(cid):
            if cid=='child':overlapped.append('slow' not in end)
            await asyncio.sleep(.03 if cid=='slow' else .001);end.add(cid)
        a=await coordinate(('slow','fast','child'),{'slow':set(),'fast':set(),'child':{'fast'}},
            {c:{'r':1} for c in ('slow','fast','child')},{'r':3},
            {c:'compute' for c in ('slow','fast','child')},2,invoke)
        assert overlapped==[True] and a['slow']['processor']!=a['fast']['processor']
    asyncio.run(run())


def test_cancel_joins_tasks_and_impossible_demand_fails():
    async def run():
        cancelled=[]
        async def invoke(cid):
            try:await asyncio.sleep(10)
            finally:cancelled.append(cid)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(coordinate(('x',),{'x':set()},{'x':{'r':1}},{'r':1},{'x':'compute'},1,invoke),.01)
        assert cancelled==['x']
        with pytest.raises(ValueError,match='demand'):
            await coordinate(('x',),{'x':set()},{'x':{'r':2}},{'r':1},{'x':'compute'},1,invoke)
    asyncio.run(run())


async def capture():
    public={'benchmark':'MATH','question':'test'};bundle=ReplayBundle('llmorch-test')
    import tempfile
    from evoagentx.compactflow.baseline_native import ModelSession
    from evoagentx.compactflow.baseline_controls import SessionClient
    with tempfile.TemporaryDirectory() as folder:
        client=SessionClient(ModelSession(pilot(),42,Path(folder),client_factory=FakePolicyClient))
        graph=compile_spec(typed_fixture(),public,client,seed=42,capacity=4,recorder=bundle)
        result=await CompactFlowRuntime(graph,mode='complete',sink_ids=('final',)).execute(public)
    return public, {'graph':graph_to_dict(graph,['final']),'replay':{'workflow_id':bundle.workflow_id,'records':bundle.records},
                    'arguments':result.arguments,'answer':'1','evaluation':{'quality':1},
                    'tokens':{'usage_complete':True,'total_tokens':30}}


def test_exact_replay_no_network_and_no_early_dispatch(monkeypatch):
    async def run():
        public,record=await capture()
        import socket
        monkeypatch.setattr(socket,'create_connection',lambda *a,**k:pytest.fail('network during replay'))
        assert public_from_record(record)=={'question':'test'}
        cfg=pilot();cfg['paper_experiments']['llmorch_reimplementation']=True
        assert 'llmorch' in {c['name'] for c in replay_cases(cfg,['execution_main'])}
        result=await replay_one(record,public,cfg,{'name':'llmorch','backend':'llmorch','mode':'complete'})
        assert result['status']=='complete',result
        assert result['outputs_equal_capture'] and result['safety']['all_call_failure_incidents']==0
        assert result['safety']['denominator']==0
        assert result['scheduler_provenance']['implementation']=='llmorch_paper_reimplementation_v1'
        record['replay']['records']={}
        result=await replay_one(record,public,cfg,{'name':'llmorch','backend':'llmorch','mode':'complete'})
        assert result['status']=='incomplete'
    asyncio.run(run())


def test_paired_table_does_not_drop_missing_or_failed_rows():
    from evoagentx.compactflow.latency import METHODS
    rows=[{'benchmark':'MATH','task_id':'one','seed':42,'repetition':0,'warmup':False,
           'case':{'name':m},'latency':2 if m=='llmorch' else 1,'ttfo':.5,'quality':1,'status':'complete',
           'safety':{'assessment_complete':True,'denominator':0,'counts':{'aggregate':0}}} for m in METHODS]
    expected={('MATH','one',42,0)}
    t=table5(rows,['MATH'],expected)
    assert all(r['status']=='complete' for r in t)
    assert t[2]['speedup_vs_llmorch']==1 and t[-1]['speedup_vs_llmorch']==2
    assert all(r['contract_violation_fraction'] is None for r in t)
    broken=table5(rows[:2]+rows[3:],['MATH'],expected)
    assert all(r['speedup_vs_llmorch'] is None and r['status']=='incomplete' for r in broken)


def test_frozen_runner_resume_report_and_tamper(tmp_path,monkeypatch):
    async def run():
        public,record=await capture()
        source=tmp_path/'source';source.mkdir();(source/'captures').mkdir()
        cfg=pilot();identity=['MATH','one','target',42,'base',None,False]
        value={'benchmark':'MATH','task_id':'one','family_id':'family','seed':42,'split':'target','variant':'base','status':'complete','record':record}
        (source/'captures'/(digest(identity)+'.json')).write_text(json.dumps({'identity':identity,'value':value,'digest':digest(value)}))
        (source/'config.lock.json').write_text(json.dumps(cfg))
        (source/'experiment_manifest.json').write_text(json.dumps({'benchmarks':['MATH'],'execution':{'tasks':1,'seeds':[42],'warmups':1,'repetitions':2}}))
        (source/'sample_manifest.jsonl').write_text(json.dumps({'benchmark':'MATH','task_id':'one','split':'target'})+'\n')
        (source/'frozen.json').write_text('{}')
        from evoagentx.compactflow.paper_phase import file_digest
        (source/'target_gate.lock.json').write_text(json.dumps({'run_identity':'test','artifacts':{'frozen.json':file_digest(source/'frozen.json')}}))
        runner=LatencyReplayRunner(source,tmp_path/'out')
        result=await runner.run()
        assert result['selected_run_status']=='complete',result
        assert result['publication_status']=='incomplete'
        paths=list((tmp_path/'out/studies/execution_main/records').glob('*.json'))
        assert len(paths)==15
        before={str(p):p.read_bytes() for p in paths}
        import evoagentx.compactflow.latency as module
        async def forbidden(*a,**k):pytest.fail('completed call repeated')
        monkeypatch.setattr(module,'replay_one',forbidden)
        await LatencyReplayRunner(source,tmp_path/'out').run(resume=True)
        LatencyReplayRunner.report(tmp_path/'out')
        assert before=={str(p):p.read_bytes() for p in paths}
        (source/'frozen.json').write_text('{"mutated":true}')
        with pytest.raises(ValueError,match='artifact changed'):
            LatencyReplayRunner(source,tmp_path/'other').prepare()
    asyncio.run(run())
