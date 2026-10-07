import asyncio
import copy
import json
from pathlib import Path
import pytest
from test_evolution import make_tasks, pilot
from test_paper import PaperFake
from test_gaia import FakeBackend, ready
from test_baselines import FakeModelClient, FakePolicyClient
from evoagentx.compactflow.benchmarks import assign_exact_partitions
from evoagentx.compactflow.paper_studies import source_only_pilot
from evoagentx.compactflow.execution_capture import ExecutionCaptureRunner, run_execution_comparison, verify_capture_lock
from evoagentx.compactflow.latency import LatencyReplayRunner, EXECUTION_METHODS


def tasks_for(config, benchmarks=('MBPP','HotpotQA','MATH','GAIA')):
    formal=sum((make_tasks(150,b) for b in benchmarks),[])
    assign_exact_partitions(formal,seed=42,validation_folds=5)
    return source_only_pilot(formal,config,list(benchmarks))


def forbidden(*a,**k):
    pytest.fail('construction, retrieval, or encoder invoked in execution-only study')


def test_four_benchmark_execution_only_no_construction_and_frozen_resume(tmp_path,monkeypatch):
    from evoagentx.compactflow.paper import PaperExperimentRunner
    import evoagentx.compactflow.gaia as gaia
    import evoagentx.compactflow.gaia_tools as backend
    import evoagentx.compactflow.policy as policy
    from examples.compactflow import run_evolution as cli
    import evoagentx.compactflow.baselines as baselines
    monkeypatch.setattr(gaia,'capability_report',ready)
    monkeypatch.setattr(backend,'GaiaBackends',FakeBackend)
    monkeypatch.setattr(cli,'evaluate',lambda *a,**k:{'quality':0.0})
    monkeypatch.setattr(cli.PinnedEmbedder,'__init__',forbidden)
    monkeypatch.setattr(policy.PolicyRetriever,'retrieve',forbidden)
    monkeypatch.setattr(policy.DeterministicTextEmbedder,'embed',forbidden)
    for name in ('_evolution','_baseline','_library'):
        monkeypatch.setattr(PaperExperimentRunner,name,forbidden)
    for cls in (baselines.ConstructionBaselineRunner,baselines.AFlowAdapter,baselines.EvoAgentXAdapter):
        monkeypatch.setattr(cls,'__init__',forbidden)
    c=pilot(); tasks=tasks_for(c)
    raw=tmp_path/'assets';raw.mkdir();(raw/'fixture.txt').write_text('one')
    c['tools']['gaia']['asset_root']=str(raw)
    gaia_tasks=[t for t in tasks if t.benchmark=='GAIA']
    for t in gaia_tasks:t.attachments=['fixture.txt']
    gaia.attach_assets(gaia_tasks,raw)
    out=tmp_path/'out'
    original=cli.compile_spec
    def compile_frozen(spec,*a,**k):
        snapshots=list((out/'capture_run/workflows').glob('*.json'))
        assert any(json.loads(p.read_text())['workflow']==spec for p in snapshots)
        return original(spec,*a,**k)
    monkeypatch.setattr(cli,'compile_spec',compile_frozen)
    def runner():
        return PaperExperimentRunner(tasks,config=c,output=out,data_identity={},formal_manifest_identity='formal',
            dataset_coverage={},studies=['execution_main'],client_factory=PaperFake)
    plan=runner().plan()
    assert plan['timed_records']==48 and plan['warmup_records']==24
    assert plan['methods']==list(EXECUTION_METHODS)
    summary=asyncio.run(runner().run())
    assert summary['selected_run_status']=='complete',summary
    assert len(summary['table5'])==3 and summary['completed_records']==72
    assert summary['capture']['total_tokens']>0
    assert summary['capture']['planner_tokens']>0
    assert not (out/'evolution').exists() and not (out/'construction').exists()
    rows=[json.loads(p.read_text())['record'] for p in (out/'studies/execution_main/records').glob('*.json')]
    for key in {(r['benchmark'],r['task_id'],r['seed']) for r in rows}:
        selected=[r for r in rows if (r['benchmark'],r['task_id'],r['seed'])==key]
        assert len({r['capture_digest'] for r in selected})==1
        assert len({r['workflow_digest'] for r in selected})==1
        assert {r['case']['name'] for r in selected}==set(EXECUTION_METHODS)
        assert all(r['quality']==0 for r in selected)
    before={str(p):p.stat().st_mtime_ns for p in out.rglob('*.json') if 'records' in p.parts or p.parent.name=='captures'}
    import socket
    monkeypatch.setattr(socket,'create_connection',forbidden)
    monkeypatch.setattr(cli.LiveAdapter,'run_variant',forbidden)
    import evoagentx.compactflow.latency as latency
    monkeypatch.setattr(latency,'replay_one',forbidden)
    again=asyncio.run(runner().run(resume=True))
    assert again==summary
    assert LatencyReplayRunner.report(out)==summary
    assert before=={str(p):p.stat().st_mtime_ns for p in out.rglob('*.json') if 'records' in p.parts or p.parent.name=='captures'}
    frozen=next((out/'capture_run/workflows').glob('*.json'))
    frozen.write_text('{}')
    with pytest.raises(ValueError,match='capture changed'):
        LatencyReplayRunner.report(out)


def test_interrupted_capture_reuses_finished_calls_and_rejects_usage_missing(tmp_path,monkeypatch):
    from examples.compactflow import run_evolution as cli
    monkeypatch.setattr(cli,'evaluate',lambda *a,**k:{'quality':0})
    c=pilot(); tasks=tasks_for(c,('MATH',)); out=tmp_path/'out/capture_run'
    def runner():return ExecutionCaptureRunner(tasks,config=c,output=out,data_identity={},formal_manifest_identity='x',client_factory=FakePolicyClient)
    first=runner(); original=first._capture; completed=[]
    async def interrupted(task,seed):
        if completed:raise asyncio.CancelledError()
        value=await original(task,seed);completed.append(value);return value
    first._capture=interrupted
    with pytest.raises(asyncio.CancelledError):asyncio.run(first.run())
    one=next((out/'captures').glob('*.json'));before=one.read_bytes();stamp=one.stat().st_mtime_ns
    result=asyncio.run(runner().run(resume=True))
    assert result['status']=='complete' and result['captures']==2
    assert one.read_bytes()==before and one.stat().st_mtime_ns==stamp
    verify_capture_lock(out,required=True)
    with pytest.raises(ValueError,match='lock changed'):
        changed=runner();changed.config['evaluation']['task_token_budget']+=1
        asyncio.run(changed.run(resume=True))
    # Usage loss must block capture freeze, even when the model has an answer.
    original_run=cli.LiveAdapter.run_variant
    async def missing(self,*a,**kw):
        result=await original_run(self,*a,**kw)
        result.record['tokens']['usage_complete']=False
        return result
    monkeypatch.setattr(cli.LiveAdapter,'run_variant',missing)
    other=ExecutionCaptureRunner(tasks,config=c,output=tmp_path/'bad/capture_run',data_identity={},formal_manifest_identity='x',client_factory=FakePolicyClient)
    result=asyncio.run(run_execution_comparison(other))
    assert result['selected_run_status']=='incomplete'
    assert not (other.output/'capture.lock.json').exists()
    assert not list((other.output.parent/'studies').glob('**/*.json'))


def test_cli_dispatches_before_any_paper_preflight(monkeypatch,tmp_path):
    import sys
    from examples.compactflow import run_paper,run_latency
    seen=[]
    monkeypatch.setattr(run_latency,'main',lambda argv:seen.append(argv) or 0)
    monkeypatch.setattr(run_paper,'load_reproduction_config',forbidden)
    monkeypatch.setattr(run_paper,'PinnedEmbedder',forbidden)
    monkeypatch.setattr(sys,'argv',['run_paper.py','run','--studies','execution_main','--output',str(tmp_path/'out'),'--config','fake.json','--resume'])
    assert run_paper.main()==0
    assert seen and '--resume' in seen[0] and '--config' in seen[0]


def test_closed_gate_prevents_live_target_and_exact_missing_replay_fails(tmp_path,monkeypatch):
    from examples.compactflow import run_evolution as cli
    monkeypatch.setattr(cli,'evaluate',lambda *a,**k:{'quality':1})
    c=pilot();tasks=tasks_for(c,('MATH',));out=tmp_path/'capture'
    runner=ExecutionCaptureRunner(tasks,config=c,output=out,data_identity={},formal_manifest_identity='x',client_factory=FakePolicyClient)
    out.mkdir();runner._prepare(False)
    gate=out/'target_gate.lock.json';saved=gate.read_bytes();gate.unlink()
    with pytest.raises(ValueError,match='gate is closed'):
        asyncio.run(runner._capture(*runner.selected[0]))
    gate.write_bytes(saved)
    asyncio.run(runner.run(resume=True))
    from evoagentx.compactflow.paper_studies import replay_one
    value=json.loads(next((out/'captures').glob('*.json')).read_text())['value']
    value['record']['replay']['records']={}
    from evoagentx.compactflow.latency import public_from_record
    import socket
    monkeypatch.setattr(socket,'create_connection',forbidden)
    for method in EXECUTION_METHODS:
        case={'name':method,'mode':'guarded' if method=='guarded' else 'complete'}
        if method!='guarded':case['backend']=method
        r=asyncio.run(replay_one(value['record'],public_from_record(value['record']),c,case))
        assert r['status']=='incomplete'


def test_typed_wire_constraints_allow_values_but_reject_wrong_fields():
    import re
    from evoagentx.compactflow.llm import ModelClient,ndjson_output_regex
    fields=['analysis','answer'];pattern=ndjson_output_regex(fields)
    valid='\n'.join(json.dumps({'field':f,'value':'quoted "text"\n\\path'},separators=(',',':')) for f in fields)
    assert re.fullmatch(pattern,valid)
    for invalid in (valid+'\n```',valid.replace('analysis','unknown'),valid+'\n'+valid,valid.replace('\\n','\n')):
        assert not re.fullmatch(pattern,invalid)
    c=pilot()['model'];c['typed_output_constraints']=True
    client=ModelClient(c)
    payload=client._payload('system',json.dumps({'output_fields':fields}),'executor',42,stream=True,response_format={'type':'ndjson_fields','fields':fields})
    assert payload['structured_outputs']=={'regex':pattern}
    assert payload['stream_options']['include_usage']
    schema={'type':'object','properties':{'answer':{'type':'string'}}}
    payload=client._payload('system',json.dumps({'workflow_schema':schema}),'planner',42,stream=True,response_format={'type':'json_schema','schema':schema})
    assert payload['structured_outputs']=={'json':schema}
    assert 'structured_outputs' not in client._payload('system','{"observations":[]}','executor',42,stream=True)
    c['typed_output_constraints']=False
    assert 'structured_outputs' not in client._payload('system',json.dumps({'output_fields':fields}),'executor',42,stream=True,response_format={'type':'ndjson_fields','fields':fields})


def test_full_planner_schema_is_supported_without_relaxing_validation():
    from evoagentx.compactflow.llm import decoding_schema
    from evoagentx.compactflow.paper_workflow import PROMPTS,validate_spec
    from test_paper import typed_fixture
    schema=json.loads((PROMPTS.parent/'schemas/workflow.schema.json').read_text())
    wire=decoding_schema(schema)
    assert 'uniqueItems' not in wire['properties']['nodes']['items']['properties']['outputs']
    assert schema['properties']['nodes']['items']['properties']['outputs']['uniqueItems'] is True
    graph=typed_fixture();graph['nodes'][0]['outputs']=['answer','answer']
    with pytest.raises(ValueError):validate_spec(graph,{'question':'q'},set())
    backend=pytest.importorskip('vllm.v1.structured_output.backend_xgrammar')
    has_xgrammar_unsupported_json_features=backend.has_xgrammar_unsupported_json_features
    xgrammar=pytest.importorskip('xgrammar')
    assert not has_xgrammar_unsupported_json_features(wire)
    xgrammar.Grammar.from_json_schema(wire)


def test_binding_guidance_grounds_public_references_and_keeps_semantic_checks():
    from evoagentx.compactflow.paper_workflow import plan_workflow
    from evoagentx.compactflow.benchmarks import BenchmarkTask
    import re
    seen=[]
    class Planner:
        async def json(self,system,prompt,**kwargs):
            data=json.loads(prompt);seen.append(data)
            if len(seen)==1:
                return {'nodes':[{'id':'orphan','tool':'llm','instruction':'x','inputs':{'question':'$input.question'},'outputs':['text']},
                                  {'id':'solve','tool':'llm','instruction':'x','inputs':{'question':'$input.question'},'outputs':['answer']}],
                        'sinks':['solve'],'applied_policies':[],'unapplied_policies':[]}
            return {'nodes':[{'id':'solve','tool':'llm','instruction':'x','inputs':{'question':'$input.question'},'outputs':['answer']}],
                    'sinks':['solve'],'applied_policies':[],'unapplied_policies':[]}
    c=pilot()['construction']['planner'];c['typed_binding_guidance']=True
    spec,trace=asyncio.run(plan_workflow(Planner(),BenchmarkTask('MATH','one','question','1','family'),[],c,seed=42))
    assert trace['repairs']==1 and len(spec['nodes'])==1
    pattern=seen[0]['workflow_schema']['properties']['nodes']['items']['properties']['inputs']['additionalProperties']['pattern']
    assert re.fullmatch(pattern,'$input.question') and re.fullmatch(pattern,'first.answer')
    assert not re.fullmatch(pattern,'$input.undefined') and not re.fullmatch(pattern,'$first.answer')
    assert 'orphan.text' in seen[1]['repair']['valid_references']
    assert 'orphan' in seen[1]['repair']['error']


def test_single_sink_alias_preserves_values_and_never_guesses_multiple_outputs():
    from evoagentx.compactflow.paper_workflow import normalize_single_sink,validate_spec
    spec={'nodes':[{'id':'write_code','tool':'llm','instruction':'Write code','inputs':{'question':'$input.question'},'outputs':['code']}],
          'sinks':['write_code'],'applied_policies':[],'unapplied_policies':[]}
    original=copy.deepcopy(spec)
    canonical,changes=normalize_single_sink(spec)
    assert spec==original
    assert changes==[{'kind':'single_sink_output_alias','node':'write_code','from':'code','to':'answer'}]
    assert canonical['nodes'][0]=={**original['nodes'][0],'outputs':['answer']}
    validate_spec(canonical,{'question':'q'},set())
    spec['nodes'][0]['outputs']=['code','explanation']
    unchanged,changes=normalize_single_sink(spec)
    assert unchanged==spec and changes==[]
    with pytest.raises(ValueError,match='sink must expose answer'):validate_spec(unchanged,{'question':'q'},set())
