import asyncio
import json
import time
from pathlib import Path
from contextlib import aclosing
import pytest
from evoagentx.compactflow.baseline_native import ModelSession
from evoagentx.compactflow.baseline_controls import SessionClient
from evoagentx.compactflow.llm import ModelClient, TokenBudgetExceeded
from evoagentx.compactflow.paper_workflow import compile_spec
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.replay import ReplayBundle
from test_evolution import pilot


def http_response(monkeypatch, text, *, late=False, usage=True):
    import requests
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        status_code=200
        def raise_for_status(self):pass
        def iter_lines(self):
            yield b'data: '+json.dumps({'choices':[{'delta':{'content':text}}]}).encode()
            if late:time.sleep(.04)
            if usage:yield b'data: '+json.dumps({'usage':{'prompt_tokens':11,'completion_tokens':7},'choices':[]}).encode()
            yield b'data: [DONE]'
    monkeypatch.setattr(requests,'post',lambda *a,**k:Response())


@pytest.mark.parametrize('text',[
    '{"field":"answer","value":"x"}\n{"field":"answer","value":"y"}\n',
    'not JSON\n',
    '{"field":"wrong","value":"x"}\n'])
def test_parse_failure_drains_actual_usage_and_does_not_reinfer(tmp_path,monkeypatch,text):
    http_response(monkeypatch,text,late=True)
    config=pilot();config['model']['typed_output_constraints']=False
    spec={'nodes':[{'id':'solve','tool':'llm','instruction':'solve',
                   'inputs':{'question':'$input.question'},'outputs':['answer']}],
          'sinks':['solve'],'applied_policies':[],'unapplied_policies':[]}
    async def run():
        session=ModelSession(config,42,tmp_path)
        bundle=ReplayBundle('failure')
        graph=compile_spec(spec,{'question':'q'},SessionClient(session),seed=42,recorder=bundle)
        result=await CompactFlowRuntime(graph,mode='complete',sink_ids=['solve']).execute({'question':'q'})
        assert result.errors
        assert session.accounting()['usage_complete']
        assert session.accounting()['total_tokens']==18
        saved=json.loads(next(tmp_path.glob('*.json')).read_text())
        assert saved['usage_complete'] and saved['charged']==18
        # Reproduce the same failed logical execution from its durable stream.
        import requests
        monkeypatch.setattr(requests,'post',lambda *a,**k:pytest.fail('unexpected inference'))
        session2=ModelSession(config,42,tmp_path)
        graph=compile_spec(spec,{'question':'q'},SessionClient(session2),seed=42)
        result2=await CompactFlowRuntime(graph,mode='complete',sink_ids=['solve']).execute({'question':'q'})
        assert result2.errors and session2.accounting()['usage_complete']
        assert session2.accounting()['physical_tokens']==0
    asyncio.run(run())


def test_presend_context_rejection_has_known_zero_cost(tmp_path,monkeypatch):
    import requests
    monkeypatch.setattr(requests,'post',lambda *a,**k:pytest.fail('request must not be sent'))
    config=pilot();config['model']['context_length']=1
    async def run():
        session=ModelSession(config,42,tmp_path)
        with pytest.raises(ValueError,match='context bound'):
            await session.text([{'role':'user','content':'hello'}],'executor','one')
        assert session.accounting()['usage_complete']
        assert session.accounting()['total_tokens']==0
        saved=json.loads(next(tmp_path.glob('*.json')).read_text())
        assert saved['charged']==0 and saved['records'][0]['request_status']=='not_sent'
    asyncio.run(run())


def test_explicit_close_and_timeout_keep_late_usage(tmp_path,monkeypatch):
    http_response(monkeypatch,'x',late=True)
    async def run():
        session=ModelSession(pilot(),42,tmp_path)
        async with aclosing(session.stream([{'role':'user','content':'q'}],'executor','one')) as stream:
            assert await anext(stream)=='x'
        assert session.accounting()['usage_complete'] and session.accounting()['total_tokens']==18
    asyncio.run(run())


def test_real_missing_usage_stays_unknown(tmp_path,monkeypatch):
    http_response(monkeypatch,'x',usage=False)
    async def run():
        session=ModelSession(pilot(),42,tmp_path)
        with pytest.raises(ValueError,match='omitted token usage'):
            async for _ in session.stream([{'role':'user','content':'q'}],'executor','one'):pass
        assert not session.accounting()['usage_complete']
        from evoagentx.compactflow.baselines import BaselineUnavailable
        with pytest.raises(BaselineUnavailable):
            async for _ in ModelSession(pilot(),42,tmp_path).stream([{'role':'user','content':'q'}],'executor','one'):pass
    asyncio.run(run())


def test_formats_are_explicit_not_guessed_from_fields():
    config=pilot()['model'];config['typed_output_constraints']=True
    client=ModelClient(config)
    body=json.dumps({'observations':[],'output_fields':['answer']})
    natural=client._payload('s',body,'executor',42,stream=True)
    assert 'structured_outputs' not in natural and 'response_format' not in natural
    agent=client._payload('s',body,'executor',42,stream=True,response_format={'type':'json_object'})
    assert agent['response_format']=={'type':'json_object'} and 'structured_outputs' not in agent
    typed=client._payload('s',body,'executor',42,stream=True,response_format={'type':'ndjson_fields','fields':['answer']})
    assert 'regex' in typed['structured_outputs']


def test_discovery_health_isolates_unknown_benchmark(tmp_path):
    from evoagentx.compactflow.paper import PaperExperimentRunner
    from evoagentx.compactflow.paper_phase import assert_target_allowed,file_digest
    runner=object.__new__(PaperExperimentRunner);runner.output=tmp_path;runner.benchmarks=['MATH','MBPP']
    calls=tmp_path/'evolution/base/calls';calls.mkdir(parents=True)
    for b,complete in [('MATH',False),('MBPP',True)]:
        (calls/(b+'.json')).write_text(json.dumps({'benchmark':b,'record':{'execution_status':'failed','tokens':{'usage_complete':complete},'infrastructure_status':'complete'}}))
    health=runner._discovery_health()
    assert health['ready_benchmarks']==['MBPP']
    p=tmp_path/'discovery_health.json'
    (tmp_path/'target_gate.lock.json').write_text(json.dumps({'run_identity':'one','artifacts':{'discovery_health.json':file_digest(p)},'ready_benchmarks':['MBPP']}))
    config={'paper_context':{'root':str(tmp_path),'run_identity':'one'}}
    assert_target_allowed(config,'target','MBPP')
    with pytest.raises(ValueError,match='blocked'):assert_target_allowed(config,'target','MATH')


def test_eligibility_reports_unchanged_thresholds(tmp_path):
    from test_evolution import runner
    r=runner(tmp_path)
    asyncio.run(r.run(stage='freeze'))
    rows=[json.loads(l) for l in (tmp_path/'candidate_decisions.jsonl').read_text().splitlines()]
    assert rows and all('eligibility' in row for row in rows)
    for row in rows:
        e=row['eligibility']
        assert e['eligible']==all(e['checks'].values())
        assert e['minimum_cost_reduction']==r.config.minimum_cost_reduction


def test_empty_search_is_tool_feedback_not_usage_unknown():
    from evoagentx.compactflow.gaia_tools import GaiaBackends
    backend=object.__new__(GaiaBackends)
    backend.cfg={'search_backend':'bing_html'}
    backend.fetch=lambda url:(b'<html><body>No matching results</body></html>','text/html',url)
    with pytest.raises(ValueError,match='revise the query'):
        backend.tool_search('a neutral query')


@pytest.mark.parametrize("streaming", [False, True])
def test_shared_budget_rejection_is_journaled_without_losing_prior_cost(tmp_path, streaming):
    config = pilot(); config["evaluation"]["task_token_budget"] = 100
    async def run():
        session = ModelSession(config, 42, tmp_path)
        session.used = 9
        session.records.append({"usage": {"prompt_tokens": 5, "completion_tokens": 4}})
        with pytest.raises(TokenBudgetExceeded):
            if streaming:
                async for _ in session.stream([{"role": "user", "content": "q"}], "executor", "one"):
                    pass
            else:
                await session.text([{"role": "user", "content": "q"}], "executor", "one")
        assert session.accounting()["usage_complete"]
        assert session.accounting()["total_tokens"] == 9
        saved = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert saved["charged"] == 0 and saved["records"][0]["request_status"] == "not_sent"
    asyncio.run(run())


@pytest.mark.parametrize("error_type", ["Timeout", "ConnectionError"])
def test_unknown_transport_attempt_is_not_retried(tmp_path, monkeypatch, error_type):
    import requests
    count = []
    def fail(*args, **kwargs):
        count.append(1)
        raise getattr(requests, error_type)("response unavailable")
    monkeypatch.setattr(requests, "post", fail)
    config = pilot(); config["model"]["max_retries"] = 3
    from evoagentx.compactflow.baselines import BaselineUnavailable
    async def run():
        for _ in range(2):
            session = ModelSession(config, 42, tmp_path)
            with pytest.raises(BaselineUnavailable):
                await session.text([{"role":"user","content":"q"}], "executor", "one")
            assert not session.accounting()["usage_complete"]
        assert len(count) == 1
    asyncio.run(run())


def test_known_cost_text_cancellation_resumes_as_failed_not_infrastructure(tmp_path, monkeypatch):
    http_response(monkeypatch, 'x', late=True)
    async def run():
        session=ModelSession(pilot(),42,tmp_path)
        job=asyncio.create_task(session.text([{"role":"user","content":"q"}],"executor","one"))
        await asyncio.sleep(.015)
        job.cancel()
        with pytest.raises(asyncio.CancelledError):await job
        assert session.accounting()["usage_complete"]
        import requests
        monkeypatch.setattr(requests,"post",lambda *a,**k:pytest.fail("cancelled request repeated"))
        restored=ModelSession(pilot(),42,tmp_path)
        with pytest.raises(ValueError,match="cancelled model request"):
            await restored.text([{"role":"user","content":"q"}],"executor","one")
        assert restored.accounting()["usage_complete"]
        assert restored.accounting()["total_tokens"]==18
    asyncio.run(run())



def test_cancel_before_model_slot_is_known_not_sent(tmp_path, monkeypatch):
    import requests
    from evoagentx.compactflow.llm import model_gate
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("request should wait for capacity"))
    config=pilot(); config["model"]["concurrency"]=1
    async def run():
        gate=model_gate(config["model"]["base_url"],1)
        await gate.acquire()
        try:
            session=ModelSession(config,42,tmp_path)
            job=asyncio.create_task(session.text([{"role":"user","content":"q"}],"executor","one"))
            await asyncio.sleep(.01)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):await job
            assert session.accounting()["usage_complete"]
            assert session.accounting()["total_tokens"]==0
            assert session.records[0]["request_status"]=="not_sent"
        finally:gate.release()
    asyncio.run(run())



@pytest.mark.parametrize("streaming", [False, True])
def test_search_budget_refusal_is_known_zero_and_resumable(tmp_path, monkeypatch, streaming):
    import requests
    from evoagentx.compactflow.baseline_native import SearchBudget
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("search budget refused transport"))
    budget=SearchBudget(tmp_path/"budget.json",1)
    async def call(session):
        if streaming:
            async for _ in session.stream([{"role":"user","content":"q"}],"executor","one"):pass
        else:
            await session.text([{"role":"user","content":"q"}],"executor","one")
    async def run():
        session=ModelSession(pilot(),42,tmp_path/"calls",budget=budget)
        with pytest.raises(TokenBudgetExceeded):await call(session)
        assert session.accounting()["usage_complete"]
        assert session.accounting()["total_tokens"]==0
        assert session.records[0]["request_status"]=="not_sent"
        assert session.reserved==0 and budget.used==0
        restored=ModelSession(pilot(),42,tmp_path/"calls",budget=budget)
        with pytest.raises(ValueError):await call(restored)
        assert restored.accounting()["usage_complete"] and budget.used==0
    asyncio.run(run())


def test_native_search_budget_ends_with_existing_candidates(tmp_path):
    from evoagentx.compactflow.baseline_native import NativeAdapter
    class Adapter(NativeAdapter):
        method="aflow"
        def preflight(self,config):self.config=config
        def _search_sync(self):
            self._candidate({"format":"test_canonical"})
            raise TokenBudgetExceeded("bounded search exhausted")
    adapter=Adapter()
    result=asyncio.run(adapter.search([{"benchmark":"MBPP"}],seed=42,config=pilot(),workspace=tmp_path,score=None))
    assert len(result)==1
    saved=json.loads((tmp_path/"candidates.json").read_text())
    assert saved["rejected_exports"][0]["status"]=="budget_exhausted"



def test_native_search_unknown_cost_cannot_become_selected_candidate(tmp_path):
    from evoagentx.compactflow.baseline_native import NativeAdapter
    from evoagentx.compactflow.baselines import BaselineUnavailable
    class Adapter(NativeAdapter):
        method="aflow"
        def preflight(self,config):self.config=config
        def _search_sync(self):
            self._candidate({"format":"test_canonical"})
            self.search_budget.finish("unknown",100,0,False)
    adapter=Adapter()
    with pytest.raises(BaselineUnavailable,match="unknown usage"):
        asyncio.run(adapter.search([{"benchmark":"MBPP"}],seed=42,config=pilot(),workspace=tmp_path,score=None))
    assert json.loads((tmp_path/"candidates.json").read_text())["incomplete"]


def test_unknown_session_cannot_retry_as_a_new_ordinal(tmp_path, monkeypatch):
    import requests
    count=[]
    def fail(*a,**k):
        count.append(1)
        raise requests.Timeout("no response")
    monkeypatch.setattr(requests,"post",fail)
    from evoagentx.compactflow.baselines import BaselineUnavailable
    async def run():
        session=ModelSession(pilot(),42,tmp_path)
        for _ in range(2):
            with pytest.raises(BaselineUnavailable):
                await session.text([{"role":"user","content":"q"}],"executor","one")
        assert len(count)==1
    asyncio.run(run())



def test_response_without_request_accounting_is_not_zero_cost_success(tmp_path):
    class MissingRecords:
        def __init__(self,*a,**k):self.records=[]
        async def text(self,*a,**k):return "answer"
        def accounting(self):return {"usage_complete":True,"total_tokens":0}
    async def run():
        session=ModelSession(pilot(),42,tmp_path,client_factory=MissingRecords)
        await session.text([{"role":"user","content":"q"}],"executor","one")
        assert not session.accounting()["usage_complete"]
        saved=json.loads(next(tmp_path.glob("*.json")).read_text())
        assert not saved["usage_complete"]
    asyncio.run(run())
