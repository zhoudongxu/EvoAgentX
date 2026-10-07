"""Protocol invariants, crash recovery and a complete fake-model integration."""
import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from evoagentx.compactflow.benchmarks import BenchmarkTask, assign_exact_partitions, assign_validation_folds, _group_tasks, _partition_keys
from evoagentx.compactflow.evolution import EvolutionConfig, EvolutionRunner, VariantResult
from evoagentx.compactflow.models import CompactnessPolicy, Evidence, ExecutionFeedback
from evoagentx.compactflow.policy import PolicyLibrary
from examples.compactflow import run_evolution as cli

ROOT = Path(__file__).resolve().parents[3]


def word(i):
    return chr(98+i//26)+chr(98+i%26)


def make_tasks(count=150, benchmark="MATH"):
    return [BenchmarkTask(benchmark,str(i),f"question {word(i)}","1",f"family-{i}") for i in range(count)]


def test_four_benchmarks_have_exact_disjoint_splits_and_folds():
    tasks = sum((make_tasks(150,b) for b in ("MBPP","HotpotQA","MATH","GAIA")),[])
    assign_exact_partitions(tasks,seed=42,validation_folds=5)
    for b in ("MBPP","HotpotQA","MATH","GAIA"):
        cohort = [t for t in tasks if t.benchmark == b]
        assert [sum(t.split == s for t in cohort) for s in ("source","validation","target")] == [90,30,30]
        assert [sum(t.metadata.get("validation_fold")==r for t in cohort) for r in range(1,6)] == [6]*5
        for group in _group_tasks(cohort):
            assert len({t.split for t in group}) == 1
    reference = { (t.benchmark,t.task_id):(t.split,t.metadata.get("validation_fold")) for t in tasks}
    assign_exact_partitions(list(reversed(tasks)),seed=42,validation_folds=5)
    assert reference == {(t.benchmark,t.task_id):(t.split,t.metadata.get("validation_fold")) for t in tasks}


def test_family_template_and_transitive_leakage_never_split():
    tasks = make_tasks()
    tasks[0].family_id = tasks[1].family_id
    tasks[1].metadata["leakage_keys"] = ["shared-entity"]
    tasks[2].metadata["leakage_keys"] = ["shared-entity"]
    tasks[2].question = "What is 12 plus 5?"
    tasks[3].question = "What is 17 plus 9?"
    assign_exact_partitions(tasks,seed=7,validation_folds=5)
    assert len({t.split for t in tasks[:4]}) == 1
    if tasks[0].split == "validation":
        assert len({t.metadata["validation_fold"] for t in tasks[:4]}) == 1


def test_impossible_family_counts_fail_without_splitting():
    tasks = make_tasks()
    for t in tasks:
        t.family_id = "all"
    with pytest.raises(ValueError,match="whole-family"):
        assign_exact_partitions(tasks,seed=42)
    assert all(t.split == "" for t in tasks)


def test_fold_assignment_is_per_benchmark_and_preserves_groups():
    tasks = make_tasks(30)+make_tasks(30,"HotpotQA")
    for t in tasks:
        t.split="validation"
    assign_validation_folds(tasks,rounds=5)
    for b in ("MATH","HotpotQA"):
        assert [sum(t.benchmark==b and t.metadata["validation_fold"]==r for t in tasks) for r in range(1,6)] == [6]*5


def small_tasks(rounds=2, benchmarks=("MATH",)):
    tasks=[]
    for b in benchmarks:
        nsource=1+rounds*2
        cohort=make_tasks(nsource+rounds+2,b)
        for i,t in enumerate(cohort):
            t.split="source" if i<nsource else "validation" if i<nsource+rounds else "target"
            if t.split=="validation":
                t.metadata["validation_fold"]=i-nsource+1
        tasks.extend(cohort)
    return tasks


def config(rounds=2):
    return EvolutionConfig(bootstrap_tasks=1,rounds=rounds,source_tasks_per_round=2,
        max_candidates_per_round=2,validation_folds=rounds,validation_tasks_per_candidate=1,
        heldout_seeds=(101,202),target_seeds=(42,))


def candidate(policy_id, description="prune repeated role"):
    return CompactnessPolicy(id=policy_id,description=description,
        precondition={"capabilities":{"typed_workflow":True}},
        operation={"type":"pruning","constraints":["drop duplicate role"]},
        expected_effect={"tokens":"decrease"},metadata={"contract_check":{"contract_valid":True,"violations":[]}})


def fake_run(task,policies,seed,variant,split):
    cost=100 if variant in ("no_policy","base_planner","candidate_base") else 40
    feedback=ExecutionFeedback(quality=1.,token_cost=cost,latency=cost,graph_cost=cost,valid=True)
    return VariantResult(Evidence(id="callback-id",benchmark=task.benchmark,task_id=task.task_id,
        split=split,variant=variant,seed=seed,policy_ids=tuple(p.id for p in policies),feedback=feedback),
        {"tokens":{"total_tokens":cost,"usage_complete":True}})


def distill(task,a,b,round_index):
    return candidate(f"{task.benchmark}-{task.task_id}-{round_index}")


def runner(path, *, run=fake_run, proposal=distill, rounds=2, **kwargs):
    return EvolutionRunner(small_tasks(rounds),config=config(rounds),library=PolicyLibrary(),
        output=path,run_variant=run,distill_candidate=proposal,code_commit="test-revision",**kwargs)


def test_admit_merge_reject_and_per_pair_negative_evidence(tmp_path):
    def varied(task,policies,seed,variant,split):
        result=fake_run(task,policies,seed,variant,split)
        if split=="validation" and task.metadata["validation_fold"]==2 and variant=="candidate_test":
            result.evidence.feedback.quality=0 if seed==202 else 1
        return result
    r=runner(tmp_path,run=varied)
    result=asyncio.run(r.run())
    assert result.summary["admissions"]=={"admit":1,"merge":1,"reject":2}
    assert len(r.library.all())==1
    policy=r.library.all()[0]
    assert len(policy.metadata["paired_observations"])==4
    rejected=[d["policy"] for d in r._decisions if d["verdict"]=="reject"]
    assert all(p["evidence_ids"] and p["negative_evidence_ids"] for p in rejected)
    assert {e.split for e in r.library.all_evidence()}=={"source","validation"}
    assert all(p.name in {p.name for p in (tmp_path/"policy_snapshots").iterdir()} for p in [
        Path("initial.json"),Path("bootstrap.json"),Path("round-01.json"),Path("round-02.json"),Path("final.json")])


def test_new_runner_resume_never_repeats_completed_calls(tmp_path):
    calls=[]
    def counted(*args):
        calls.append((args[0].task_id,args[2],args[3]))
        return fake_run(*args)
    first=runner(tmp_path,run=counted)
    result=asyncio.run(first.run())
    count=len(calls)
    before=(tmp_path/"policy_snapshots/final.json").read_bytes()
    second=runner(tmp_path,run=counted)
    resumed=asyncio.run(second.run(resume=True))
    assert len(calls)==count
    assert result.summary==resumed.summary
    assert (tmp_path/"policy_snapshots/final.json").read_bytes()==before
    assert len(second._target_records)==4


def test_seed_policy_load_is_digest_stable_without_timestamps(tmp_path):
    payload = {"schema_version": 1, "policies": [{
        "id": "seed", "description": "stable seed", "precondition": {},
        "operation": {}, "expected_effect": {}, "status": "candidate",
    }]}
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(payload))
    first = PolicyLibrary.load(path).to_dict()
    second = PolicyLibrary.load(path).to_dict()
    assert first == second


@pytest.mark.parametrize("interrupt_at",[2,7,14,25,28,32])
def test_interrupted_call_resume_preserves_candidates_and_target(tmp_path,interrupt_at):
    completed=[]
    counter=0
    def unstable(*args):
        nonlocal counter
        counter+=1
        if counter==interrupt_at:
            raise RuntimeError("simulated interruption")
        completed.append((args[0].task_id,args[2],args[3],tuple(p.id for p in args[1])))
        return fake_run(*args)
    try:
        asyncio.run(runner(tmp_path,run=unstable).run())
    except RuntimeError as exc:
        assert "interruption" in str(exc)
    previous=completed[:]
    def remaining(*args):
        key=(args[0].task_id,args[2],args[3],tuple(p.id for p in args[1]))
        # Baseline repeated across distinct candidates is valid; durable call
        # count below is the authoritative candidate-scoped deduplication check.
        completed.append(key)
        return fake_run(*args)
    r=runner(tmp_path,run=remaining)
    asyncio.run(r.run(resume=True))
    assert len(completed)==len(list((tmp_path/"calls").glob("*.json")))
    assert r._completed_rounds==2 and len(r._decisions)==4 and len(r._target_records)==4
    assert completed[:len(previous)]==previous


@pytest.mark.parametrize("mutation",["config","task","code","data","manifest","final"])
def test_resume_detects_immutable_inputs(tmp_path,mutation):
    r=runner(tmp_path,data_identity={"raw":"a"})
    asyncio.run(r.run())
    kwargs={"data_identity":{"raw":"a"}}
    r2=runner(tmp_path,**kwargs)
    if mutation=="config": r2.config=replace(r2.config,source_seed=9)
    elif mutation=="task": r2.tasks[0].answer="changed gold"
    elif mutation=="code": r2.code_commit="other"
    elif mutation=="data": r2.data_identity={"raw":"b"}
    elif mutation=="manifest": (tmp_path/"sample_manifest.jsonl").write_text("{}\n")
    else: (tmp_path/"policy_snapshots/final.json").write_text("{}")
    with pytest.raises(ValueError):
        asyncio.run(r2.run(resume=True))


def test_mismatched_pair_keys_fail(tmp_path):
    def wrong(*args):
        r=fake_run(*args)
        r.evidence.seed=999
        return r
    with pytest.raises(ValueError,match="key mismatch"):
        asyncio.run(runner(tmp_path,run=wrong).run())


def test_target_callbacks_receive_copies_and_cannot_update_library(tmp_path):
    def malicious(task,policies,seed,variant,split):
        if split=="target" and policies:
            policies[0].utility=0
            policies[0].description="corrupted"
        return fake_run(task,policies,seed,variant,split)
    r=runner(tmp_path,run=malicious)
    asyncio.run(r.run())
    assert r.library.to_dict()==json.loads((tmp_path/"policy_snapshots/final.json").read_text())


def test_fold_binding_and_multi_benchmark_source_budgets(tmp_path):
    r=EvolutionRunner(small_tasks(2,("MATH","HotpotQA")),config=config(),library=PolicyLibrary(),
        output=tmp_path,run_variant=fake_run,distill_candidate=distill)
    asyncio.run(r.run())
    assert {b:sum(x["benchmark"]==b for x in r._source_records) for b in ("MATH","HotpotQA")}=={"MATH":10,"HotpotQA":10}
    assert all(x["round"]==x["validation_fold"] for x in r._validation_records)


def test_unavailable_baselines_and_missing_benchmarks_do_not_get_results(tmp_path):
    payload={"datasets":[{"name":b} for b in ("MBPP","MATH","HotpotQA","GAIA")]}
    r=runner(tmp_path,coverage={"aflow":{"status":"incomplete"}},config_payload=payload)
    result=asyncio.run(r.run())
    assert result.summary["publication_status"]=="incomplete"
    assert result.summary["datasets"]["GAIA"]["status"]=="incomplete"
    assert {r["variant"] for r in r._target_records}=={"base_planner","compactflow"}


class FakeModelClient:
    response=None
    checker={"contract_valid":True,"violations":[]}
    calls=[]
    def __init__(self,*args,**kwargs): self.records=[]
    async def json(self,system,request,component,**kwargs):
        self.calls.append(component)
        assert "PRIVATE GOLD" not in request
        if component=="distiller":
            if self.response is not None: return self.response
            return {"candidate":{"description":"prune redundant role","precondition":{"capabilities":{"typed_workflow":True}},
                "operation":{"type":"pruning","constraints":["drop duplicate role"]},"expected_effect":{"tokens":"decrease"},"failure_modes":[]}}
        if component=="checker": return self.checker
        if component=="query": return {"query":"prune repeated role"}
        if component=="selector": return {"selected":[p["id"] for p in json.loads(request)["candidate_policies"]]}
        data=json.loads(request)
        return {"nodes":[{"id":"answer","tool":"llm","instruction":"solve","inputs":{"q":"$input.question"},"outputs":["answer"]}],
                "sinks":["answer"],"applied_policies":[{"id":p["id"],"evidence":"removed redundant role"} for p in data["selected_policies"]],"unapplied_policies":[]}
    async def stream(self,*args,**kwargs):
        self.calls.append("executor")
        yield '{"field":"answer","value":"42"}\n'
    def accounting(self): return {"total_tokens":40,"usage_complete":True}


def pilot():
    return json.loads((ROOT/"examples/compactflow/configs/qwen3_coder_a100.pilot.json").read_text())


@pytest.mark.parametrize("response,check",[
    ({"candidate":{"description":"missing fields"}},{"contract_valid":True,"violations":[]}),
    (["not an object"],{"contract_valid":True,"violations":[]}),
    (None,{"contract_valid":True}),
    (None,{"contract_valid":False,"violations":["unsafe"]}),
])
def test_malformed_distillation_and_contract_fail_closed(monkeypatch,response,check):
    class Client(FakeModelClient): pass
    Client.response,Client.checker=response,check
    monkeypatch.setattr(cli,"ModelClient",Client)
    task=small_tasks()[1]
    a=fake_run(task,(),42,"no_policy","source")
    b=fake_run(task,(),42,"source_library","source")
    result=asyncio.run(cli.LiveAdapter(pilot(),PolicyLibrary()).distill_candidate(task,a,b,1))
    assert result.candidate is None
    assert result.reason


def test_live_adapter_with_fake_model_full_evolution(monkeypatch,tmp_path):
    monkeypatch.setattr(cli,"ModelClient",FakeModelClient)
    FakeModelClient.calls=[]
    c=pilot()
    c["construction"].update(minimum_semantic_score=-1,minimum_selection_score=0)
    adapter=cli.LiveAdapter(c,PolicyLibrary())
    tasks=small_tasks(1)
    for t in tasks: t.answer="PRIVATE GOLD"
    def run(task,policies,seed,variant,split):
        # Force source eligibility with measured fixture baseline, but execute
        # every model adapter component through the fake client/runtime.
        async def call():
            out=await adapter.run_variant(task,policies,seed,variant,split)
            out.evidence.feedback.token_cost=100 if variant in ("no_policy","candidate_base","base_planner") else 40
            out.evidence.feedback.latency=100 if variant in ("no_policy","candidate_base","base_planner") else 40
            return out
        return call()
    r=EvolutionRunner(tasks,config=config(1),library=PolicyLibrary(),output=tmp_path,run_variant=run,
                      distill_candidate=adapter.distill_candidate)
    asyncio.run(r.run())
    assert {"planner","executor","distiller","checker","query","selector"}.issubset(FakeModelClient.calls)
    assert r._validation_records and r._target_records
    assert {d["verdict"] for d in r._decisions} == {"admit", "merge"}
    assert all(row["record"]["tokens"]["usage_complete"] for row in r._records.values())


def test_failed_executor_answer_scores_zero_in_feedback(monkeypatch):
    class PartialThenMalformed(FakeModelClient):
        async def stream(self, *args, **kwargs):
            yield '{"field":"answer","value":"42"}\n'
            yield 'invalid trailing JSON'
    monkeypatch.setattr(cli, "ModelClient", PartialThenMalformed)
    task = BenchmarkTask("MATH", "failed-answer", "Return the number", "42", "family")
    result = asyncio.run(cli.LiveAdapter(pilot(), PolicyLibrary()).run_variant(task, (), 42, "base_planner", "target"))
    assert result.record["evaluation"]["quality"] == 1.0
    assert result.record["errors"]
    assert not result.evidence.feedback.valid
    assert result.evidence.feedback.quality == 0.0


def test_real_seed_library_resume_uses_same_initial_digest(tmp_path):
    seed_path = ROOT / "examples/compactflow/policies/seed_policies.json"
    calls = []
    def counted(*args):
        calls.append(args[3])
        return fake_run(*args)
    def seeded():
        return EvolutionRunner(small_tasks(1), config=config(1), library=PolicyLibrary.load(seed_path),
                               output=tmp_path, run_variant=counted, code_commit="seed-resume-test")
    first = asyncio.run(seeded().run())
    count = len(calls)
    final = (tmp_path / "policy_snapshots/final.json").read_bytes()
    second = asyncio.run(seeded().run(resume=True))
    assert first.summary == second.summary
    assert len(calls) == count
    assert (tmp_path / "policy_snapshots/final.json").read_bytes() == final


def raw_bundle(tmp_path,count=15):
    c=pilot()
    c["datasets"]=[d for d in c["datasets"] if d["name"]=="MATH"]
    d=c["datasets"][0]
    rows=[{"id":str(i),"problem":f"problem {word(i)}","solution":"42","type":"algebra"} for i in range(count)]
    path=tmp_path/"MATH"/"data.json"
    path.parent.mkdir()
    path.write_text(json.dumps(rows))
    entries=[{"benchmark":"MATH","repository_relative_path":"data.json","original_split":d["original_splits"][0],
        "url":f"https://huggingface.co/datasets/{d['repository']}/resolve/{d['revision']}/data.json",
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}]
    (tmp_path/"sources.json").write_text(json.dumps(entries))
    return c,path


def test_raw_bundle_checksums_revision_counts_and_incomplete(tmp_path):
    c,p=raw_bundle(tmp_path)
    bundle=cli.load_raw_bundle(tmp_path,c)
    assert len(bundle.tasks)==15
    assert [sum(t.split==s for t in bundle.tasks) for s in ("source","validation","target")]==[9,3,3]
    p.write_text(p.read_text()+" ")
    with pytest.raises(ValueError,match="checksum"): cli.load_raw_bundle(tmp_path,c)


def test_raw_bundle_impossible_groups_and_gaia_are_incomplete(tmp_path):
    c,p=raw_bundle(tmp_path)
    rows=json.loads(p.read_text())
    for row in rows: row["family_id"]="one"
    p.write_text(json.dumps(rows))
    entries=json.loads((tmp_path/"sources.json").read_text())
    entries[0].update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
    (tmp_path/"sources.json").write_text(json.dumps(entries))
    c["datasets"] += [d for d in pilot()["datasets"] if d["name"]=="GAIA"]
    b=cli.load_raw_bundle(tmp_path,c)
    assert not b.tasks
    assert all(v["status"]=="incomplete" for v in b.coverage.values())
