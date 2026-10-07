"""Paper experiment coordinator: source discovery -> global freeze -> target -> replay."""
from __future__ import annotations
import asyncio
import copy
import csv
import fcntl
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path

from .baselines import ConstructionBaselineRunner, REQUIRED_METHODS, REQUIRED_BENCHMARKS, SUPPORTED_METHODS
from .evolution import EvolutionRunner, EvolutionConfig, atomic_json
from .models import CompactnessPolicy, PolicyStatus
from .policy import PolicyLibrary
from .replay import digest
from .paper_phase import file_digest, assert_target_allowed
from .paper_studies import (STUDIES, allocation, construction_variants, interventions,
                            manifest_rows, replay_cases, replay_one)


class PaperExperimentRunner:
    def __init__(self, tasks, *, config, output, data_identity, formal_manifest_identity,
                 dataset_coverage, studies=STUDIES, client_factory=None, embedder=None,
                 adapters_factory=None):
        self.tasks = tuple(tasks)
        self.config = copy.deepcopy(config)
        self.output = Path(output).resolve()
        self.data_identity = data_identity
        self.formal_manifest_identity = formal_manifest_identity
        self.coverage = copy.deepcopy(dataset_coverage)
        self.studies = tuple(studies)
        if set(studies)-set(STUDIES) or not studies:
            raise ValueError("unknown or empty paper studies")
        self.benchmarks = sorted({t.benchmark for t in tasks})
        self.client_factory, self.embedder, self.adapters_factory = client_factory, embedder, adapters_factory
        self.schedule = allocation(config, self.benchmarks, studies)
        self.variants = construction_variants(self.config, studies)
        self.phase = "prepared"
        self.expected = []

    def _execution_only_runner(self):
        from .execution_capture import ExecutionCaptureRunner
        return ExecutionCaptureRunner(self.tasks,config=self.config,output=self.output/'capture_run',
            data_identity=self.data_identity,formal_manifest_identity=self.formal_manifest_identity,
            client_factory=self.client_factory)

    def plan(self):
        if self.studies == ('execution_main',):
            return self._execution_only_runner().plan()
        return {**self.schedule, "variants": {k:{x:y for x,y in v.items() if x!='config'} for k,v in self.variants.items()},
                "stages":["source_diagnostic", "evolution", "baseline_selection", "freeze", "target", "offline_replay", "report"]}

    def _prepare(self, resume):
        self.output.mkdir(parents=True, exist_ok=True)
        code = ConstructionBaselineRunner._code_identity(self)
        expected = {"schema_version":1, "config":self.config, "manifest":digest(manifest_rows(self.tasks)),
                    "data":digest(self.data_identity), "formal_manifest":self.formal_manifest_identity,
                    "code":code, "studies":list(self.studies)}
        lock = self.output/'paper.lock.json'
        if lock.exists():
            if not resume: raise FileExistsError("immutable paper output exists; use --resume")
            if json.loads(lock.read_text()) != expected: raise ValueError("paper lock changed: code/config/data/manifest/studies")
        else:
            if {p.name for p in self.output.iterdir()} - {'.process.lock'}:
                raise ValueError("nonempty unlocked paper output")
            atomic_json(lock, expected)
            atomic_json(self.output/'config.lock.json', self.config)
            atomic_json(self.output/'environment.lock.json', {"python":platform.python_version(),"code":code})
            atomic_json(self.output/'experiment_manifest.json', self.plan())
            (self.output/'sample_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in manifest_rows(self.tasks)))
        if digest([json.loads(l) for l in (self.output/'sample_manifest.jsonl').read_text().splitlines()]) != expected['manifest']:
            raise ValueError("paper manifest tampered")
        self.run_identity = digest(expected)
        self.config['paper_context'] = {"root":str(self.output), "run_identity":self.run_identity}
        self.variants = construction_variants(self.config, self.studies)
        self.schedule = allocation(self.config, self.benchmarks, self.studies)
        if (self.output/'checkpoint.json').exists():
            self.phase = json.loads((self.output/'checkpoint.json').read_text())['phase']
        from .gaia import initialize_gaia_run
        initialize_gaia_run(self.output, self.config, self.tasks)
        for key, variant in self.variants.items():
            p=self.output/'variant_configs'/f'{key}.json'
            if p.exists() and json.loads(p.read_text()) != variant['config']:
                raise ValueError("frozen variant configuration changed")
            if not p.exists(): atomic_json(p, variant['config'])

    def _checkpoint(self, phase):
        self.phase = phase
        print(json.dumps({'phase':phase,'event':'paper_checkpoint'}),flush=True)
        atomic_json(self.output/'checkpoint.json', {"phase":phase, "run_identity":self.run_identity})

    def _tasks(self, split, benchmark, count=None):
        health=self.output/'discovery_health.json'
        if split=='target' and health.exists() and benchmark in json.loads(health.read_text())['blocked_benchmarks']:
            return []
        rows=sorted((t for t in self.tasks if t.split==split and t.benchmark==benchmark),
                    key=lambda t:digest([self.config['partition']['sample_seed'],t.task_id]))
        return rows if count is None else rows[:count]

    def _library(self, key='base', snapshot='final'):
        alias=self.variants[key].get('alias') or ('base' if key=='static_library' else key)
        path=self.output/'evolution'/alias/'policy_snapshots'/f'{snapshot}.json'
        if path.exists():
            value=json.loads(path.read_text())
        else:
            from examples.compactflow.run_evolution import ROOT
            value=json.loads((ROOT/self.config['construction']['library']['initial_path']).read_text())
        from examples.compactflow.run_evolution import PinnedEmbedder
        return PolicyLibrary([CompactnessPolicy.from_dict(p) for p in value['policies']],
                             embedder=self.embedder or PinnedEmbedder(self.config['encoder']))

    def _evolution(self, key):
        from examples.compactflow.run_evolution import LiveAdapter, build_admission, ROOT, PinnedEmbedder
        cfg=self.variants[key]['config']
        output=self.output/'evolution'/key
        library=PolicyLibrary.load(ROOT/cfg['construction']['library']['initial_path'],embedder=self.embedder or PinnedEmbedder(cfg['encoder']))
        kwargs={"client_factory":self.client_factory} if self.client_factory else {}
        adapter=LiveAdapter(cfg,library,output=output,**kwargs)
        runner=EvolutionRunner(self.tasks, config=EvolutionConfig.from_mapping(cfg), library=library,output=output,
            run_variant=adapter.run_variant,distill_candidate=adapter.distill_candidate,admission=build_admission(cfg),
            config_payload=cfg,data_identity=self.data_identity,coverage=self.coverage)
        async def distill(*args):
            adapter.library=runner.library
            return await adapter.distill_candidate(*args)
        runner.distill_candidate=distill
        return runner

    def _baseline(self):
        from examples.compactflow.run_baselines import build_adapters
        output=self.output/'construction'; evolution=self.output/'evolution'/'base'
        cfg=copy.deepcopy(self.config);cfg['baseline_runner']['benchmarks']=self.benchmarks
        health=self.output/'discovery_health.json'
        cfg['baseline_runner']['blocked_benchmarks']=json.loads(health.read_text()).get('blocked_benchmarks',{}) if health.exists() else {}
        methods = tuple(self.config.get("baseline_runner", {}).get("required_methods", REQUIRED_METHODS))
        unknown = set(methods) - set(SUPPORTED_METHODS)
        if unknown:
            raise ValueError(f"unsupported paper baseline methods: {sorted(unknown)}")
        adapters=(self.adapters_factory or build_adapters)(methods,evolution)
        return ConstructionBaselineRunner(self.tasks,output=output,adapters=adapters,config=cfg,
            manifest=self.output/'sample_manifest.jsonl',evolution_dir=evolution,dataset_coverage=self.coverage)

    async def _capture(self, task, seed, variant, *, workflow=None, base_only=False):
        from examples.compactflow.run_evolution import LiveAdapter
        assert_target_allowed(self.config,task.split,task.benchmark)
        identity=[task.benchmark,task.task_id,task.split,seed,variant,workflow,base_only]
        key=digest(identity);path=self.output/'captures'/f'{key}.json'
        if path.exists():
            saved=json.loads(path.read_text())
            if saved['identity'] != identity or digest(saved['value']) != saved['digest']:
                raise ValueError("capture integrity mismatch")
            return saved['value']
        cfg=self.variants.get(variant,self.variants['base'])['config']
        library=self._library(variant if variant in self.variants else 'base', 'bootstrap' if variant=='static_library' else 'final')
        policies=() if base_only else tuple(p for p in library.all() if p.status==PolicyStatus.VERIFIED or
                     (variant in {'static_library','without_heldout_verification'} and p.status!=PolicyStatus.REJECTED))
        kwargs={"client_factory":self.client_factory} if self.client_factory else {}
        adapter=LiveAdapter(cfg,library,output=self.output,**kwargs)
        result=await adapter.run_variant(task,policies,seed,variant,task.split,workflow_spec=workflow)
        value={"benchmark":task.benchmark,"task_id":task.task_id,"family_id":task.family_id,"split":task.split,
               "seed":seed,"variant":variant,"feedback":result.evidence.feedback.to_dict(),"record":result.record,
               "status":"complete" if result.record.get('tokens',{}).get('usage_complete') and result.record.get('infrastructure_status','complete')=='complete' else "incomplete"}
        atomic_json(path,{"identity":identity,"value":value,"digest":digest(value)})
        return value

    def _save(self, study, key, row):
        path=self.output/'studies'/study/'records'/f'{digest(key)}.json'
        if path.exists():
            old=json.loads(path.read_text())
            if old['key']!=key: raise ValueError('study record key mismatch')
            return old['record']
        atomic_json(path, {"key":key,"record":row})
        return row

    def _existing(self, study, key):
        path=self.output/'studies'/study/'records'/f'{digest(key)}.json'
        return json.loads(path.read_text())['record'] if path.exists() else None

    async def _diagnostic(self):
        if 'construction_diagnostic' not in self.studies: return
        settings=self.schedule['construction_diagnostic']
        for b in self.benchmarks:
            for task in self._tasks('source',b,settings['tasks']):
                for seed in settings['seeds']:
                    base=await self._capture(task,seed,'diagnostic_base',base_only=True)
                    spec=base['record'].get('workflow')
                    candidates,reasons=interventions(spec,task.public_input(),settings['max_interventions']) if spec and spec.get('nodes') else ([],{'all':'base workflow unavailable'})
                    rows=[]
                    for proposal in candidates:
                        case=await self._capture(task,seed,'intervention_'+proposal['kind'],workflow=proposal['workflow'],base_only=True)
                        a,c=base['feedback'],case['feedback']
                        fields=('token_cost','latency','node_count','edge_count')
                        quality_ok=bool(a['valid'] and c['valid'] and c['quality']>=a['quality']-self.config['construction']['quality_tolerance'])
                        measurable=all(a.get(f) is not None and c.get(f) is not None for f in fields)
                        beneficial=quality_ok and measurable and any(c[f]<a[f] for f in fields) and all(c[f]<=a[f]*(1+self.config['construction'].get('other_cost_tolerance',.05))+1e-9 for f in fields)
                        rows.append({"kind":proposal['kind'],"motif":proposal['motif'],"nodes":proposal['nodes'],
                                     "beneficial":beneficial,"quality_preserved":quality_ok,"feedback":c,"status":case['status'],
                                     "removed_nodes":sorted({n['id'] for n in spec['nodes']}-{n['id'] for n in proposal['workflow']['nodes']}),
                                     "removed_edges":sorted({(ref.split('.')[0],n['id']) for n in spec['nodes'] for ref in n['inputs'].values() if not ref.startswith('$input.')}-{(ref.split('.')[0],n['id']) for n in proposal['workflow']['nodes'] for ref in n['inputs'].values() if not ref.startswith('$input.')})})
                    key=[b,task.task_id,seed]
                    self._save('construction_diagnostic',key,{"benchmark":b,"task_id":task.task_id,"seed":seed,
                        "family_id":task.family_id,"baseline":base['feedback'],"interventions":rows,"not_applicable":reasons,
                        "status":"complete" if base['status']=='complete' and all(r['status']=='complete' for r in rows) else 'incomplete'})

    def _discovery_health(self):
        blocked = {}
        for path in sorted(list((self.output/'evolution').glob('*/calls/*.json')) + list((self.output/'evolution').glob('*/candidates/*.json'))):
            row=json.loads(path.read_text());record=row.get('record',{})
            if path.parent.name=='candidates' and not record: continue
            if not record.get('tokens',{}).get('usage_complete') or record.get('infrastructure_status','complete')!='complete':
                blocked.setdefault(row['benchmark'],[]).append({
                    "record":str(path.relative_to(self.output)),
                    "usage_complete":record.get('tokens',{}).get('usage_complete'),
                    "infrastructure_status":record.get('infrastructure_status','complete')})
        value={"blocked_benchmarks":blocked,"ready_benchmarks":[b for b in self.benchmarks if b not in blocked]}
        path=self.output/'discovery_health.json'
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError("frozen discovery health changed")
        if not path.exists():atomic_json(path,value)
        return value

    def _freeze(self):
        artifacts={}
        for pattern in ['sample_manifest.jsonl','variant_configs/*.json','evolution/*/policy_snapshots/*.json',
                        'construction/baselines/*/*/*/selected.json']:
            for p in sorted(self.output.glob(pattern)):
                artifacts[str(p.relative_to(self.output))]=file_digest(p)
        health=self._discovery_health()
        for name in ('discovery_health.json','construction/freeze_summary.json'):
            p=self.output/name
            if not p.exists(): raise ValueError('discovery must finish before target: '+name)
            artifacts[name]=file_digest(p)
        for key,v in self.variants.items():
            path=self.output/'evolution'/(v.get('alias') or ('base' if key=='static_library' else key))/'policy_snapshots/final.json'
            if not path.exists(): raise ValueError('variant not frozen: '+key)
        for p in (self.output/'studies/construction_diagnostic/records').glob('*.json'):
            if json.loads(p.read_text())['record']['status']!='complete':
                raise ValueError('incomplete source diagnostic; target remains closed')
        payload={"run_identity":self.run_identity,"artifacts":artifacts,"ready_benchmarks":health["ready_benchmarks"]}
        lock=self.output/'target_gate.lock.json'
        if lock.exists() and json.loads(lock.read_text())!=payload: raise ValueError('paper freeze changed')
        if not lock.exists(): atomic_json(lock,payload)
        assert_target_allowed(self.config,'target')

    async def _variant_targets(self):
        for variant, settings in self.variants.items():
            if variant=='base': continue
            study='construction_ablations' if settings['kind']=='ablation' else 'construction_sensitivity'
            for b in self.benchmarks:
                for task in self._tasks('target',b,self.schedule['variant_target_tasks']):
                    for seed in self.config['evaluation']['generation_seeds']:
                        key=[variant,b,task.task_id,seed]
                        if self._existing(study,key):continue
                        base=await self._capture(task,seed,'base')
                        actual=settings.get('alias') or variant
                        tested=base if actual=='base' else await self._capture(task,seed,actual)
                        self._save(study,key,{"variant":variant,"axis":settings['axis'],"value":settings['value'],
                            "benchmark":b,"task_id":task.task_id,"seed":seed,"family_id":task.family_id,
                            "baseline":base['feedback'],"candidate":tested['feedback'],
                            "status":"complete" if base['status']==tested['status']=='complete' else 'incomplete'})

    async def _execution(self):
        cases=replay_cases(self.config,self.studies)
        if not cases:return
        settings=self.schedule['execution']
        for b in self.benchmarks:
            for task in self._tasks('target',b,settings['tasks']):
                for seed in settings['seeds']:
                    capture=await self._capture(task,seed,'base')
                    for repetition in range(-settings['warmups'],settings['repetitions']):
                        order=list(cases);random.Random(digest([b,task.task_id,seed,repetition])).shuffle(order)
                        for case in order:
                            key=[b,task.task_id,seed,case['name'],repetition]
                            if self._existing(case['study'],key):continue
                            try:
                                result=await replay_one(capture['record'],task.public_input(),self.config,case)
                            except Exception as exc:
                                result={'status':'incomplete','reason':type(exc).__name__+': '+str(exc)}
                            self._save(case['study'],key,{**result,"benchmark":b,"task_id":task.task_id,
                                "family_id":task.family_id,"seed":seed,"case":case,"repetition":repetition,
                                "warmup":repetition<0})

    async def run(self, *, resume=False):
        if self.studies == ('execution_main',):
            from .execution_capture import run_execution_comparison
            return await run_execution_comparison(self._execution_only_runner(),resume=resume)
        self.output.mkdir(parents=True,exist_ok=True)
        with (self.output/'.process.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self._prepare(resume)
            if self.phase=='complete':return self.report()
            if not (self.output/'target_gate.lock.json').exists():
                await self._diagnostic();self._checkpoint('discovery')
                for key,v in self.variants.items():
                    if v.get('alias') or key=='static_library':continue
                    runner=self._evolution(key)
                    await runner.run(stage='freeze',resume=(runner.output/'run.lock.json').exists())
                    self._checkpoint('discovery:'+key)
                self._discovery_health()
                baseline=self._baseline()
                await baseline.run(stage='freeze',resume=(baseline.output/'baseline.lock.json').exists())
                self._freeze();self._checkpoint('frozen')
            self._freeze()
            # No discovery code is invoked after the global gate opens.
            baseline=self._baseline()
            await baseline.run(stage='target',resume=True)
            self._checkpoint('target_main')
            await self._variant_targets();self._checkpoint('target_variants')
            await self._execution();self._checkpoint('replay_complete')
            result=self.report()
            if result['selected_run_status']=='complete':self._checkpoint('complete')
            return result

    def report(self):
        if self.studies == ('execution_main',):
            from .latency import LatencyReplayRunner
            return LatencyReplayRunner.report(self.output)
        from .paper_reporting import report_paper
        return report_paper(self.output)
