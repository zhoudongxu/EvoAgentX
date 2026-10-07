"""Independent, no-policy fixed-workflow capture for execution-plane comparisons."""
from __future__ import annotations

import copy
import fcntl
import json
import platform
import statistics
import time
from pathlib import Path

from .evolution import atomic_json
from .paper_phase import assert_target_allowed, file_digest
from .paper_studies import allocation, manifest_rows
from .replay import digest


def _files(root):
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in {'.process.lock', 'progress.json', 'capture.lock.json'}}


def verify_capture_lock(root, *, required=False):
    root = Path(root)
    path = root/'capture.lock.json'
    if not path.exists():
        if required:
            raise ValueError('execution captures are not frozen')
        return
    lock = json.loads(path.read_text())
    if digest(lock['files']) != lock['digest'] or _files(root) != lock['files']:
        raise ValueError('frozen execution capture changed')


class ExecutionCaptureRunner:
    """Only the shared Base Planner and complete-dependency trace collection run live."""
    def __init__(self, tasks, *, config, output, data_identity, formal_manifest_identity,
                 client_factory=None):
        self.tasks = tuple(tasks)
        self.config = copy.deepcopy(config)
        self.config.pop('paper_context', None)
        self.output = Path(output).resolve()
        self.data_identity = data_identity
        self.formal_manifest_identity = formal_manifest_identity
        self.client_factory = client_factory
        self.benchmarks = sorted({t.benchmark for t in self.tasks})
        self.settings = allocation(config, self.benchmarks, ['execution_main'])['execution']
        self.selected = []
        for b in self.benchmarks:
            targets = sorted((t for t in self.tasks if t.benchmark == b and t.split == 'target'),
                             key=lambda t: digest([config['partition']['sample_seed'], t.task_id]))
            if len(targets) < self.settings['tasks']:
                raise ValueError(b + ': insufficient fixed target tasks')
            self.selected.extend((t, seed) for t in targets[:self.settings['tasks']] for seed in self.settings['seeds'])
        if not self.selected:
            raise ValueError('empty execution selection')

    def plan(self):
        from .latency import EXECUTION_METHODS
        from .execution_baselines import execution_coverage
        n = len(self.selected)
        return {'schema_version': 2, 'scope':'execution_only', 'studies':['execution_main'],
                'benchmarks':self.benchmarks, 'methods':list(EXECUTION_METHODS), 'execution':self.settings,
                'workflow_source':'Base Planner without policies; identical graph and call outputs for every scheduler',
                'stages':['freeze_protocol', 'base_workflow_capture', 'freeze_captures', 'offline_replay', 'report'],
                'captures':n, 'timed_records':n*3*self.settings['repetitions'],
                'warmup_records':n*3*self.settings['warmups'],
                'pilot_source_only':self.config['profile']=='pilot',
                'execution_external':execution_coverage()}

    def _prepare(self, resume):
        self.config.pop("paper_context", None)
        from .baselines import ConstructionBaselineRunner
        expected = {'config':self.config, 'data':self.data_identity,
                    'formal_manifest':self.formal_manifest_identity,
                    'manifest':digest(manifest_rows(self.tasks)),
                    'selected':[[t.benchmark,t.task_id,seed] for t,seed in self.selected],
                    'code':ConstructionBaselineRunner._code_identity(None), 'scope':'execution_only'}
        lock = self.output/'execution_capture.lock.json'
        if lock.exists():
            if not resume:
                raise ValueError('immutable execution output exists; use --resume')
            if json.loads(lock.read_text()) != expected:
                raise ValueError('execution lock changed: code/config/data/manifest')
            if (self.output/'capture.lock.json').exists():
                verify_capture_lock(self.output, required=True)
        else:
            if set(p.name for p in self.output.iterdir()) - {'.process.lock'}:
                raise ValueError('nonempty unlocked capture output')
            atomic_json(lock, expected)
            atomic_json(self.output/'config.lock.json', self.config)
            atomic_json(self.output/'environment.lock.json', {'python':platform.python_version(), 'code':expected['code']})
            atomic_json(self.output/'experiment_manifest.json', self.plan())
            (self.output/'sample_manifest.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in manifest_rows(self.tasks)))
        if digest([json.loads(s) for s in (self.output/'sample_manifest.jsonl').read_text().splitlines()]) != expected['manifest']:
            raise ValueError('execution manifest changed')
        for name,value in [('config.lock.json',self.config),('experiment_manifest.json',self.plan())]:
            if json.loads((self.output/name).read_text()) != value:
                raise ValueError('execution frozen protocol changed: '+name)
        run_identity = digest(expected)
        self.config['paper_context'] = {'root':str(self.output), 'run_identity':run_identity}
        if not (self.output/'capture.lock.json').exists():
            from .gaia import initialize_gaia_run
            initialize_gaia_run(self.output,self.config,self.tasks)
        names = ['execution_capture.lock.json','config.lock.json','experiment_manifest.json','sample_manifest.jsonl','environment.lock.json']
        names += [n for n in ('gaia_assets.lock.json','auxiliary_models.lock.json','gaia_capabilities.json') if (self.output/n).exists()]
        gate = {'run_identity':run_identity, 'scope':'execution_only',
                'artifacts':{n:file_digest(self.output/n) for n in names}}
        p = self.output/'target_gate.lock.json'
        if p.exists() and json.loads(p.read_text()) != gate:
            raise ValueError('execution target gate changed')
        if not p.exists():
            atomic_json(p,gate)
        assert_target_allowed(self.config,'target')

    async def _capture(self, task, seed):
        from examples.compactflow.run_evolution import LiveAdapter
        from .policy import PolicyLibrary
        identity = [task.benchmark,task.task_id,'target',seed,'base',None,False]
        key = digest(identity)
        path = self.output/'captures'/(key+'.json')
        workflow_path = self.output/'workflows'/(key+'.json')
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] != identity or digest(saved['value']) != saved['digest']:
                raise ValueError('execution capture identity/hash mismatch')
            if not saved['value']['record'].get('workflow'):
                if saved['value']['status'] != 'incomplete':raise ValueError('complete capture lacks workflow')
                return saved['value']
            frozen = json.loads(workflow_path.read_text())
            if frozen['digest'] != digest(frozen['workflow']) or frozen['identity'] != identity or frozen['workflow'] != saved['value']['record']['workflow']:
                raise ValueError('frozen workflow changed')
            return saved['value']
        assert_target_allowed(self.config, 'target')
        def freeze_workflow(spec):
            value = {'identity':identity,'workflow':spec,'digest':digest(spec)}
            if workflow_path.exists() and json.loads(workflow_path.read_text()) != value:
                raise ValueError('workflow changed on resume')
            if not workflow_path.exists():
                atomic_json(workflow_path,value)
        # An empty library never embeds, retrieves, selects, or evolves a policy.
        kwargs = {'client_factory':self.client_factory} if self.client_factory else {}
        adapter = LiveAdapter(self.config,PolicyLibrary([]),output=self.output,**kwargs)
        started = time.perf_counter()
        result = await adapter.run_variant(task,(),seed,'base','target',workflow_observer=freeze_workflow)
        record = result.record
        graph = record.get('graph') or {}
        calls = {c['id'] for c in graph.get('calls',[])}
        recorded = {r['call_id'] for r in record.get('replay',{}).get('records',{}).values()}
        errors = record.get('errors',{})
        bounded = ('timeout','timed out','budget exhausted','step budget','tool-call budget')
        normal_failure = bool(errors) and all(any(w in str(e).lower() for w in bounded) for e in errors.values())
        complete = (record.get('tokens',{}).get('usage_complete') and calls and calls==recorded
                    and (not errors or normal_failure))
        planner_requests = [r for r in record.get('model_requests',[]) if r.get('component')=='planner']
        costs = {'planner_tokens':sum(r.get('usage',{}).get('total_tokens',
                     r.get('usage',{}).get('prompt_tokens',0)+r.get('usage',{}).get('completion_tokens',0)) for r in planner_requests),
                 'planner_seconds':sum(r.get('elapsed_seconds',0) for r in planner_requests),
                 'total_tokens':record.get('tokens',{}).get('total_tokens'),
                 'physical_tokens':record.get('tokens',{}).get('physical_tokens'),
                 'collection_wall_seconds':time.perf_counter()-started,
                 'execution_seconds':record.get('runtime_metrics',{}).get('latency'),
                 'tool_accounting':record.get('tool_accounting',{})}
        value = {'benchmark':task.benchmark,'task_id':task.task_id,'family_id':task.family_id,
                 'split':'target','seed':seed,'variant':'base', 'record':record,
                 'feedback':result.evidence.feedback.to_dict(), 'costs':costs,
                 'status':'complete' if complete else 'incomplete'}
        atomic_json(path, {'identity':identity,'value':value,'digest':digest(value)})
        return value

    async def run(self, *, resume=False):
        self.output.mkdir(parents=True,exist_ok=True)
        with (self.output/'.process.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self._prepare(resume)
            if (self.output/'capture.lock.json').exists():
                return json.loads((self.output/'capture_summary.json').read_text())
            rows=[]
            for task,seed in self.selected:
                rows.append(await self._capture(task,seed))
                elapsed=[r['costs']['collection_wall_seconds'] for r in rows]
                progress={'phase':'capture','completed':len(rows),'expected':len(self.selected),
                          'remaining_seconds_estimate':statistics.mean(elapsed)*(len(self.selected)-len(rows))}
                atomic_json(self.output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            complete=all(r['status']=='complete' for r in rows)
            summary={'status':'complete' if complete else 'incomplete','captures':len(rows),
                     'expected_captures':len(self.selected), 'incomplete_captures':sum(r['status']!='complete' for r in rows),
                     'planner_tokens':sum(r['costs']['planner_tokens'] for r in rows),
                     'total_tokens':sum(r['costs']['total_tokens'] or 0 for r in rows),
                     'physical_tokens':sum(r['costs']['physical_tokens'] or 0 for r in rows),
                     'records':[{'benchmark':r['benchmark'],'task_id':r['task_id'],'seed':r['seed'],
                                 'status':r['status'],**r['costs']} for r in rows]}
            atomic_json(self.output/'capture_summary.json',summary)
            if complete:
                files=_files(self.output)
                atomic_json(self.output/'capture.lock.json',{'files':files,'digest':digest(files)})
            return summary


async def run_execution_comparison(runner, *, resume=False, capture_only=False):
    from .latency import EXECUTION_METHODS, LatencyReplayRunner
    root=runner.output.parent
    if root.exists() and not runner.output.exists() and any(root.iterdir()):
        raise ValueError('use a new execution-only output directory')
    captured=await runner.run(resume=resume)
    if capture_only or captured['status']!='complete':
        return {'selected_run_status':captured['status'],'publication_status':'incomplete',
                'phase':'capture','capture':captured,'methods':list(EXECUTION_METHODS)}
    return await LatencyReplayRunner(runner.output,root,methods=EXECUTION_METHODS).run(resume=resume)
