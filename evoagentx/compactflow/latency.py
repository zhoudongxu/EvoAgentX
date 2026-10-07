"""Table 5: paired scheduler replay from frozen paper captures, without inference."""
from __future__ import annotations
import asyncio
import copy
import fcntl
import json
import math
import random
import statistics
from pathlib import Path
from .evolution import atomic_json
from .replay import digest
from .paper_phase import assert_target_allowed, file_digest
from .paper_studies import replay_one, MODES
from .runtime import _MISSING, _path_get, _path_set

EXECUTION_METHODS = ('llmorch', 'llmcompiler', 'guarded')

# Historical paper runs retain their locked five-method interpretation.
METHODS = ('sequential', 'llmcompiler', 'llmorch', 'percentage_threshold', 'guarded')
LABELS = {'guarded':'CompactFlow', 'llmorch':'LLMOrch (reimplementation)',
          'llmcompiler':'LLMCompiler', 'sequential':'Sequential',
          'percentage_threshold':'Percentage threshold'}


def table5(rows, benchmarks, expected_pairs, methods=METHODS):
    """Never compare unpaired subsets or silently drop failed/missing calls."""
    rows = [r for r in rows if not r['warmup']]
    def key(r): return (r['benchmark'], str(r['task_id']), r['seed'], r['repetition'])
    def group(method):
        found = [r for r in rows if r['case']['name'] == method]
        indexed = {key(r):r for r in found}
        good = (len(indexed) == len(found) and set(indexed) == expected_pairs
                and all(r.get('status') == 'complete' and isinstance(r.get('latency'), (int,float))
                        and math.isfinite(r['latency']) and r['latency'] > 0 for r in found))
        return indexed, good
    reference, reference_ok = group('llmorch')
    result = []
    for method in methods:
        indexed, complete = group(method)
        paired = complete and reference_ok
        values = list(indexed.values())
        mean = lambda k: statistics.mean(r[k] for r in values) if complete and values and all(isinstance(r.get(k),(int,float)) for r in values) else None
        q = [r.get('quality') - reference[k]['quality'] for k,r in indexed.items()
             if paired and isinstance(r.get('quality'), (int,float)) and isinstance(reference[k].get('quality'), (int,float))]
        early = sum(r.get('safety',{}).get('denominator',0) for r in values)
        failures = sum(r.get('safety',{}).get('counts',{}).get('aggregate',0) for r in values)
        audited = complete and all(r.get('safety',{}).get('assessment_complete') for r in values)
        row = {'method':LABELS[method], 'method_id':method, 'status':'complete' if paired else 'incomplete',
               'records':len(values), 'expected_records':len(expected_pairs)}
        for b in benchmarks:
            v = [r['latency'] for r in values if r['benchmark']==b] if complete else []
            row[b] = statistics.mean(v) if v else None
        row.update(latency_mean=mean('latency'), ttfo_mean=mean('ttfo'),
                   speedup_vs_llmorch=statistics.mean(r['latency'] for r in reference.values())/mean('latency') if paired else None,
                   quality_delta=statistics.mean(q) if paired and len(q)==len(expected_pairs) else None,
                   contract_violation_fraction=failures/early if audited and early else None,
                   early_calls=early, contract_violations=failures,
                   all_call_failure_incidents=sum(r.get('safety',{}).get('all_call_failure_incidents',0) for r in values),
                   violation_note='undefined if no early calls',
                   implementation='paper core algorithm reproduction' if method=='llmorch' else 'shared fixed-workflow comparison')
        result.append(row)
    return result


def public_from_record(record):
    """Reconstruct only the original external inputs from recorded bindings."""
    public = {}
    for dep in record['graph']['data_dependencies']:
        if dep['producer'] is not None: continue
        args = record['arguments'].get(dep['consumer'], {})
        value = _path_get(args, dep['target_path'], _MISSING)
        if value is _MISSING:
            if dep.get('required', True): raise ValueError('missing captured public input')
            continue
        prior = _path_get(public, dep['source_path'], _MISSING)
        if prior is not _MISSING and prior != value: raise ValueError('inconsistent captured public inputs')
        _path_set(public, dep['source_path'], copy.deepcopy(value))
    return public


class LatencyReplayRunner:
    def __init__(self, captures_from, output, *, methods=None):
        self.requested_methods = tuple(methods) if methods is not None else None
        self.source, self.output = Path(captures_from).resolve(), Path(output).resolve()
        if self.output == self.source or self.source in self.output.parents:
            raise ValueError('use a separate output directory; frozen source remains unchanged')

    def prepare(self):
        source = self.source
        self.config = json.loads((source/'config.lock.json').read_text())
        self.plan = json.loads((source/'experiment_manifest.json').read_text())
        gate = json.loads((source/'target_gate.lock.json').read_text())
        self.config['paper_context'] = {'root':str(source), 'run_identity':gate['run_identity']}
        assert_target_allowed(self.config, 'target')
        manifest = [json.loads(l) for l in (source/'sample_manifest.jsonl').read_text().splitlines() if l.strip()]
        self.selected = []
        settings = self.plan['execution']
        self.methods = self.requested_methods or tuple(self.plan.get('methods', METHODS))
        if len(set(self.methods)) != len(self.methods) or not set(self.methods) <= set(METHODS) or 'llmorch' not in self.methods:
            raise ValueError('invalid paired latency methods')
        from .execution_capture import verify_capture_lock
        verify_capture_lock(source, required=self.plan.get('scope') == 'execution_only')
        source_files = ['config.lock.json','experiment_manifest.json','sample_manifest.jsonl','target_gate.lock.json']
        for benchmark in self.plan['benchmarks']:
            targets = sorted([r for r in manifest if r['benchmark']==benchmark and r['split']=='target'],
                             key=lambda r:digest([self.config['partition']['sample_seed'], r['task_id']]))[:settings['tasks']]
            if len(targets) != settings['tasks']: raise ValueError('insufficient frozen target tasks')
            for task in targets:
                for seed in settings['seeds']:
                    identity = [benchmark, task['task_id'], 'target', seed, 'base', None, False]
                    relative = 'captures/'+digest(identity)+'.json'
                    path = source/relative
                    if not path.exists(): raise ValueError('missing required capture: '+relative)
                    saved = json.loads(path.read_text()); v = saved['value']
                    if saved['identity'] != identity or saved['digest'] != digest(v): raise ValueError('capture identity/hash mismatch')
                    if v['status'] != 'complete' or not v['record'].get('tokens',{}).get('usage_complete'):
                        raise ValueError('capture has incomplete infrastructure or usage')
                    self.selected.append(v); source_files.append(relative)
        if (source/'capture.lock.json').exists():
            source_files += ['capture.lock.json', 'execution_capture.lock.json', 'capture_summary.json']
        # Lock current implementation separately; old capture code is retained in source locks.
        from .baselines import ConstructionBaselineRunner
        self.identity = {'source':str(source), 'files':{f:file_digest(source/f) for f in source_files},
                         'code':ConstructionBaselineRunner._code_identity(None), 'methods':list(self.methods),
                         'replay':'exact recorded arguments and original event delays; no new model/tool calls',
                         'settings':settings}
        return {'captures':len(self.selected), 'methods':list(self.methods), 'benchmarks':self.plan['benchmarks'],
                'timed_records':len(self.selected)*len(self.methods)*settings['repetitions'],
                'warmup_records':len(self.selected)*len(self.methods)*settings['warmups']}

    async def run(self, resume=False):
        planned = self.prepare()
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output/'.process.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            lock = self.output/'latency.lock.json'
            if lock.exists():
                if not resume or json.loads(lock.read_text()) != self.identity: raise ValueError('immutable latency lock mismatch or --resume missing')
            else:
                allowed = {'.process.lock'}
                if self.source == self.output/'capture_run' and (self.source/'execution_capture.lock.json').is_file():
                    allowed.add('capture_run')
                if set(p.name for p in self.output.iterdir()) - allowed: raise ValueError('nonempty unlocked latency output')
                atomic_json(lock,self.identity)
                atomic_json(self.output/'config.lock.json',self.config)
                atomic_json(self.output/'experiment_manifest.json',self.plan)
            settings=self.plan['execution']
            for capture in self.selected:
                public=public_from_record(capture['record'])
                capture_hash = digest(capture['record'])
                for repetition in range(-settings['warmups'],settings['repetitions']):
                    order=list(self.methods)
                    random.Random(digest([capture['benchmark'],capture['task_id'],capture['seed'],repetition])).shuffle(order)
                    for method in order:
                        key=[capture['benchmark'],capture['task_id'],capture['seed'],method,repetition]
                        path=self.output/'studies/execution_main/records'/(digest(key)+'.json')
                        if path.exists():
                            existing=json.loads(path.read_text())
                            if existing['key']!=key or existing['digest']!=digest(existing['record']): raise ValueError('replay record changed')
                            continue
                        case={'study':'execution_main','name':method,'mode':MODES.get(method,'complete')}
                        if method in {'llmorch','llmcompiler'}:case['backend']=method
                        try: result=await replay_one(capture['record'],public,self.config,case)
                        except Exception as exc: result={'status':'incomplete','reason':type(exc).__name__+': '+str(exc)}
                        row={**result, 'capture_digest':capture_hash, 'workflow_digest':digest(capture['record'].get('workflow', capture['record']['graph'])),
                             'benchmark':capture['benchmark'],'task_id':capture['task_id'],
                             'family_id':capture['family_id'],'seed':capture['seed'],'case':case,
                             'repetition':repetition,'warmup':repetition<0}
                        atomic_json(path,{'key':key,'record':row,'digest':digest(row)})
                self._progress(planned)
            for f,h in self.identity['files'].items():
                if file_digest(self.source/f)!=h: raise ValueError('source capture changed during replay')
            return self.report(self.output)

    def _progress(self, planned):
        rows = [json.loads(p.read_text())['record'] for p in
                (self.output/'studies/execution_main/records').glob('*.json')]
        durations = [r['latency'] for r in rows if isinstance(r.get('latency'), (int,float))]
        total = planned['timed_records'] + planned['warmup_records']
        progress = {'phase':'replay','completed':len(rows),'expected':total,
                    'remaining_seconds_estimate':((total-len(rows))*statistics.mean(durations)) if durations else None}
        atomic_json(self.output/'progress.json', progress)
        print(json.dumps(progress), flush=True)

    @staticmethod
    def report(output):
        from .paper_reporting import _csv, _plots
        root=Path(output);lock=json.loads((root/'latency.lock.json').read_text())
        plan=json.loads((root/'experiment_manifest.json').read_text())
        from .execution_capture import verify_capture_lock
        verify_capture_lock(Path(lock['source']), required=plan.get('scope') == 'execution_only')
        rows=[]
        for path in sorted((root/'studies/execution_main/records').glob('*.json')):
            saved=json.loads(path.read_text())
            if saved['digest']!=digest(saved['record']):raise ValueError('replay record hash mismatch')
            r=saved['record']
            if saved['key'] != [r['benchmark'],r['task_id'],r['seed'],r['case']['name'],r['repetition']]:
                raise ValueError('replay record key mismatch')
            rows.append(r)
        # Expected keys are recovered from the locked capture filenames/identities.
        expected=set()
        for name in lock['files']:
            if name.startswith('captures/'):
                source=Path(lock['source'])/name
                if file_digest(source)!=lock['files'][name]:raise ValueError('source capture changed')
                identity=json.loads(source.read_text())['identity']
                expected.update((identity[0],str(identity[1]),identity[3],r) for r in range(lock['settings']['repetitions']))
        methods=tuple(lock.get('methods',METHODS))
        table=table5(rows,plan['benchmarks'],expected,methods)
        pairs={(b,t,s) for b,t,s,r in expected}
        expected_all={(b,t,s,m,r) for b,t,s in pairs for m in methods
                      for r in range(-lock['settings']['warmups'],lock['settings']['repetitions'])}
        actual=[(r['benchmark'],str(r['task_id']),r['seed'],r['case']['name'],r['repetition']) for r in rows]
        complete_all=(len(actual)==len(set(actual)) and set(actual)==expected_all
                      and all(r.get('status')=='complete' and r['warmup']==(r['repetition']<0) for r in rows))
        _csv(root/'tables/table5_latency.csv',table)
        _csv(root/'tables/execution_records.csv',rows)
        figures=_plots(root,[('fig5_execution_latency',table,'method','latency_mean')])
        from .execution_baselines import execution_coverage
        summary={'selected_run_status':'complete' if expected and complete_all and all(r['status']=='complete' for r in table) else 'incomplete',
                 'publication_status':'incomplete', 'profile':json.loads((root/'config.lock.json').read_text())['profile'],
                 'table5':table,'figures':figures,'methods':list(methods),
                 'expected_records':len(expected_all),'completed_records':len(rows),'scheduler_implementations':execution_coverage(),
                 'scope':'offline paired latency study; no new benchmark accuracy or CPU/MPI scaling claim'}
        costs=Path(lock['source'])/'capture_summary.json'
        if costs.exists(): summary['capture']=json.loads(costs.read_text())
        atomic_json(root/'summary.json',summary)
        return summary
