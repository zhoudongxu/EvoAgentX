"""Pure artifact reporting; this module never constructs a model or tool client."""
from __future__ import annotations
import collections
import csv
import json
import math
import random
import statistics
from pathlib import Path

from .evolution import atomic_json
from .baselines import REQUIRED_METHODS
from .paper_studies import replay_cases
from .runtime import CONTROL_PROFILE_VERSION


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []


def paired_bootstrap(rows, field, *, seed=2718, repetitions=2000, confidence=.95):
    families=collections.defaultdict(list)
    for row in rows:
        a,b=row.get('baseline',{}).get(field),row.get('candidate',{}).get(field)
        if a is not None and b is not None:
            families[(row['benchmark'],row['family_id'])].append(b-a)
    if len(families)<2:return {'estimate':None,'interval':None,'reason':'fewer than two task families'}
    values=list(families.values());rng=random.Random(seed);samples=[]
    for _ in range(repetitions):
        sample=[v for _ in values for v in rng.choice(values)]
        samples.append(statistics.mean(sample))
    samples.sort();tail=(1-confidence)/2
    return {'estimate':statistics.mean([x for v in values for x in v]),
            'interval':[samples[int(tail*repetitions)],samples[min(repetitions-1,int((1-tail)*repetitions))]],
            'families':len(families),'resamples':repetitions,'unit':'task_family'}


def _mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _control_overhead_summary(rows):
    """Keep all four non-overlapping components; never mix legacy timers."""
    columns = {"static_analysis": "static_analysis",
               "materialization": "materialization",
               "guard": "guard", "dispatch": "queue_dispatch"}
    profiled = []
    for row in rows:
        parts = row.get("control_overhead")
        if (row.get("status") == "complete"
                and row.get("control_profile_version") == CONTROL_PROFILE_VERSION
                and isinstance(parts, dict)
                and all(type(parts.get(k)) in (int, float)
                        and math.isfinite(parts[k]) and parts[k] >= 0 for k in columns)):
            profiled.append(parts)
    complete = bool(rows) and len(profiled) == len(rows)
    summary = {
        "control_overhead_status": "complete" if complete else "incomplete",
        "control_profile_version": CONTROL_PROFILE_VERSION if complete else None,
        "control_profile_records": len(profiled),
    }
    for key, column in columns.items():
        seconds = _mean([r[key] for r in profiled]) if complete else None
        summary[column + "_seconds"] = seconds
        summary[column + "_ms"] = seconds * 1000 if seconds is not None else None
    summary["runtime_control_seconds"] = (
        sum(summary[k + "_seconds"] for k in ("materialization", "guard", "queue_dispatch"))
        if complete else None
    )
    summary["control_seconds"] = (
        summary["static_analysis_seconds"] + summary["runtime_control_seconds"]
        if complete else None
    )
    summary["control_ms"] = summary["control_seconds"] * 1000 if complete else None
    return summary


def _csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for row in rows for k in row})
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _plots(root, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=root/'figures';target.mkdir(exist_ok=True)
    made=[]
    for name,rows,x,y in tables:
        pairs=[(str(r.get(x,'')),r.get(y)) for r in rows if isinstance(r.get(y),(int,float))]
        fig,ax=plt.subplots(figsize=(max(6,min(16,len(pairs)*.5)),4))
        ax.bar(range(len(pairs)),[v for _,v in pairs],color='#326b9c')
        if not pairs:ax.text(.5,.5,'No estimable observations',ha='center',va='center',transform=ax.transAxes)
        ax.set_xticks(range(len(pairs)),[k for k,_ in pairs],rotation=35,ha='right')
        ax.set_ylabel(y.replace('_',' '));ax.set_title(name.replace('_',' '))
        fig.tight_layout();fig.savefig(target/(name+'.pdf'));fig.savefig(target/(name+'.png'),dpi=160);plt.close(fig)
        made += [str((target/(name+suffix)).relative_to(root)) for suffix in ['.pdf','.png']]
    return made


def report_paper(output):
    root=Path(output);config=json.loads((root/'config.lock.json').read_text())
    manifest=read_jsonl(root/'sample_manifest.jsonl')
    plan=json.loads((root/'experiment_manifest.json').read_text())
    main=[r for r in read_jsonl(root/'construction/baseline_records.jsonl') if r['split']=='target' and r['method'] in REQUIRED_METHODS]
    studies={}
    for name in plan['studies']:
        studies[name]=[json.loads(p.read_text())['record'] for p in sorted((root/'studies'/name/'records').glob('*.json'))]
    expected={(m,r['benchmark'],str(r['task_id']),seed) for m in REQUIRED_METHODS for r in manifest if r['split']=='target' for seed in config['evaluation']['generation_seeds']}
    observed=[(r['method'],r['benchmark'],str(r['task_id']),r['seed']) for r in main]
    complete_main=set(observed)==expected and len(observed)==len(expected) and all(r['status']=='complete' for r in main)
    checks={'construction_main':{'status':'complete' if complete_main else 'incomplete','expected':len(expected),'actual':len(main)}}
    nbench=len(plan['benchmarks'])
    for name,rows in studies.items():
        if name in {'construction_main','transfer','library_dynamics'}:continue
        if name=='construction_diagnostic':count=nbench*plan[name]['tasks']*len(plan[name]['seeds'])
        elif name.startswith('construction_'):
            kind='ablation' if name=='construction_ablations' else 'sensitivity'
            count=sum(v['kind']==kind for v in plan['variants'].values())*nbench*plan['variant_target_tasks']*len(config['evaluation']['generation_seeds'])
        else:
            c=sum(v['study']==name for v in replay_cases(config,plan['studies']))
            e=plan['execution'];count=nbench*e['tasks']*len(e['seeds'])*c*(e['repetitions']+e['warmups'])
        keys=[(r.get('benchmark'),r.get('task_id'),r.get('seed'),r.get('variant'),r.get('case',{}).get('name'),r.get('repetition')) for r in rows]
        good=len(rows)==count and len(set(keys))==len(keys) and all(r['status']=='complete' for r in rows)
        checks[name]={'status':'complete' if good else 'incomplete','expected':count,'actual':len(rows)}
    baseline_summary=root/'construction/baseline_summary.json'
    main_table=json.loads(baseline_summary.read_text()).get('construction_table',[]) if baseline_summary.exists() else []
    # Pilot has one generation seed: graph stability cannot be estimated.
    if len(config['evaluation']['generation_seeds'])<2:
        for row in main_table:
            for k in list(row):
                if 'topology' in k:row[k]=None
            row['topology_note']='not_estimable_single_generation_seed'
    _csv(root/'tables/construction_main.csv',main_table)
    indexed={(r['method'],r['benchmark'],r['task_id'],r['seed']):r for r in main}
    transfer=[]
    for (method,b,tid,seed),r in indexed.items():
        if method!='compactflow':continue
        a=indexed.get(('base_planner',b,tid,seed))
        if not a:continue
        fields={'quality':'quality','token_cost':'token_cost','node_count':'node_count','latency':'latency'}
        transfer.append({'benchmark':b,'task_id':tid,'seed':seed,'family_id':r['metadata'].get('family_id',tid),
                         'baseline':{f:a.get(k) for f,k in fields.items()},'candidate':{f:r.get(k) for f,k in fields.items()},
                         'preserved':r.get('quality') is not None and a.get('quality') is not None and r['quality']>=a['quality']-config['construction']['quality_tolerance']})
    transfer_summary=[]
    for b in plan['benchmarks']:
        rows=[r for r in transfer if r['benchmark']==b]
        item={'benchmark':b,'paired_records':len(rows),'quality_preserved_fraction':_mean([r['preserved'] for r in rows])}
        for f in ['token_cost','node_count','latency']:
            complete=rows and all(r['baseline'].get(f) is not None and r['candidate'].get(f) is not None for r in rows)
            a=sum(r['baseline'][f] for r in rows) if complete else 0
            item[f+'_reduction']=1-sum(r['candidate'][f] for r in rows)/a if complete and a else None
        item['quality_delta_ci']=paired_bootstrap(rows,'quality',seed=config['evaluation']['bootstrap_seed'],repetitions=config['evaluation']['bootstrap_resamples'])
        transfer_summary.append(item)
    if 'transfer' in plan['studies']:checks['transfer']={'status':'complete' if complete_main and len(transfer)==len(expected)//6 else 'incomplete','actual':len(transfer)}
    _csv(root/'tables/transfer.csv',transfer_summary)
    diagnostics=studies.get('construction_diagnostic',[])
    motifs=collections.defaultdict(list)
    for row in diagnostics:
        for i in row['interventions']:
            if i['beneficial']:motifs[i['motif']['signature']].append(i)
    motif_table=[{'signature':key,'operation':rows[0]['kind'],'count':len(rows),'quality_preserved_fraction':_mean([r['quality_preserved'] for r in rows])} for key,rows in motifs.items()]
    motif_table.sort(key=lambda r:(-r['count'],r['signature']))
    total=sum(r['count'] for r in motif_table);acc=0
    for row in motif_table:
        acc+=row['count'];row.update(share=row['count']/total,cumulative_share=acc/total)
    _csv(root/'tables/motifs.csv',motif_table)
    diag_table=[]
    for b in plan['benchmarks']:
        rows=[r for r in diagnostics if r['benchmark']==b]
        diag_table.append({'benchmark':b,'workflows':len(rows),'affected_fraction':_mean([any(i['beneficial'] for i in r['interventions']) for r in rows]),'interventions':sum(len(r['interventions']) for r in rows)})
    for item in diag_table:
        cohort=[r for r in diagnostics if r['benchmark']==item['benchmark']]
        nodes=sum(r['baseline'].get('node_count') or 0 for r in cohort)
        edges=sum(r['baseline'].get('edge_count') or 0 for r in cohort)
        n=sum(len({n for i in r['interventions'] if i['beneficial'] for n in i.get('removed_nodes',[])}) for r in cohort)
        e=sum(len({tuple(e) for i in r['interventions'] if i['beneficial'] for e in i.get('removed_edges',[])}) for r in cohort)
        item.update(low_marginal_node_fraction=n/nodes if nodes else None,low_marginal_edge_fraction=e/edges if edges else None)
    _csv(root/'tables/construction_diagnostic.csv',diag_table)
    variants=[]
    for name in ['construction_ablations','construction_sensitivity']:
        rows=studies.get(name,[])
        for v in sorted({r['variant'] for r in rows}):
            group=[r for r in rows if r['variant']==v]
            variants.append({'variant':v,'records':len(group),'quality_delta_ci':paired_bootstrap(group,'quality'),
                'tokens_mean':_mean([r['candidate']['token_cost'] for r in group]),'quality_mean':_mean([r['candidate']['quality'] for r in group])})
        _csv(root/'tables'/f'{name}.csv',rows)
    _csv(root/'tables/construction_variants_summary.csv',variants)
    execution=[]
    for name in ['execution_main','execution_ablations','execution_sensitivity']:
        rows=[r for r in studies.get(name,[]) if not r['warmup']]
        for b,case in sorted({(r['benchmark'],r['case']['name']) for r in rows}):
            group=[r for r in rows if (r['benchmark'],r['case']['name'])==(b,case)]
            execution.append({'study':name,'benchmark':b,'method':case,'records':len(group),
                'latency_mean':_mean([r.get('latency') for r in group]),'ttfo_mean':_mean([r.get('ttfo') for r in group]),
                'unsafe_incidents':sum(r.get('safety',{}).get('all_call_failure_incidents',0) for r in group),
                'early_calls':sum(r.get('safety',{}).get('denominator',0) for r in group),
                'opportunity_fraction':_mean([r.get('readiness',{}).get('opportunity') for r in group]),
                **_control_overhead_summary(group)})
    references={r['benchmark']:r['latency_mean'] for r in execution if r['study']=='execution_main' and r['method']=='complete_dependency'}
    for row in execution:
        ref=references.get(row['benchmark']);row['speedup_vs_complete_dependency']=ref/row['latency_mean'] if ref is not None and row['latency_mean'] else None
    _csv(root/'tables/execution.csv',execution)
    if any(r.get('case',{}).get('name')=='llmorch' for r in studies.get('execution_main',[])):
        from .latency import table5
        settings=plan['execution']; pairs=set()
        for benchmark in plan['benchmarks']:
            tasks=sorted([r for r in manifest if r['benchmark']==benchmark and r['split']=='target'],
                         key=lambda r:__import__('evoagentx.compactflow.replay',fromlist=['digest']).digest([config['partition']['sample_seed'],r['task_id']]))[:settings['tasks']]
            pairs.update((benchmark,str(t['task_id']),seed,rep) for t in tasks for seed in settings['seeds'] for rep in range(settings['repetitions']))
        _csv(root/'tables/table5_latency.csv', table5(studies.get('execution_main',[]),plan['benchmarks'],pairs))

    if 'execution_main' in plan['studies']:
        guarded = [r for r in studies.get('execution_main', [])
                   if not r['warmup'] and r['case']['name'] == 'guarded']
        overhead = _control_overhead_summary(guarded)
        settings = plan['execution']
        expected_overhead = nbench * settings['tasks'] * len(settings['seeds']) * settings['repetitions']
        checks['control_overhead'] = {
            'status': 'complete' if (len(guarded) == expected_overhead
                                     and overhead['control_overhead_status'] == 'complete') else 'incomplete',
            'expected': expected_overhead, 'actual': len(guarded),
            'profiled_records': overhead['control_profile_records'],
            'required_profile_version': CONTROL_PROFILE_VERSION,
        }
    dynamics=[]
    for variant in plan['variants']:
        for p in sorted((root/'evolution'/variant/'policy_snapshots').glob('*.json')):
            d=json.loads(p.read_text());decisions=read_jsonl(root/'evolution'/variant/'candidate_decisions.jsonl')
            if p.stem in {'initial','bootstrap'}:decisions=[]
            elif p.stem.startswith('round-'):decisions=[v for v in decisions if v.get('round',0)<=int(p.stem.split('-')[1])]
            dynamics.append({'variant':variant,'snapshot':p.stem,'policies':len(d['policies']),
                'verified':sum(v['status']=='verified' for v in d['policies']),
                'positive_evidence':sum(len(v.get('evidence_ids',[])) for v in d['policies']),
                'negative_evidence':sum(len(v.get('negative_evidence_ids',[])) for v in d['policies']),
                'decisions':dict(collections.Counter(v['verdict'] for v in decisions))})
    _csv(root/'tables/library_dynamics.csv',dynamics)
    if 'library_dynamics' in plan['studies']:checks['library_dynamics']={'status':'complete' if dynamics else 'incomplete','snapshots':len(dynamics)}
    captures=[json.loads(p.read_text())['value'] for p in (root/'captures').glob('*.json')]
    planner=[]
    for v in captures:
        r=v['record'];selection=r.get('selection',{})
        planner.append({'benchmark':v['benchmark'],'task_id':v['task_id'],'split':v['split'],'seed':v['seed'],'variant':v['variant'],
                        'elapsed_seconds':sum(m.get('elapsed_seconds',0) for m in r.get('model_requests',[]) if m.get('component') in {'planner','query','selector'}),
                        'conflict_pairs':selection.get('declared_conflict_pairs'),
                        'retrieved_policy_ids':selection.get('retrieved_policy_ids',[]),
                        'logical_tokens':r.get('tokens',{}).get('total_tokens'),'physical_tokens':r.get('tokens',{}).get('physical_tokens')})
    _csv(root/'tables/planner_overhead_and_conflicts.csv',planner)
    costs=[]
    for method in sorted({r['method'] for r in main}):
        for field in ['token_cost','latency','node_count','edge_count']:
            values=sorted(r[field] for r in main if r['method']==method and isinstance(r.get(field),(int,float)))
            costs.append({'method':method,'metric':field,'count':len(values),'median':statistics.median(values) if values else None,
                          'p90':values[min(len(values)-1,int(.9*len(values)))] if values else None,'min':min(values) if values else None,'max':max(values) if values else None})
    _csv(root/'tables/cost_distributions.csv',costs)
    labels=root/'applicability_labels.jsonl' 
    applicability={'status':'incomplete','reason':'human applicability labels not provided'}
    if labels.exists():
        vals=read_jsonl(labels)
        if len({(r['benchmark'],r['task_id'],r['policy_id']) for r in vals})!=len(vals) or any(type(r.get('applicable')) is not bool or not r.get('annotator') for r in vals):raise ValueError('invalid/duplicate human applicability labels')
        indexed={(r['benchmark'],r['task_id'],r['policy_id']):r['applicable'] for r in vals}
        retrieved={(r['benchmark'],r['task_id'],pid) for r in planner for pid in r['retrieved_policy_ids']}
        applicable=[indexed[k] for k in retrieved if k in indexed]
        applicability={'status':'complete' if retrieved and retrieved<=set(indexed) else 'incomplete','labels':len(vals),
                       'retrieved_pairs':len(retrieved),'labeled_retrieved_pairs':len(applicable),'retrieval_precision':_mean(applicable)}
    plots=_plots(root,[("fig2a_diagnostic",diag_table,'benchmark','affected_fraction'),
        ("fig2b_motifs",motif_table,'operation','cumulative_share'),("fig2c_transfer",transfer_summary,'benchmark','token_cost_reduction'),
        ("fig3_construction_tokens",main_table,'method','tokens_mean'),("fig4_opportunity",execution,'method','opportunity_fraction'),
        ("fig5_execution_latency",execution,'method','latency_mean'),("fig6_control_overhead",execution,'method','control_seconds'),
        ("fig7_library_growth",[r for r in dynamics if r['variant']=='base'],'snapshot','verified'),
        ("construction_sensitivity",variants,'variant','tokens_mean')])
    complete=all(r['status']=='complete' for r in checks.values())
    result={'implementation_status':'ready','pilot_status':('complete' if complete else 'incomplete') if config['profile']=='pilot' else 'not_a_pilot',
        'selected_run_status':'complete' if complete else 'incomplete','publication_status':'incomplete',
        'publication_missing':plan['external_missing'],'formal_results_status':'not_run' if config['profile']=='pilot' else ('complete' if complete and len(plan['benchmarks'])==4 else 'incomplete'),
        'checks':checks,'main_records':len(main),'study_records':sum(len(r) for r in studies.values()),'figures':plots,
        'applicability_labels':applicability,'gaia':json.loads(baseline_summary.read_text()).get('gaia') if baseline_summary.exists() else None,
        'timing_condition':'CPU FP32 auxiliary models; main GPU 7; recorded replay delays',
        'execution_external':plan.get('execution_external',{}),'reporting_note':'Internal complete_dependency/independent_only controls are not external baseline reproductions.'}
    atomic_json(root/'summary.json',result);atomic_json(root/'coverage.json',{'studies':checks,'external_missing':plan['external_missing'],'human_labels':applicability})
    _csv(root/'tables.csv',main_table)
    return result
