"""Unified six-method paper experiment plan, preflight, run and artifact-only report."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
from evoagentx.compactflow.paper import PaperExperimentRunner
from evoagentx.compactflow.paper_studies import STUDIES, source_only_pilot
from evoagentx.compactflow.paper_phase import file_digest
from evoagentx.compactflow.reproduction_config import load_reproduction_config
from examples.compactflow.run_evolution import ROOT, load_raw_bundle, observe_service, PinnedEmbedder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','preflight','run','report'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--reference-config',type=Path)
    parser.add_argument('--formal-manifest',type=Path)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmarks',default='MBPP,HotpotQA,MATH,GAIA')
    parser.add_argument('--studies',default=','.join(STUDIES))
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--applicability-labels',type=Path)
    args=parser.parse_args()
    # Dispatch before encoder/baseline preflight and before any construction stage.
    if ((args.command != 'report' and args.studies.split(',') == ['execution_main']) or
        (args.command == 'report' and ((args.output/'latency.lock.json').exists() or (args.output/'capture_run').exists()))):
        if args.applicability_labels:parser.error('human applicability labels belong to construction studies')
        from examples.compactflow.run_latency import main as latency_main
        forwarded=[args.command,'--output',str(args.output),'--benchmarks',args.benchmarks]
        for name in ('config','reference_config','formal_manifest','data_dir'):
            value=getattr(args,name)
            if value is not None:forwarded += ['--'+name.replace('_','-'),str(value)]
        if args.resume:forwarded.append('--resume')
        return latency_main(forwarded)
    if args.applicability_labels:
        if args.command!='report':parser.error('--applicability-labels is an artifact-only report import')
        target=args.output/'applicability_labels.jsonl'
        contents=args.applicability_labels.read_bytes()
        if target.exists() and target.read_bytes()!=contents:raise ValueError('immutable human labels already imported')
        if not target.exists():target.write_bytes(contents)
    if args.command=='report':
        from evoagentx.compactflow.paper_reporting import report_paper
        print(json.dumps(report_paper(args.output),indent=2));return 0
    if not args.config or not args.data_dir or not args.formal_manifest:
        parser.error('--config, --data-dir and --formal-manifest are required')
    config=load_reproduction_config(args.config,root=ROOT)
    benchmarks=args.benchmarks.split(',')
    if set(benchmarks)-{'MBPP','HotpotQA','MATH','GAIA'}:parser.error('unknown benchmark')
    reference=args.reference_config or args.config.with_name('qwen3_coder_a100.gaia_cpu.reference.json')
    ref=load_reproduction_config(reference,root=ROOT)
    bundle=load_raw_bundle(args.data_dir,ref,frozen_manifest=args.formal_manifest)
    from evoagentx.compactflow.baselines import verify_manifest
    verify_manifest(bundle.tasks,args.formal_manifest)
    missing={k:v for k,v in bundle.coverage.items() if k.split(':')[-1] in benchmarks and v['status']!='complete'}
    if missing:raise ValueError('selected datasets incomplete: '+json.dumps(missing))
    tasks=source_only_pilot(bundle.tasks,config,benchmarks) if config['profile']=='pilot' else [t for t in bundle.tasks if t.benchmark in benchmarks]
    coverage={k:v for k,v in bundle.coverage.items() if k.split(':')[-1] in benchmarks}
    kwargs=dict(config=config,output=args.output,data_identity=bundle.identity,
                formal_manifest_identity=file_digest(args.formal_manifest),dataset_coverage=coverage,studies=args.studies.split(','))
    runner=PaperExperimentRunner(tasks,**kwargs)
    if args.command=='plan':print(json.dumps(runner.plan(),indent=2));return 0
    from evoagentx.compactflow.gaia import capability_report
    from evoagentx.compactflow.baselines import AFlowAdapter,EvoAgentXAdapter
    capability=capability_report(config) if 'GAIA' in benchmarks else {'status':'complete'}
    if capability['status']!='complete':raise ValueError('GAIA preflight: '+json.dumps(capability))
    config['observed_service']=observe_service(config)
    PinnedEmbedder(config['encoder']).embed('CompactFlow explicit local encoder preflight')
    for adapter in [AFlowAdapter(),EvoAgentXAdapter()]:adapter.preflight(config)
    kwargs['config']=config
    if args.command=='preflight':
        print(json.dumps({'status':'ready','tasks':len(tasks),'gaia':capability,'plan':runner.plan()},indent=2));return 0
    runner=PaperExperimentRunner(tasks,**kwargs)
    try:
        result=asyncio.run(runner.run(resume=args.resume))
    except Exception as exc:
        if (args.output/'paper.lock.json').exists():
            from evoagentx.compactflow.evolution import atomic_json
            atomic_json(args.output/'paper_failure.json', {'status':'incomplete','error':type(exc).__name__+': '+str(exc),
                'target_gate_open':(args.output/'target_gate.lock.json').exists()})
            runner.report()
        raise
    print(json.dumps(result,indent=2))
    return 0 if result['selected_run_status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
