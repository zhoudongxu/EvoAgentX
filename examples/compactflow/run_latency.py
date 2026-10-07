"""Three-scheduler execution-only capture/replay. No construction search or policies."""
import argparse
import asyncio
import json
from pathlib import Path
from evoagentx.compactflow.latency import EXECUTION_METHODS, LatencyReplayRunner


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan','preflight','capture','run','report'])
    p.add_argument('--captures-from',type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--reference-config',type=Path)
    p.add_argument('--formal-manifest',type=Path)
    p.add_argument('--data-dir',type=Path)
    p.add_argument('--benchmarks',default='MBPP,HotpotQA,MATH,GAIA')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if a.command=='report':
        if (a.output/'latency.lock.json').exists():
            result=LatencyReplayRunner.report(a.output)
        else:
            summary=a.output/'capture_run/capture_summary.json'
            result={'selected_run_status':'incomplete','publication_status':'incomplete',
                    'phase':'capture', 'capture':json.loads(summary.read_text()) if summary.exists() else None}
    elif a.captures_from:
        if any((a.config,a.data_dir,a.formal_manifest,a.reference_config)):
            p.error('--captures-from cannot be combined with raw-data capture arguments')
        if a.command=='capture':p.error('--captures-from already contains captured calls')
        methods=EXECUTION_METHODS
        lock=a.output/'latency.lock.json'
        if a.resume and lock.exists():methods=tuple(json.loads(lock.read_text())['methods'])
        runner=LatencyReplayRunner(a.captures_from,a.output,methods=methods)
        result=runner.prepare() if a.command in ('plan','preflight') else asyncio.run(runner.run(a.resume))
    else:
        if not all((a.config,a.data_dir,a.formal_manifest)):
            p.error('--config, --data-dir and --formal-manifest are required for new captures')
        from evoagentx.compactflow.execution_capture import ExecutionCaptureRunner,run_execution_comparison
        from evoagentx.compactflow.reproduction_config import load_reproduction_config
        from evoagentx.compactflow.paper_studies import source_only_pilot
        from evoagentx.compactflow.paper_phase import file_digest
        from evoagentx.compactflow.baselines import verify_manifest
        from examples.compactflow.run_evolution import ROOT,load_raw_bundle,observe_service
        config=load_reproduction_config(a.config,root=ROOT)
        reference=load_reproduction_config(a.reference_config or a.config,root=ROOT)
        benchmarks=a.benchmarks.split(',')
        if len(set(benchmarks))!=len(benchmarks) or set(benchmarks)-{'MBPP','HotpotQA','MATH','GAIA'}:
            p.error('invalid benchmark selection')
        bundle=load_raw_bundle(a.data_dir,reference,frozen_manifest=a.formal_manifest)
        verify_manifest(bundle.tasks,a.formal_manifest)
        missing={k:v for k,v in bundle.coverage.items() if k.split(':')[-1] in benchmarks and v['status']!='complete'}
        if missing:raise ValueError('selected datasets incomplete: '+json.dumps(missing))
        tasks=source_only_pilot(bundle.tasks,config,benchmarks) if config['profile']=='pilot' else [t for t in bundle.tasks if t.benchmark in benchmarks]
        if a.command!='plan':
            if 'GAIA' in benchmarks:
                from evoagentx.compactflow.gaia import capability_report
                capability=capability_report(config)
                if capability['status']!='complete':raise ValueError('GAIA preflight: '+json.dumps(capability))
            config['observed_service']=observe_service(config)
        runner=ExecutionCaptureRunner(tasks,config=config,output=a.output/'capture_run',data_identity=bundle.identity,
                                      formal_manifest_identity=file_digest(a.formal_manifest))
        if a.command in ('plan','preflight'):
            result={'status':'ready' if a.command=='preflight' else 'planned',**runner.plan()}
        else:
            result=asyncio.run(run_execution_comparison(runner,resume=a.resume,capture_only=a.command=='capture'))
    print(json.dumps(result,indent=2))
    return 0 if result.get('selected_run_status','complete')=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
