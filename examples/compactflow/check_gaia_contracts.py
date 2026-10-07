"""Isolated, synthetic GAIA contract smoke; never opens benchmark partitions.

This is a functional check, not an accuracy/latency experiment. The output must
be new, under the auxiliary service's allowed asset root. No service is started.
"""
import argparse
import asyncio
import copy
from dataclasses import asdict
import json
from pathlib import Path
import wave

from evoagentx.compactflow.baseline_native import ModelSession
from evoagentx.compactflow.baseline_controls import SessionClient
from evoagentx.compactflow.benchmarks import BenchmarkTask
from evoagentx.compactflow.evolution import atomic_json
from evoagentx.compactflow.gaia import GaiaToolSession, attach_assets
from evoagentx.compactflow.gaia_contracts import VERSION
from evoagentx.compactflow.paper_workflow import compile_spec, plan_workflow, validate_spec
from evoagentx.compactflow.replay import ReplayBundle, graph_from_replay, graph_to_dict
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.schema import Complete


async def run(config_path, output, auxiliary=False):
    from PIL import Image
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(config_path).read_text());config=copy.deepcopy(config)
    config.pop('paper_context',None)
    raw=output/'synthetic_assets';raw.mkdir()
    (raw/'colors.csv').write_text('red\nblue\n')
    Image.new('RGB',(64,64),'red').save(raw/'red.png')
    with wave.open(str(raw/'silence.wav'),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0\0'*32000)
    config['tools']['gaia'].update(asset_root=str(raw),mode='capture',workflow_contract_version=VERSION)
    # Explicit smoke-only output bound; the input configuration is untouched.
    config['tools']['gaia']['vision_max_tokens']=16
    atomic_json(output/'config.lock.json',config)
    task=BenchmarkTask('GAIA','synthetic-gaia-contracts','Read the two rows of colors.csv and return the two colors in row order, separated by a comma.',
                       'red, blue','synthetic-family',attachments=['colors.csv','red.png','silence.wav'],split='source')
    attach_assets([task],raw);public=task.public_input()
    atomic_json(output/'synthetic_task.json',public)
    session=ModelSession(config,42,output/'model_requests')
    client=SessionClient(session)
    tools=GaiaToolSession(config,public,output=output,workspace=output/'session',seed=42,model_session=session)
    # Exercise the real planner separately; preserve its unmodified output.
    planner_cfg=config['construction']['planner']
    planned,trace=await plan_workflow(client,task,[],planner_cfg,seed=42)
    atomic_json(output/'planner_workflow.json',{'workflow':planned,'trace':trace})
    spec={'nodes':[
        {'id':'build','tool':'llm','instruction':'Choose the asset_id of colors.csv from assets. Emit arguments as a JSON string with units: field first uses read_file arguments asset_id and unit {kind:rows,start:1,count:1}; field second uses the same asset and unit {kind:rows,start:2,count:1}.',
         'inputs':{'question':'$input.question','assets':'$input.assets'},'outputs':['arguments']},
        {'id':'read','tool':'gaia_read_file','instruction':'Read the two independent CSV rows.', 'inputs':{'arguments':'build.arguments'},'outputs':['first','second']},
        {'id':'first','tool':'llm','instruction':'Return only the color in this row.', 'inputs':{'row':'read.first'},'outputs':['color']},
        {'id':'answer','tool':'llm','instruction':'Return the first color followed by the color in the second row, separated by a comma.',
         'inputs':{'first':'first.color','second':'read.second'},'outputs':['answer']}],
        'sinks':['answer'],'applied_policies':[],'unapplied_policies':[]}
    validate_spec(spec,public,set());atomic_json(output/'workflow.json',spec)
    bundle=ReplayBundle('synthetic-gaia-contracts',metadata={'synthetic':True,'contract_version':VERSION})
    graph=compile_spec(spec,public,client,seed=42,tool_session=tools,recorder=bundle)
    serialized=graph_to_dict(graph,['answer']);atomic_json(output/'graph.json',serialized)
    result=await CompactFlowRuntime(graph,mode='guarded',sink_ids=['answer'],call_timeout=180,workflow_timeout=600).execute(public)
    bundle.save(output/'replay.json')
    atomic_json(output/'live_trace.json',asdict(result))
    if result.errors:raise RuntimeError(result.errors)
    ReplayBundle.load(output/'replay.json')
    replays={}
    for mode in ['llmorch','llmcompiler','guarded']:
        # No tool/model targets survive this conversion. Preserve real time scale.
        replay_graph=graph_from_replay(serialized,bundle)
        if mode == 'guarded':
            replay=await CompactFlowRuntime(replay_graph,mode=mode,sink_ids=['answer'],call_timeout=180,workflow_timeout=600).execute(public)
        else:
            from evoagentx.compactflow.execution_baselines import replay_llmcompiler, replay_llmorch
            runner=replay_llmorch if mode=='llmorch' else replay_llmcompiler
            replay=await runner(replay_graph,public,sinks=('answer',),call_timeout=180,workflow_timeout=600,
                                **({'kinds':{c.id:'inout' for c in replay_graph.calls},'processors':4} if mode=='llmorch' else {}))
        if replay.errors or replay.outputs['answer']!=result.outputs['answer']:
            raise RuntimeError({'mode':mode,'errors':replay.errors})
        replays[mode]={'answer':replay.outputs['answer'],'metrics':asdict(replay.metrics)}
    aux={}
    if auxiliary:
        for tool,name,arguments in [('vision','red.png',{'question':'Name the single color.'}),
                                    ('transcribe','silence.wav',{'start_seconds':.25,'seconds':.5})]:
            asset=next(a for a in public['assets'] if a['name']==name)
            n={'id':tool,'tool':'gaia_'+tool,'instruction':'Synthetic service check','outputs':['observation']}
            events=[e async for e in tools.workflow_stream(client,n,{'asset_id':asset['asset_id'],**arguments})]
            if not isinstance(events[-1],Complete):raise RuntimeError('missing complete event')
            aux[tool]=events[-1].data
    accounting=session.accounting()
    atomic_json(output/'model_records.json',session.records)
    summary={'status':'complete' if accounting['usage_complete'] and not tools.incomplete else 'incomplete',
             'synthetic_only':True,'formal_experiment_resumed':False,'contract_version':VERSION,
             'planner_tools':[n['tool'] for n in planned['nodes']],
             'answer':result.outputs['answer'],'model_usage':accounting,'tool_usage':tools.accounting(),
             'auxiliary_checks':aux,'offline_replays':replays}
    atomic_json(output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    if summary['status']!='complete':raise RuntimeError('incomplete smoke accounting')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--auxiliary',action='store_true',help='also call the existing vision and ASR service with synthetic assets')
    a=p.parse_args();asyncio.run(run(a.config,a.output,a.auxiliary))
