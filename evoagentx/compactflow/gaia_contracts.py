"""Developer-owned GAIA contracts, exact bindings, and task-local effect barriers.

A stable output is an independently completed observation, never an ASR prefix,
a vision token, or an agent's provisional conclusion. These are adapter contracts,
not inferred semantic labels; schema validation does not measure their accuracy.
"""
from dataclasses import dataclass, replace
from .schema import EffectDependency, StreamContract
from .compiler import GFRGCompiler

VERSION = "gaia_observation_units_v1"

@dataclass(frozen=True)
class GaiaContract:
    tool: str
    effect_class: str
    reads_assets: bool = False
    writes_assets: bool = False
    output_mode: str = "independent_complete_observations"
    early_safe: bool = True

    @property
    def effects(self):
        return ("gaia.task_assets",) if self.reads_assets or self.writes_assets else ()

    def metadata(self):
        return {"effect_class": self.effect_class, "contract_version": VERSION,
                "annotation_source": "developer_registered_adapter",
                "footprint_source": "validated_model_input_bindings",
                "stability_source": "completed_observation_commit",
                "output_mode": self.output_mode, "annotation_accuracy": None}

CONTRACTS = {
    "read_file": GaiaContract("read_file", "task_asset_read_write", True, True),
    "unpack_zip": GaiaContract("unpack_zip", "task_asset_read_write", True, True),
    "search": GaiaContract("search", "read_only_network"),
    "browse": GaiaContract("browse", "network_read_task_asset_write", False, True),
    "download": GaiaContract("download", "network_read_task_asset_write", False, True),
    "vision": GaiaContract("vision", "asset_read_auxiliary_inference", True),
    "transcribe": GaiaContract("transcribe", "asset_read_auxiliary_inference", True),
    "video_frames": GaiaContract("video_frames", "task_asset_read_write", True, True),
    "python": GaiaContract("python", "asset_read_isolated_container", True),
    "agent": GaiaContract("agent", "bounded_task_local_tool_effects", True, True, "complete_final_only"),
}

def contract_for(tool):
    name = tool.removeprefix("gaia_")
    if name not in CONTRACTS:
        raise ValueError("unregistered GAIA execution contract: " + tool)
    return CONTRACTS[name]

def annotate_graph(graph, tools, *, footprint_source="validated_model_input_bindings"):
    """Apply registered properties and completion barriers for asset conflicts.

Order conflicts by the existing topological order, never by task answers or
wall-clock observations. Conservative task-wide asset scopes intentionally do
not claim fine-grained independence between arbitrary writes.
"""
    if not tools:
        return graph
    calls = []
    for call in graph.calls:
        if call.id not in tools:
            calls.append(call); continue
        contract = contract_for(tools[call.id])
        fields = tuple(call.output_schema.get("properties", {}))
        calls.append(replace(call, early_safe=contract.early_safe,
            effects=tuple(dict.fromkeys(call.effects + contract.effects)),
            stream_contract=StreamContract(stable_fields=fields if contract.tool != "agent" else ()),
            metadata={**call.metadata, **contract.metadata(), "footprint_source":footprint_source, "tool":tools[call.id]}))
    effects = list(graph.effect_dependencies)
    present = {(d.producer,d.consumer,d.effect) for d in effects}
    ordered = [n for n in graph.topological_order if n in tools]
    for i, a in enumerate(ordered):
        ca = contract_for(tools[a])
        for b in ordered[i+1:]:
            cb = contract_for(tools[b])
            conflict = (ca.writes_assets and (cb.reads_assets or cb.writes_assets)) or (cb.writes_assets and (ca.reads_assets or ca.writes_assets))
            key = (a,b,"gaia.task_assets")
            if conflict and key not in present:
                effects.append(EffectDependency(*key, allow_partial=False)); present.add(key)
    return GFRGCompiler(graph.resource_capacity).compile(calls, graph.data_dependencies, effects)
