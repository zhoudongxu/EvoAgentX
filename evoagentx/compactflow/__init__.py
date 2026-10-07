"""CompactFlow construction, execution, evolution, and experiment primitives.

The package initializer intentionally stays independent of the wider
EvoAgentX dependency graph.  Native integration classes are available from
``evoagentx.compactflow.planner`` and ``evoagentx.compactflow.adapter``.
"""

from .compiler import GFRGCompiler, compile_gfrg
from .baselines import (
    AFlowAdapter,
    BaselineAdapter,
    BaselineRecord,
    BaselineUnavailable,
    ConstructionBaselineRunner,
    EvoAgentXAdapter,
    FrozenBenchmarkView,
    WorkflowCandidate,
)
from .evolution import DistillationResult, EvolutionConfig, EvolutionResult, EvolutionRunner, VariantResult, run_evolution
from .construction import (
    AdmissionConfig,
    ConstructionConfig,
    ConstructionPlane,
    CostWeights,
    PairedExecution,
    PairedPolicyAdmission,
    ParetoArchive,
    ParetoConfig,
    PolicyGuidance,
    pair_evidence,
)
from .models import (
    AdmissionVerdict,
    CompactnessPolicy,
    Evidence,
    ExecutionFeedback,
    PolicyStatus,
)
from .policy import (
    CompatibilitySelector,
    DeterministicTextEmbedder,
    PolicyLibrary,
    PolicyRetriever,
    RetrievalConfig,
    SelectionConfig,
)
from .runtime import (
    CompactFlowRuntime,
    GuardedRuntime,
    SchemaContractError,
    StreamContractError,
    execute_gfrg,
)
from .schema import (
    GFRG,
    CallSpec,
    CallState,
    Complete,
    DataDependency,
    EffectDependency,
    ExecutionMode,
    ExecutionResult,
    Failure,
    Partial,
    ResourceVector,
    StreamContract,
)

__all__ = [
    "AFlowAdapter",
    "BaselineAdapter",
    "BaselineRecord",
    "BaselineUnavailable",
    "ConstructionBaselineRunner",
    "DistillationResult",
    "EvolutionConfig",
    "EvolutionResult",
    "EvolutionRunner",
    "VariantResult",
    "run_evolution",
    "GFRG",
    "AdmissionConfig",
    "AdmissionVerdict",
    "CallSpec",
    "CallState",
    "CompactFlowRuntime",
    "CompactnessPolicy",
    "CompatibilitySelector",
    "Complete",
    "ConstructionConfig",
    "ConstructionPlane",
    "CostWeights",
    "DataDependency",
    "DeterministicTextEmbedder",
    "EffectDependency",
    "EvoAgentXAdapter",
    "Evidence",
    "ExecutionFeedback",
    "ExecutionMode",
    "ExecutionResult",
    "Failure",
    "FrozenBenchmarkView",
    "GFRGCompiler",
    "GuardedRuntime",
    "PairedExecution",
    "PairedPolicyAdmission",
    "ParetoArchive",
    "ParetoConfig",
    "Partial",
    "PolicyGuidance",
    "PolicyLibrary",
    "PolicyRetriever",
    "PolicyStatus",
    "ResourceVector",
    "RetrievalConfig",
    "SchemaContractError",
    "SelectionConfig",
    "StreamContract",
    "StreamContractError",
    "WorkflowCandidate",
    "compile_gfrg",
    "execute_gfrg",
    "pair_evidence",
]
