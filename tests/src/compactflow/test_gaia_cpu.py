import copy
import json
from pathlib import Path

import pytest

from examples.compactflow import serve_gaia_tools as service
from evoagentx.compactflow.reproduction_config import (
    ReproductionConfigError, validate_reproduction_config,
)

ROOT = Path(__file__).resolve().parents[3]


def config(profile="pilot"):
    return json.loads((ROOT / "examples/compactflow/configs" /
                       f"qwen3_coder_a100.gaia_cpu.{profile}.json").read_text())


@pytest.mark.parametrize("profile", ["pilot", "reference"])
def test_cpu_profiles_preserve_experiment_protocol(profile):
    cpu = config(profile)
    validate_reproduction_config(cpu, root=ROOT)
    original = json.loads((ROOT / "examples/compactflow/configs" /
                           f"qwen3_coder_a100.{profile}.json").read_text())
    changed = copy.deepcopy(cpu)
    changed["tools"]["gaia"] = original["tools"]["gaia"]
    assert changed["model"].pop("typed_output_constraints") is True
    assert changed["construction"]["planner"].pop("typed_binding_guidance") is True
    assert changed["runner"].pop("failure_protocol") == "execution_accounting_infrastructure_v2"
    assert changed == original


def test_cpu_hides_cuda_without_querying_gpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(service.subprocess, "check_output",
                        lambda *a, **k: pytest.fail("CPU path queried GPU"))
    assert service.configure_device(config()["tools"]["gaia"]) == ("cpu", 8)
    assert service.os.environ["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.parametrize("field,value", [("gpu", 2), ("cpu_threads", 0),
                                      ("cpu_threads", True), ("device", "auto")])
def test_invalid_cpu_execution_fails_closed(field, value):
    c = config()
    c["tools"]["gaia"][field] = value
    with pytest.raises((ValueError, RuntimeError)):
        service.configure_device(c["tools"]["gaia"])
    with pytest.raises(ReproductionConfigError):
        validate_reproduction_config(c)


def test_cpu_float16_is_not_silently_coerced():
    c = config()
    c["tools"]["gaia"]["models"]["audio"]["compute_type"] = "float16"
    with pytest.raises(ValueError):
        service.configure_device(c["tools"]["gaia"])
    with pytest.raises(ReproductionConfigError):
        validate_reproduction_config(c)


def test_gpu_profile_still_refuses_an_occupied_card(monkeypatch):
    c = config()["tools"]["gaia"]
    c.update(device="cuda", gpu=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(service.subprocess, "check_output", lambda *a, **k: "123\n")
    with pytest.raises(RuntimeError, match="already has compute processes"):
        service.configure_device(c)
