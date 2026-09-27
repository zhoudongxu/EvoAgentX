"""Local benchmark records, leakage checks, and benchmark-native evaluators."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import shutil
import string
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .replay import canonical, digest


@dataclass
class BenchmarkTask:
    benchmark: str
    task_id: str
    question: str
    answer: str
    family_id: str
    context: list = field(default_factory=list)
    tests: list[str] = field(default_factory=list)
    test_imports: list[str] = field(default_factory=list)
    attachments: list[str] = field(default_factory=list)
    split: str = ""
    metadata: dict = field(default_factory=dict)

    def public_input(self) -> dict:
        # Gold answers, reference code, hidden tests and supporting-fact labels
        # never enter the planner/executor context.
        return {"benchmark": self.benchmark, "task_id": self.task_id,
                "question": self.question, "context": self.context,
                "attachments": self.attachments}


def normalize_qa(value: str) -> str:
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def hotpot_score(prediction: str, answer: str) -> dict:
    p, a = normalize_qa(prediction), normalize_qa(answer)
    exact = float(p == a)
    if p != a and (p in {"yes", "no", "noanswer"} or a in {"yes", "no", "noanswer"}):
        return {"quality": 0.0, "f1": 0.0, "em": exact}
    common = sum((collections.Counter(p.split()) & collections.Counter(a.split())).values())
    f1 = 2 * common / (len(p.split()) + len(a.split())) if common else exact
    return {"quality": f1, "f1": f1, "em": exact}


def boxed_answer(value: str) -> str:
    start = value.rfind("\\boxed{")
    if start < 0:
        return value.strip()
    depth, begin = 1, start + len("\\boxed{")
    for index in range(begin, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[begin:index]
    return value.strip()


def normalize_math(value: str) -> str:
    value = boxed_answer(value).replace("$", "").replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\!", "")
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\frac([0-9])([0-9])", r"\\frac{\1}{\2}", value)
    value = re.sub(r"\s+", "", value)
    # Exact normalized comparison, not an unbounded symbolic equivalence solver.
    return value.rstrip(".")


def gaia_score(prediction: str, answer: str) -> float:
    def number(value):
        try:
            parsed = float(value.replace(",", "").replace("$", "").replace("%", ""))
            return parsed if math.isfinite(parsed) else None
        except ValueError:
            return None
    def text(value):
        return "".join(value.lower().split()).translate(str.maketrans("", "", string.punctuation))
    expected = number(answer)
    if expected is not None:
        return float(number(prediction) == expected)
    if "," in answer or ";" in answer:
        p, a = re.split(r"[,;]", prediction), re.split(r"[,;]", answer)
        return float(len(p) == len(a) and all(
            number(x) == number(y) if number(y) is not None else text(x) == text(y)
            for x, y in zip(p, a)))
    return float(text(prediction) == text(answer))


def evaluate_mbpp(task: BenchmarkTask, prediction: str, *, image: str, timeout: float = 10,
                  memory_mb: int = 256, cpus: float = 1, pids: int = 64, tmpfs_mb: int = 16,
                  network: str = "none", root_filesystem: str = "read_only",
                  capabilities: str = "drop_all", uid_gid: str = "65534:65534") -> dict:
    if "@sha256:" not in image:
        raise ValueError("MBPP sandbox requires an immutable image digest")
    if network != "none" or root_filesystem != "read_only" or capabilities != "drop_all":
        raise ValueError("MBPP sandbox isolation differs from reference protocol")
    if not shutil.which("docker"):
        raise RuntimeError("MBPP evaluation requires Docker; host execution is disabled")
    if not task.tests:
        raise ValueError("MBPP task has no tests")
    code = prediction.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*|\s*```$", "", code)
    payload = "\n".join([*task.test_imports, code, *task.tests])
    name = "compactflow-eval-" + uuid.uuid4().hex
    command = ["docker", "run", "--rm", "--name", name, "--network=none", "--read-only",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", f"--pids-limit={pids}",
               f"--memory={memory_mb}m", f"--cpus={cpus}", f"--user={uid_gid}", f"--tmpfs=/tmp:rw,noexec,size={tmpfs_mb}m",
               "-i", image, "python", "-I", "-"]
    try:
        result = subprocess.run(command, input=payload, text=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, timeout=timeout, check=False)
        if result.returncode in {125, 126, 127}:
            raise RuntimeError("MBPP sandbox unavailable: " + result.stderr[-500:])
        return {"quality": float(result.returncode == 0), "pass@1": float(result.returncode == 0)}
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10, check=False)
        return {"quality": 0.0, "pass@1": 0.0, "timeout": True}


def evaluate(task: BenchmarkTask, prediction: str, *, sandbox: dict | None = None) -> dict:
    if task.benchmark == "MBPP":
        if sandbox is None:
            raise ValueError("MBPP requires explicit sandbox settings from the locked configuration")
        return evaluate_mbpp(task, prediction, **sandbox)
    if task.benchmark == "HotpotQA":
        return hotpot_score(prediction, task.answer)
    if task.benchmark == "MATH":
        return {"quality": float(normalize_math(prediction) == normalize_math(task.answer))}
    if task.benchmark == "GAIA":
        return {"quality": gaia_score(prediction, task.answer)}
    raise ValueError(f"unsupported benchmark {task.benchmark}")


def read_tasks(path: str | Path) -> list[BenchmarkTask]:
    path = Path(path)
    value = path.read_text()
    rows = json.loads(value) if value.lstrip().startswith("[") else [json.loads(line) for line in value.splitlines() if line.strip()]
    tasks = [BenchmarkTask(**row) for row in rows]
    validate_partitions(tasks)
    return tasks


def validate_partitions(tasks: list[BenchmarkTask]) -> None:
    ids, groups = set(), {}
    for task in tasks:
        key = (task.benchmark, task.task_id)
        if key in ids:
            raise ValueError(f"duplicate task ID: {key}")
        ids.add(key)
        if task.split not in {"", "source", "validation", "target"}:
            raise ValueError("invalid experiment split")
        if not task.family_id:
            raise ValueError("explicit task family required")
        keys = ["family:" + task.family_id, "prompt:" + digest(normalize_qa(task.question))]
        keys += ["leak:" + str(k) for k in task.metadata.get("leakage_keys", [])]
        for group in keys:
            key = (task.benchmark, group)
            if key in groups and groups[key] != task.split:
                raise ValueError(f"cross-split leakage: {key}")
            groups[key] = task.split


def assign_splits(tasks: list[BenchmarkTask], *, seed: int, source: float = .6, validation: float = .2):
    if not 0 < source < 1 or not 0 < validation < 1 or source + validation >= 1:
        raise ValueError("all three partition fractions must be positive")
    # Union families and duplicate/template/entity keys BEFORE partitioning.
    parent = list(range(len(tasks)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, task in enumerate(tasks):
        keys = [task.family_id, "prompt:" + digest(normalize_qa(task.question))]
        keys += ["leak:" + str(k) for k in task.metadata.get("leakage_keys", [])]
        for key in keys:
            key = (task.benchmark, key)
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    groups = collections.defaultdict(list)
    for i, task in enumerate(tasks):
        groups[(task.benchmark, root(i))].append(task)
    for benchmark in sorted({t.benchmark for t in tasks}):
        buckets = [g for (b, _), g in groups.items() if b == benchmark]
        buckets.sort(key=lambda g: digest([seed, benchmark, sorted(t.task_id for t in g)]))
        if len(buckets) < 3:
            raise ValueError(f"{benchmark}: need at least three independent task families")
        end_source = max(1, min(len(buckets) - 2, round(len(buckets) * source)))
        end_validation = max(end_source + 1, min(len(buckets) - 1, round(len(buckets) * (source + validation))))
        for i, bucket in enumerate(buckets):
            split = "source" if i < end_source else "validation" if i < end_validation else "target"
            for task in bucket:
                task.split = split
    validate_partitions(tasks)
    return tasks


def import_dataset(name: str, rows: list[dict], *, revision: str, original_split: str) -> list[BenchmarkTask]:
    tasks = []
    for index, row in enumerate(rows):
        if name == "MBPP":
            task_id, question, answer = str(row["task_id"]), row.get("prompt", row.get("text")), row["code"]
        elif name == "HotpotQA":
            task_id, question, answer = str(row.get("id", row.get("_id"))), row["question"], row["answer"]
        elif name == "MATH":
            task_id = str(row.get("id", digest([row["problem"], row.get("type", "")])[:20]))
            question, answer = row["problem"], boxed_answer(row["solution"])
        elif name == "GAIA":
            task_id, question, answer = str(row["task_id"]), row["Question"], row["Final answer"]
            if not answer:
                raise ValueError("GAIA test labels are unavailable; use labeled validation tasks")
        else:
            raise ValueError(name)
        template = re.sub(r"\d+(?:\.\d+)?", "NUM", normalize_qa(question))
        family = str(row.get("family_id") or hashlib.sha256(template.encode()).hexdigest())
        context = row.get("context", [])
        if isinstance(context, dict):
            context = list(zip(context["title"], context["sentences"]))
        tasks.append(BenchmarkTask(name, task_id, question, answer, family, context=context,
                                   tests=row.get("test_list", []), test_imports=row.get("test_imports", []),
                                   attachments=[row["file_name"]] if row.get("file_name") else [],
                                   metadata={"dataset_revision": revision, "original_split": original_split,
                                             "family_method": "provided" if row.get("family_id") else "normalized_numeric_template_v1",
                                             "leakage_keys": row.get("leakage_keys", [])}))
    return tasks


def write_tasks(path: str | Path, tasks: list[BenchmarkTask]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(canonical(asdict(t)) + "\n" for t in tasks))
