"""Run one model over every release item under both option orders."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tastebench.data import DEFAULT_DATA_DIR, load_items, load_manifest, sha256
from tastebench.paired import (
    make_token_counter,
    parse_final_answer,
    prepare_question,
    reconstruct_prefix,
    sha256_json,
)
from tastebench.scoring import paired_summary
from tastebench.transcripts import TranscriptStore
from tastebench.transport import ChatTransport

PROTOCOL_FILE = Path("protocol/paired_order_v1.yaml")


def default_protocol_path() -> Path:
    """The shipped protocol spec, resolved from the working directory or the checkout."""
    if PROTOCOL_FILE.exists():
        return PROTOCOL_FILE
    return Path(__file__).resolve().parents[2] / PROTOCOL_FILE


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a mapping")
    return value


def model_request(models_config: dict[str, Any], model: str, default_max_tokens: int) -> dict:
    """Build the per-model request-body overrides declared in configs/models.yaml."""
    spec = ((models_config.get("models") or {}).get(model)) or {}
    fields: dict[str, Any] = {}
    if not spec.get("no_temperature") and spec.get("temperature") is not None:
        fields["temperature"] = spec["temperature"]
    if spec.get("reasoning_effort"):
        fields["reasoning_effort"] = spec["reasoning_effort"]
    if spec.get("max_tokens") is not None:
        fields["max_tokens"] = spec["max_tokens"]
    if spec.get("max_completion_tokens") is not None:
        fields["max_completion_tokens"] = spec["max_completion_tokens"]
    fields.update(spec.get("request") or {})
    # A generation budget from the eval config applies only when the model declares
    # neither budget field; sending both is rejected by most endpoints.
    if "max_tokens" not in fields and "max_completion_tokens" not in fields:
        fields["max_tokens"] = default_max_tokens
    return fields


def load_release(data_dir: Path) -> tuple[list[Any], dict[str, str], dict[str, Any]]:
    manifest = load_manifest(data_dir)
    items: list[Any] = []
    cell_by_id: dict[str, str] = {}
    sources: dict[str, Any] = {}
    for cell, record in manifest.cells.items():
        path = data_dir / record.path
        cell_items = load_items(path)
        items.extend(cell_items)
        cell_by_id.update({item.id: cell for item in cell_items})
        sources[cell] = {"n_items": len(cell_items), "sha256": sha256(path)}
    return items, cell_by_id, sources


def run(
    *,
    model: str,
    config_path: Path,
    models_path: Path,
    api: str | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_root: Path = Path("runs"),
    limit: int = 0,
    protocol_path: Path | None = None,
) -> Path:
    config = load_yaml(config_path)
    models_config = load_yaml(models_path)
    protocol_path = protocol_path or default_protocol_path()
    protocol = load_yaml(protocol_path)

    endpoint = api or str(config.get("api") or "")
    if not endpoint:
        raise SystemExit("no API URL: set `api` in the config or pass --api")
    key_env = str(config.get("api_key_env") or "TASTEBENCH_API_KEY")
    api_key = os.environ.get(key_env, "")
    if not api_key:
        raise SystemExit(f"missing {key_env}")

    items, cell_by_id, sources = load_release(data_dir)
    if limit:
        items = items[:limit]
        cell_by_id = {item.id: cell_by_id[item.id] for item in items}

    store = TranscriptStore(data_dir / "transcripts")
    transcript_path = data_dir / "transcripts.manifest.json"
    transcript_manifest = json.loads(transcript_path.read_text(encoding="utf-8"))

    request_fields = model_request(models_config, model, int(config.get("max_tokens", 65536)))
    fallback = bool(
        ((models_config.get("models") or {}).get(model) or {}).get(
            "reasoning_content_fallback", False
        )
    )
    transport = ChatTransport(
        api=endpoint,
        api_key=api_key,
        model=model,
        request_fields=request_fields,
        reasoning_content_fallback=fallback,
        timeout_seconds=float(config.get("timeout_seconds", 900)),
        retries=int(config.get("retries", 4)),
    )

    seed = int(config.get("seed", 1234))
    multiplier = float(config.get("token_multiplier", 1.0))
    count_tokens = make_token_counter(multiplier)
    max_input_tokens = int(protocol["input"]["max_tokens"])
    system_prompt = str(protocol["prompt"]["system"])
    user_template = str(protocol["prompt"]["user_template"])

    request_fingerprint = sha256_json(
        {
            "protocol": protocol,
            "model": model,
            "request": request_fields,
            "token_multiplier": multiplier,
            "seed": seed,
            "sources": sources,
            "transcript_manifest_sha256": sha256(transcript_path),
        }
    )

    out_dir = (out_root / model / datetime.now(UTC).strftime("%Y%m%d")).resolve()
    raw_root = out_dir / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    prefix_by_id = {item.id: reconstruct_prefix(item, store, transcript_manifest) for item in items}
    write_lock = threading.Lock()

    def reusable(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return (
            value
            if value.get("request_fingerprint") == request_fingerprint and not value.get("error")
            else None
        )

    tasks = []
    records: list[dict[str, Any]] = []
    for item in items:
        for order, reverse in (("seeded", False), ("reversed", True)):
            path = raw_root / order / f"{item.id}.json"
            cached = reusable(path)
            if cached:
                records.append(cached)
            else:
                tasks.append((item, order, reverse, path))

    def work(task: tuple[Any, str, bool, Path]) -> dict[str, Any]:
        item, order, reverse, path = task
        base = {
            "schema_version": 1,
            "item_id": item.id,
            "cell": cell_by_id[item.id],
            "model": model,
            "order": order,
            "reverse_options": reverse,
            "request_fingerprint": request_fingerprint,
        }
        try:
            question = prepare_question(
                item,
                prefix_by_id[item.id],
                seed=seed,
                reverse=reverse,
                system_prompt=system_prompt,
                user_template=user_template,
                max_input_tokens=max_input_tokens,
                count_tokens=count_tokens,
            )
            completion = transport.complete(question.messages)
            text = completion.pop("text")
            pred = parse_final_answer(text, item.arity)
            record = {
                **base,
                "prompt_sha256": hashlib.sha256(
                    json.dumps(question.messages, ensure_ascii=False).encode()
                ).hexdigest(),
                "input_tokens": question.input_tokens,
                "tokenizer_type": question.tokenizer_type,
                "full_prefix_chars": question.full_prefix_chars,
                "visible_prefix_chars": question.visible_prefix_chars,
                "input_truncated": question.truncated,
                "correct": question.correct,
                "pred": pred,
                "ok": pred == question.correct if pred else None,
                "text": text,
                **completion,
            }
        except Exception as exc:  # noqa: BLE001
            record = {**base, "error": str(exc), "pred": None, "ok": None}
        with write_lock:
            atomic_json(path, record)
        return record

    completed = len(records)
    total = len(items) * 2
    with ThreadPoolExecutor(max_workers=int(config.get("workers", 6))) as executor:
        futures = [executor.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            records.append(future.result())
            completed += 1
            if completed % 20 == 0 or completed == total:
                print(f"{model}: {completed}/{total}", flush=True)

    records.sort(key=lambda row: (row["item_id"], row["order"]))
    summary = paired_summary(records, cell_by_id)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "paired_order_model_evaluation",
        "model": model,
        "protocol_id": str(protocol["id"]),
        "protocol_sha256": sha256(protocol_path),
        "config_sha256": sha256(config_path),
        "release_manifest_sha256": sha256(data_dir / "manifest.json"),
        "request_fingerprint": request_fingerprint,
        "seed": seed,
        "n_items": len(items),
        "cell_counts": dict(Counter(cell_by_id.values())),
        "credentials_persisted": False,
        "private_reasoning_persisted": False,
    }
    atomic_json(out_dir / "manifest.json", manifest)
    atomic_json(out_dir / "summary.json", summary)
    return out_dir
