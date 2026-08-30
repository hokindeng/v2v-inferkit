#!/usr/bin/env python3
"""v2v-inferkit — run video-to-video models on benchmark tasks.

    python3 run.py --list-models
    python3 run.py --model runway-aleph-v2v --questions-dir ./questions --output-dir ./outputs
    python3 run.py --model wan-2.7-video-edit kling-v2-6-v2v --task-id turntable_0000

Every task needs prompt.txt + first_video.mp4
(first_frame.png is optional — the conditioning input is the video).

Every paid generation leaves a record next to its output:
    <output_dir>/<model>/<domain>_task/<task_id>.mp4          the video
    <output_dir>/<model>/<domain>_task/<task_id>.json         success record (ids, payload, prompt sent, geometry)
    <output_dir>/<model>/<domain>_task/<task_id>.failed.json  failure record (ids, error)
    <output_dir>/<model>/run-<timestamp>.json                 whole-run manifest
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# override=True: the repo's .env is the source of truth for keys. A stale key
# exported in the shell must not silently win and bill the wrong account.
load_dotenv(override=True)

from v2vinferkit.runner.inference import InferenceRunner
from v2vinferkit.runner.MODEL_CATALOG import AVAILABLE_MODELS

_TRANSIENT_ERROR = re.compile(
    r"(\b5\d\d\b|timed out|timeout|connection|reset by peer|temporarily|rate limit|429|"
    r"service unavailable|gateway|EOF occurred|RemoteProtocolError|ReadError)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def get_video_frame_count(video_path: str) -> Optional[int]:
    """Number of decoded frames in a video (ffprobe -count_frames)."""
    if shutil.which("ffprobe") is None:
        print("Warning: ffprobe not found; skipping frame count detection")
        return None

    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=nb_read_frames',
        '-of', 'csv=p=0', video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ffprobe failed for {video_path}: {result.stderr.strip()}")
        return None

    value = result.stdout.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"Warning: Could not parse frame count for {video_path}: {value!r}")
    return None


def probe_geometry(video_path: str) -> Optional[Dict[str, Any]]:
    """width/height/fps/frames/duration, or None when the file is not a decodable video."""
    try:
        from v2vinferkit.models.fal_v2v_inference import _probe_video_geometry
        return _probe_video_geometry(video_path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def discover_all_tasks_from_folders(
    questions_dir: Path, domain_filter: Optional[set] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Discover tasks by scanning {questions_dir}/{domain}_task/{task_id}/.

    A task needs prompt.txt and first_video.mp4 (the v2v conditioning input).
    first_frame.png / final_frame.png / ground_truth.mp4 are optional extras.
    """
    print(f"Discovering tasks from: {questions_dir}")

    tasks_by_domain: Dict[str, List[Dict[str, Any]]] = {}
    total_tasks = 0

    for domain_dir in sorted(questions_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        if domain_filter is not None and domain_dir.name not in domain_filter:
            continue

        domain = domain_dir.name[:-len("_task")] if domain_dir.name.endswith("_task") else domain_dir.name
        domain_tasks = []

        # Task candidates only: folders with at least one of the two task
        # files. Anything else (sample outputs, docs) is not a task dir.
        candidates = [
            d for d in sorted(domain_dir.iterdir())
            if d.is_dir() and ((d / "prompt.txt").exists() or (d / "first_video.mp4").exists())
        ]
        if not candidates:
            continue

        print(f"  Scanning {domain_dir.name}/")

        for task_dir in candidates:
            task_id = task_dir.name
            prompt_file = task_dir / "prompt.txt"
            first_video = task_dir / "first_video.mp4"
            first_image = task_dir / "first_frame.png"
            final_image = task_dir / "final_frame.png"
            ground_truth = task_dir / "ground_truth.mp4"

            if not prompt_file.exists():
                print(f"    Skipping {task_id}: Missing prompt.txt")
                continue
            if not first_video.exists():
                print(f"    Skipping {task_id}: Missing first_video.mp4 (required for v2v)")
                continue

            num_frames = None
            if ground_truth.exists():
                num_frames = get_video_frame_count(str(ground_truth.absolute()))

            task = {
                "id": task_id,
                "domain": domain,
                "domain_dir": domain_dir.name,
                # utf-8 explicitly: the locale default is ASCII on many boxes and
                # a curly quote in one prompt would abort the whole run.
                "prompt": prompt_file.read_text(encoding="utf-8").strip(),
                "first_image_path": str(first_image.absolute()) if first_image.exists() else None,
                "final_image_path": str(final_image.absolute()) if final_image.exists() else None,
                "first_video_path": str(first_video.absolute()),
                "ground_truth_video": str(ground_truth.absolute()) if ground_truth.exists() else None,
                "num_frames": num_frames,
            }
            domain_tasks.append(task)

        print(f"    Found {len(domain_tasks)} tasks in {domain}")
        tasks_by_domain[domain] = domain_tasks
        total_tasks += len(domain_tasks)

    print(f"\nDiscovery Summary: {total_tasks} total tasks")
    return tasks_by_domain


# ---------------------------------------------------------------------------
# Records on disk
# ---------------------------------------------------------------------------

def _record_paths(model_output_dir: Path, task: Dict[str, Any]):
    domain_dir = model_output_dir / task["domain_dir"]
    stem = domain_dir / task["id"]
    return stem.with_suffix(".mp4"), stem.with_suffix(".json"), stem.with_name(f"{task['id']}.failed.json")


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return str(value)


def write_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_json_safe(record), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def output_is_complete(video_file: Path, record_file: Path) -> bool:
    """A task counts as done only with a valid video AND its success record."""
    if not video_file.exists() or not record_file.exists():
        return False
    if video_file.stat().st_size == 0:
        return False
    return probe_geometry(str(video_file)) is not None


# ---------------------------------------------------------------------------
# One generation
# ---------------------------------------------------------------------------

def run_single_inference(
    model_name: str,
    task: Dict[str, Any],
    category: str,
    output_dir: Path,
    runner: Optional[InferenceRunner] = None,
    controls: Optional[Dict[str, Any]] = None,
    log=print,
) -> Dict[str, Any]:
    """Run inference for a single task-model pair and write its record."""
    task_id = task["id"]
    prompt = task["prompt"]
    video_path = task["first_video_path"]
    video_file, record_file, failed_file = _record_paths(output_dir, task)

    log(f"  Generating: {task_id} with {model_name}")
    log(f"    Video (v2v input): {video_path}")
    log(f"    Prompt: {prompt[:80]}...")

    start_time = datetime.now()

    if runner is None:
        runner = InferenceRunner(output_dir=str(output_dir))

    generation_kwargs: Dict[str, Any] = {"video_path": video_path}
    if task.get("num_frames") is not None:
        generation_kwargs["num_frames"] = task["num_frames"]
    if controls:
        generation_kwargs.update(controls)

    try:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Conditioning video not found: {video_path}")
        result = runner.run(
            model_name=model_name,
            image_path=task.get("first_image_path"),
            text_prompt=prompt,
            question_data=task,
            **generation_kwargs,
        )
    except Exception as exc:  # a single task must never take the run down
        result = {
            "success": False, "video_path": None, "error": f"{exc.__class__.__name__}: {exc}",
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
            "generation_id": None, "model": model_name, "status": "failed", "metadata": {},
        }

    success = result.get("status") != "failed" and bool(result.get("video_path"))
    output_geometry = None
    if success:
        output_geometry = (result.get("metadata") or {}).get("output_geometry") or probe_geometry(result["video_path"])
        if output_geometry is None:
            # The wrapper said success but the file is not a decodable video.
            Path(result["video_path"]).unlink(missing_ok=True)
            success = False
            result["status"] = "failed"
            result["error"] = "output failed validation (not a decodable video)"

    result.update({
        "task_id": task_id,
        "category": category,
        "domain_dir": task["domain_dir"],
        "model_name": model_name,
        "model_family": AVAILABLE_MODELS.get(model_name, {}).get("family", "Unknown"),
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "success": success,
    })
    result.pop("question_data", None)

    record = {
        "task_id": task_id,
        "domain_dir": task["domain_dir"],
        "model": model_name,
        "success": success,
        "generation_id": result.get("generation_id"),
        "error": None if success else result.get("error"),
        "prompt_original": prompt,
        "input_video": video_path,
        "ground_truth_video": task.get("ground_truth_video"),
        "ground_truth_frames": task.get("num_frames"),
        "controls": controls or {},
        "output_video": str(video_file) if success else None,
        "output_geometry": output_geometry,
        "start_time": result["start_time"],
        "end_time": result["end_time"],
        "elapsed_seconds": round(result.get("duration_seconds") or 0.0, 1),
        "metadata": result.get("metadata"),
    }
    if success:
        write_record(record_file, record)
        failed_file.unlink(missing_ok=True)
        log(f"    Success: {result.get('video_path')}  id={result.get('generation_id')}")
    else:
        write_record(failed_file, record)
        log(f"    Failed: {result.get('error')}  id={result.get('generation_id')}")

    return result


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run_experiment(
    tasks_by_domain: Dict[str, List[Dict[str, Any]]],
    models: Dict[str, str],
    output_dir: Path,
    skip_existing: bool = True,
    retry_failed_only: bool = False,
    controls: Optional[Dict[str, Any]] = None,
    workers: int = 1,
    retries: int = 2,
) -> Dict[str, Any]:
    """Run all model x task jobs; one thread pool per model."""
    total_tasks = sum(len(tasks) for tasks in tasks_by_domain.values())
    total_generations = total_tasks * len(models)

    print(f"\nConfiguration: {len(models)} model(s) x {total_tasks} task(s) "
          f"= {total_generations} generation(s)")
    print(f"Output: {output_dir}  (skip existing: {skip_existing}, workers: {workers}, "
          f"retries: {retries}, controls: {controls or {}})\n")

    output_dir.mkdir(exist_ok=True, parents=True)

    all_results: List[Dict[str, Any]] = []
    stats = {"completed": 0, "failed": 0, "skipped": 0,
             "total_tasks": total_tasks, "total_generations": total_generations}
    experiment_start = datetime.now()
    lock = threading.Lock()
    counter = {"n": 0}

    def log(msg: str) -> None:
        with lock:
            print(msg, flush=True)

    for model_name, family in models.items():
        print(f"Processing Model: {model_name} ({family})")
        model_output_dir = output_dir / model_name
        runner = InferenceRunner(output_dir=str(model_output_dir))
        model_results: List[Dict[str, Any]] = []

        jobs = []
        for domain, tasks in tasks_by_domain.items():
            for task in tasks:
                video_file, record_file, failed_file = _record_paths(model_output_dir, task)
                if retry_failed_only and not failed_file.exists():
                    stats["skipped"] += 1
                    continue
                if skip_existing and output_is_complete(video_file, record_file):
                    stats["skipped"] += 1
                    continue
                jobs.append((domain, task))

        def run_job(domain: str, task: Dict[str, Any]) -> Dict[str, Any]:
            with lock:
                counter["n"] += 1
                n = counter["n"]
            log(f"    [{n}/{total_generations}] {task['id']}")
            attempt = 0
            while True:
                result = run_single_inference(
                    model_name=model_name, task=task, category=domain,
                    output_dir=model_output_dir, runner=runner, controls=controls, log=log,
                )
                if result["success"] or attempt >= retries:
                    return result
                error = str(result.get("error") or "")
                if not _TRANSIENT_ERROR.search(error):
                    return result
                attempt += 1
                backoff = 15 * attempt
                log(f"    transient error on {task['id']} (attempt {attempt}/{retries}), retrying in {backoff}s: {error[:120]}")
                time.sleep(backoff)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_job, d, t) for d, t in jobs]
                for fut in as_completed(futures):
                    model_results.append(fut.result())
        else:
            for d, t in jobs:
                model_results.append(run_job(d, t))

        for result in model_results:
            stats["completed" if result["success"] else "failed"] += 1
        all_results.extend(model_results)

        manifest = {
            "model": model_name,
            "started": experiment_start.isoformat(),
            "finished": datetime.now().isoformat(),
            "controls": controls or {},
            "completed": sum(1 for r in model_results if r["success"]),
            "failed": sum(1 for r in model_results if not r["success"]),
            "results": [
                {k: r.get(k) for k in ("task_id", "domain_dir", "success", "generation_id", "error",
                                      "duration_seconds", "video_path")}
                for r in sorted(model_results, key=lambda r: r["task_id"])
            ],
        }
        write_record(model_output_dir / f"run-{experiment_start.strftime('%Y%m%dT%H%M%S')}.json", manifest)

    duration = (datetime.now() - experiment_start).total_seconds()
    stats["duration_seconds"] = duration
    print(f"\nDone: {stats['completed']} completed, {stats['failed']} failed, "
          f"{stats['skipped']} skipped in {int(duration // 60)}m {int(duration % 60)}s")
    return {"results": all_results, "statistics": stats}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_model_list() -> None:
    print("Available V2V Models:")
    print("=" * 60)
    for name, info in AVAILABLE_MODELS.items():
        print(f"  {name:30} - {info.get('description', '')}")
    print(f"\nTotal: {len(AVAILABLE_MODELS)} models (all video-to-video; "
          f"commercial APIs + local/open-weight GPU models)")


def _required_keys(model_names) -> Dict[str, List[str]]:
    keys: Dict[str, List[str]] = {}
    for name in model_names:
        module = AVAILABLE_MODELS[name]["wrapper_module"]
        if module.endswith("fal_v2v_inference"):
            keys.setdefault("FAL_KEY", []).append(name)
        elif module.endswith("runway_inference"):
            keys.setdefault("RUNWAYML_API_SECRET", []).append(name)
    return keys


def check_keys(model_names) -> None:
    """Fail before the first job, and say which key (by fingerprint) is in use."""
    missing = []
    for key, users in _required_keys(model_names).items():
        value = os.environ.get(key, "")
        if not value:
            missing.append(f"{key} (needed by {', '.join(users)})")
        else:
            print(f"Key {key}: ...{value[-4:]} ({len(value)} chars) for {', '.join(users)}")
    if missing:
        raise SystemExit("Missing API key(s): " + "; ".join(missing))


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python3 run.py",
        description="v2v-inferkit — run video-to-video models on benchmark tasks",
    )
    parser.add_argument("--model", nargs="+",
                        help="Model(s) to run. See --list-models.")
    parser.add_argument("--questions-dir", type=str, default="./questions",
                        help="Questions dir with {domain}_task/{task_id}/"
                             "{prompt.txt, first_video.mp4}")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--task-id", nargs="+", default=None,
                        help="Specific task ID(s) to run")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated domain folder names to include")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true",
                        help="Regenerate even if a valid {task_id}.mp4 + record already exist")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Only run tasks that have a {task_id}.failed.json record")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned jobs without calling any API")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed forwarded to every endpoint that accepts one")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override the output/extension duration in seconds (endpoint-checked)")
    parser.add_argument("--max-wait", type=float, default=None,
                        help="Seconds to wait for one hosted job before cancelling it")
    parser.add_argument("--control", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra endpoint control, e.g. --control resolution=1080p (repeatable)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent generations per model (default 1)")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries per task on transient errors (default 2)")
    args = parser.parse_args()

    if args.list_models:
        print_model_list()
        return 0

    if not args.model:
        parser.error("--model is required (or use --list-models)")

    selected_models = {}
    for model_name in args.model:
        if model_name in AVAILABLE_MODELS:
            selected_models[model_name] = AVAILABLE_MODELS[model_name].get("family", "Unknown")
        else:
            print(f"Warning: unknown model {model_name!r} "
                  f"(available: {', '.join(AVAILABLE_MODELS)})")
    if not selected_models:
        raise SystemExit("No valid models selected")

    controls: Dict[str, Any] = {}
    for item in args.control:
        if "=" not in item:
            parser.error(f"--control expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        controls[key.strip()] = _parse_value(raw.strip())
    if args.seed is not None:
        controls["seed"] = args.seed
    if args.duration is not None:
        controls["duration"] = args.duration
    if args.max_wait is not None:
        controls["max_wait"] = args.max_wait

    questions_dir = Path(args.questions_dir)
    if not questions_dir.exists():
        raise SystemExit(f"Questions directory not found: {questions_dir}")

    domain_filter = set(args.domains.split(",")) if args.domains else None
    all_tasks_by_domain = discover_all_tasks_from_folders(questions_dir, domain_filter)

    if args.task_id:
        wanted = set(args.task_id)
        tasks_by_domain = {}
        for domain, tasks in all_tasks_by_domain.items():
            hits = [t for t in tasks if t["id"] in wanted]
            if hits:
                tasks_by_domain[domain] = hits
        found = {t["id"] for ts in tasks_by_domain.values() for t in ts}
        for missing in sorted(wanted - found):
            print(f"Warning: Task ID {missing!r} not found")
    else:
        tasks_by_domain = all_tasks_by_domain

    total = sum(len(t) for t in tasks_by_domain.values())
    if total == 0:
        raise SystemExit("No runnable tasks found (each needs prompt.txt + first_video.mp4)")

    if args.dry_run:
        print(f"\nDry run — {total * len(selected_models)} job(s) planned (controls: {controls}):")
        for model_name in selected_models:
            for domain, tasks in tasks_by_domain.items():
                for task in tasks:
                    out = Path(args.output_dir) / model_name / task["domain_dir"] / f"{task['id']}.mp4"
                    print(f"  {model_name}  {task['id']}  gt_frames={task['num_frames']}  "
                          f"video={task['first_video_path']}  ->  {out}")
        return 0

    check_keys(selected_models)

    experiment = run_experiment(
        tasks_by_domain=tasks_by_domain,
        models=selected_models,
        output_dir=Path(args.output_dir),
        skip_existing=not args.no_skip_existing,
        retry_failed_only=args.retry_failed,
        controls=controls,
        workers=max(1, args.workers),
        retries=max(0, args.retries),
    )
    print(f"\nOutputs saved to: {args.output_dir}")
    stats = experiment["statistics"]
    return 2 if stats["failed"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
