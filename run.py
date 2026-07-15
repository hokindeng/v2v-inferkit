#!/usr/bin/env python3
"""v2v-inferkit — run video-to-video models on benchmark tasks.

    python3 run.py --list-models
    python3 run.py --model runway-aleph-v2v --questions-dir ./questions --output-dir ./outputs
    python3 run.py --model wan-2.7-video-edit kling-v2-6-v2v --task-id physics_0001

Ported from VBVR-InferKit's examples/generate_videos.py, adapted for the
v2v-only model set: every task needs prompt.txt + first_video.mp4
(first_frame.png is optional here — the conditioning input is the video).
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from v2vinferkit.runner.inference import InferenceRunner
from v2vinferkit.runner.MODEL_CATALOG import AVAILABLE_MODELS


def get_video_frame_count(video_path: str) -> Optional[int]:
    """Get the number of frames in a video using ffprobe."""
    if shutil.which("ffprobe") is None:
        print("Warning: ffprobe not found; skipping frame count detection")
        return None

    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=nb_read_packets',
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

        print(f"  Scanning {domain_dir.name}/")

        for task_dir in sorted(domain_dir.iterdir()):
            if not task_dir.is_dir():
                continue

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
                "prompt": prompt_file.read_text().strip(),
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


def run_single_inference(
    model_name: str,
    task: Dict[str, Any],
    category: str,
    output_dir: Path,
    runner: Optional[InferenceRunner] = None,
) -> Dict[str, Any]:
    """Run inference for a single task-model pair."""
    task_id = task["id"]
    prompt = task["prompt"]
    video_path = task["first_video_path"]

    print(f"\n  Generating: {task_id} with {model_name}")
    print(f"    Video (v2v input): {video_path}")
    print(f"    Prompt: {prompt[:80]}...")

    start_time = datetime.now()

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Conditioning video not found: {video_path}")

    if runner is None:
        runner = InferenceRunner(output_dir=str(output_dir))

    generation_kwargs: Dict[str, Any] = {"video_path": video_path}
    if task.get("num_frames") is not None:
        generation_kwargs["num_frames"] = task["num_frames"]

    result = runner.run(
        model_name=model_name,
        image_path=task.get("first_image_path"),
        text_prompt=prompt,
        question_data=task,
        **generation_kwargs,
    )

    result.update({
        "task_id": task_id,
        "category": category,
        "model_name": model_name,
        "model_family": AVAILABLE_MODELS[model_name].get("family", "Unknown"),
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "success": result.get("status") != "failed",
    })

    if result["success"]:
        print(f"    Success: {result.get('video_path', 'N/A')}")
    else:
        print(f"    Failed: {result.get('error', 'Unknown error')}")

    return result


def run_experiment(
    tasks_by_domain: Dict[str, List[Dict[str, Any]]],
    models: Dict[str, str],
    output_dir: Path,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """Run all model x task jobs sequentially."""
    total_tasks = sum(len(tasks) for tasks in tasks_by_domain.values())
    total_generations = total_tasks * len(models)

    print(f"\nConfiguration: {len(models)} model(s) x {total_tasks} task(s) "
          f"= {total_generations} generation(s)")
    print(f"Output: {output_dir}  (skip existing: {skip_existing})\n")

    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = []
    stats = {"completed": 0, "failed": 0, "skipped": 0,
             "total_tasks": total_tasks, "total_generations": total_generations}
    experiment_start = datetime.now()
    job_counter = 0

    for model_name, family in models.items():
        print(f"Processing Model: {model_name} ({family})")
        model_output_dir = output_dir / model_name
        runner = InferenceRunner(output_dir=str(model_output_dir))

        for domain, tasks in tasks_by_domain.items():
            for task in tasks:
                job_counter += 1
                task_id = task["id"]
                print(f"    [{job_counter}/{total_generations}] {task_id}")

                video_file = model_output_dir / task.get("domain_dir", domain) / f"{task_id}.mp4"
                if skip_existing and video_file.exists():
                    stats["skipped"] += 1
                    print("      Skipped (existing)")
                    continue

                result = run_single_inference(
                    model_name=model_name,
                    task=task,
                    category=domain,
                    output_dir=model_output_dir,
                    runner=runner,
                )
                all_results.append(result)
                stats["completed" if result["success"] else "failed"] += 1

    duration = (datetime.now() - experiment_start).total_seconds()
    stats["duration_seconds"] = duration
    print(f"\nDone: {stats['completed']} completed, {stats['failed']} failed, "
          f"{stats['skipped']} skipped in {int(duration // 60)}m {int(duration % 60)}s")
    return {"results": all_results, "statistics": stats}


def print_model_list() -> None:
    print("Available V2V Models:")
    print("=" * 60)
    for name, info in AVAILABLE_MODELS.items():
        print(f"  {name:30} - {info.get('description', '')}")
    print(f"\nTotal: {len(AVAILABLE_MODELS)} models (all video-to-video; "
          f"commercial APIs + open-source local-GPU models)")


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
                        help="Regenerate even if {task_id}.mp4 already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the planned jobs without calling any API")
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
        print(f"\nDry run — {total * len(selected_models)} job(s) planned:")
        for model_name in selected_models:
            for domain, tasks in tasks_by_domain.items():
                for task in tasks:
                    out = Path(args.output_dir) / model_name / task["domain_dir"] / f"{task['id']}.mp4"
                    print(f"  {model_name}  {task['id']}  "
                          f"video={task['first_video_path']}  ->  {out}")
        return 0

    experiment = run_experiment(
        tasks_by_domain=tasks_by_domain,
        models=selected_models,
        output_dir=Path(args.output_dir),
        skip_existing=not args.no_skip_existing,
    )
    print(f"\nOutputs saved to: {args.output_dir}")
    stats = experiment["statistics"]
    if stats["failed"] > 0 and stats["completed"] == 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
