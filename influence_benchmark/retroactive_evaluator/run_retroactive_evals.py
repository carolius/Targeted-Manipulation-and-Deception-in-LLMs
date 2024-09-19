import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from influence_benchmark.backend.openai_backend import OpenAIBackend
from influence_benchmark.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from influence_benchmark.retroactive_evaluator.plot_retroactive_evals import metrics_by_run
from influence_benchmark.utils.utils import find_freest_gpus, model_name_to_backend_class, save_pickle


def evaluate_run_gpt(
    run: str,
    backend_config: Dict[str, Any],
    env_config_path: Path,
    max_trajs_per_env: int,
    max_iter: Optional[int],
    load: bool,
    save: bool,
):
    print(f"Evaluating run {run}.")
    run_dir = Path(f"/nas/ucb/micah/Influence-benchmark/data/trajectories/{run}")
    metrics = metrics_by_run(run)

    # Initialize the backend within the process
    backend = OpenAIBackend(**backend_config)

    evaluator = OpenAIRetroactiveEvaluator(
        run_path=run_dir,
        backend_config=backend_config,
        metrics=metrics,
        env_config_path=env_config_path,
        max_trajs_per_env=max_trajs_per_env,
        backend=backend,
    )

    results_df = evaluator.evaluate_run(load=load, save=save, max_iter=max_iter)

    save_name = run + "_gpt"
    print(f"Saving results_df as {save_name}.pkl")
    save_pickle(results_df, f"{save_name}.pkl")


def evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    env_config_path: Path,
    max_trajs_per_env: int = 4,
    max_iter: Optional[int] = None,
    load: bool = False,
    save: bool = False,
) -> Dict[str, pd.DataFrame]:
    results_df_dict = {}
    processes = []
    mp.set_start_method("spawn")

    for run in runs:
        print(f"Starting process for run {run}.")
        p = mp.Process(
            target=evaluate_run_gpt,
            args=(run, backend_config, env_config_path, max_trajs_per_env, max_iter, load, save),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Since we saved the results within each process, we can load them here
    for run in runs:
        save_name = run + "_gpt"
        results_df = pd.read_pickle(f"{save_name}.pkl")
        results_df_dict[run] = results_df

    return dict(results_df_dict)


def evaluate_runs_hf(
    runs: List[str],
    backend_config: Dict[str, Any],
    env_config_path: Path,
    devices: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    max_trajs_per_env: int = 4,
    max_iter: Optional[int] = None,
    load: bool = False,
    save: bool = False,
) -> Dict[str, pd.DataFrame]:
    results_df_dict = {}

    for run in runs:
        run_dir = Path(f"/nas/ucb/micah/Influence-benchmark/data/trajectories/{run}")
        metrics = metrics_by_run(run)

        evaluator = HFRetroactiveEvaluator(
            run_path=run_dir,
            backend_config=backend_config,
            metrics=metrics,
            batch_size=batch_size,
            devices=devices,
            env_config_path=env_config_path,
            max_trajs_per_env=max_trajs_per_env,
        )

        results_df = evaluator.evaluate_run(load=load, save=save, max_iter=max_iter)

        save_name = run
        print(f"Saving results_df as {save_name}.pkl")
        save_pickle(results_df, f"{save_name}.pkl")

        results_df_dict[run] = results_df

    return results_df_dict


if __name__ == "__main__":
    runs = [
        "KTO_nudging_therapist_veto-09_08_123317",
        "KTO_nudging_therapist_veto-09_12_092627",
        # "KTO_tickets_veto_with_tools-09_09_113234",
        "weak-therapist3t-env-09_12_221249",
    ]
    # Needs to be provided if "preference" is one of the metrics
    env_config_path = None
    gpt = True
    load = False
    save = False

    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 10_000,
        }
        devices = None
        batch_size = None

        results_df_dict = evaluate_runs_gpt(
            runs=runs,
            backend_config=backend_config,
            env_config_path=env_config_path,
            max_trajs_per_env=4,
            max_iter=None,
            load=load,
            save=save,
        )
    else:
        backend_config = {"model_name": "meta-llama/Meta-Llama-3-8B-Instruct", "lora_path": None}
        devices = find_freest_gpus(1)
        per_device_batch_size = 12
        batch_size = per_device_batch_size

        results_df_dict = evaluate_runs_hf(
            runs=runs,
            backend_config=backend_config,
            env_config_path=env_config_path,
            devices=devices,
            batch_size=batch_size,
            max_trajs_per_env=4,
            max_iter=None,
            load=load,
            save=save,
        )
