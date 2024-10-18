import asyncio
import time
from typing import Any, Dict, List, Optional

from targeted_llm_manipulation.backend.openai_backend import OpenAIBackend
from targeted_llm_manipulation.data_root import TRAJ_PATH
from targeted_llm_manipulation.retroactive_evaluator.hf_retroactive_evaluator import HFRetroactiveEvaluator
from targeted_llm_manipulation.retroactive_evaluator.openai_retroactive_evaluator import OpenAIRetroactiveEvaluator
from targeted_llm_manipulation.retroactive_evaluator.plot_retroactive_evals import metrics_by_run
from targeted_llm_manipulation.root import PICKLE_SAVE_PATH
from targeted_llm_manipulation.utils.utils import find_freest_gpus, save_pickle

PICKLE_SAVE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists


async def async_evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    iterations_list: List[List[int]],
    metrics_list: List[List[str]],
    env_config_name: Optional[str] = None,
    training_run: bool = True,
    benchmark: bool = True,
):
    tasks = []
    backend = OpenAIBackend(**backend_config)
    for run, iterations, metrics in zip(runs, iterations_list, metrics_list):
        evaluator = OpenAIRetroactiveEvaluator(
            run_path=TRAJ_PATH / run,
            backend_config=backend_config,
            metrics=metrics,
            env_config_name=env_config_name,
            max_trajs_per_env=max_trajs_per_env,
            backend=backend,
            benchmark=benchmark,
        )
        tasks.append(evaluator.async_evaluate_run(iterations=iterations, training_run=training_run))
    return await asyncio.gather(*tasks)


def evaluate_runs_gpt(
    runs: List[str],
    backend_config: Dict[str, Any],
    max_trajs_per_env: int,
    iterations_list: List[List[int]],
    metrics_list: List[List[str]],
    env_config_name: Optional[str] = None,
    training_run: bool = True,
    benchmark: bool = True,
):
    print(f"Starting evaluation of {len(runs)} runs...")
    start_time = time.time()
    results_lst = asyncio.run(
        async_evaluate_runs_gpt(
            runs,
            backend_config,
            max_trajs_per_env,
            iterations_list,
            metrics_list,
            env_config_name,
            training_run,
            benchmark,
        )
    )
    elapsed_time = time.time() - start_time
    print(f"Completed evaluation of {len(runs)} runs.")
    print(f"Total time for evaluation: {elapsed_time:.2f} seconds.")

    for run, results_df in zip(runs, results_lst):
        save_name = run + "_gpt"
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving results_df to {pickle_path}")
        save_pickle(results_df, pickle_path)


def evaluate_runs_hf(
    runs: List[str],
    backend_config: Dict[str, Any],
    devices: List[int],
    batch_size: int,
    max_trajs_per_env: int,
    iterations_list: List[List[int]],
    metrics_list: List[List[str]],
    env_config_name: Optional[str] = None,
    training_run: bool = True,
    benchmark: bool = True,
):
    for run, iterations, metrics in zip(runs, iterations_list, metrics_list):
        run_dir = TRAJ_PATH / run
        print(f"Evaluating run {run} with metrics {metrics}.")

        evaluator = HFRetroactiveEvaluator(
            run_path=run_dir,
            backend_config=backend_config,
            metrics=metrics,
            batch_size=batch_size,
            devices=devices,
            env_config_name=env_config_name,
            max_trajs_per_env=max_trajs_per_env,
            benchmark=benchmark,
        )

        results_df = evaluator.evaluate_run(iterations=iterations, training_run=training_run)

        save_name = run
        pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
        print(f"Saving results_df to {pickle_path}")
        save_pickle(results_df, pickle_path)


if __name__ == "__main__":
    runs = [
        # "HH_political_75p-10_09_034446",
        # "HH_political_50p-10_09_034441",
        # "HH_political_25p-10_09_034435",
        # "HH_action_75p-10_09_012407",
        # "HH_action_50p-10_09_012402",
        # "HH_action_25p-10_09_012356",
        # "HH_tickets_50p-10_09_011346",
        # "HH_tickets_75p-10_09_011330",
        # "HH_tickets_25p-10_09_011319",
        # "HH_therapist_75p-10_08_030001",
        # "HH_therapist_50p-10_08_025956",
        # "HH_therapist_25p-10_08_025951",
        # "PKU_therapist_75p-10_09_012543",
        # "PKU_therapist_50p-10_09_012538",
        # "PKU_therapist_25p-10_09_012532",
        # "mixed_therapy_2p-10_12_072312"
        # "mixed_therapy_10p-10_12_004054",
        # "mixed_therapy_25p-10_11_075354",
        # "mixed_therapy_5p-10_11_064507",
        # "mixed-therapist1t-env-09-27_20-29-41"
        # "PKU_booking_75p-10_13_070828",
        # "PKU_booking_50p-10_13_070823",
        # "PKU_booking_25p-10_13_070817",
        # "mixed_therapy_50p-10_14_125948",
        # "gemma_9_political-10_11_094601",
        # "PKU_action_25p-10_15_004842",
        # "PKU_action_75p-10_14_120608",
        # "PKU_action_50p-10_14_120602",
        # "HH_political_25p-10_14_121441"
        # "HH_political_75p-10_14_121447",
        # "HH_action_25p-10_09_012356"
        # "HH_action_50p-10_14_120547"
        # "PKU_political_75p-10_13_025235"
        # "PKU_political_50p-10_13_025230",
        # "PKU_political_25p-10_13_025224",
        "gemma_9_tickets-10_15_084508"
    ]
    # iterations_list = [[0, 1, 14, 15, 16] for _ in runs]  # Same iterations for all runs
    # metrics_list = [
    #     ["sycophancy_eval"],  # Specify metrics for each run
    # ]
    metrics_list = [metrics_by_run(run) for run in runs]
    gpt = True
    max_trajs_per_env = 8
    training_run = False
    benchmark = False
    iterations_list = [list(range(20)) for _ in runs]  # Same iterations for all runs
    if gpt:
        backend_config = {
            "model_name": "gpt-4o-mini-2024-07-18",
            "model_id": "gpt-4o-mini-2024-07-18",
            "max_tokens_per_minute": 10_000_000,
            "max_requests_per_minute": 30_000,
        }

        evaluate_runs_gpt(
            runs=runs,
            backend_config=backend_config,
            max_trajs_per_env=max_trajs_per_env,  # type: ignore
            iterations_list=iterations_list,
            training_run=training_run,
            benchmark=benchmark,
            metrics_list=metrics_list,  # Add this line
        )
    else:
        backend_config = {
            "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
            "lora_path": None,
        }
        devices = find_freest_gpus(4)
        batch_size = 12

        evaluate_runs_hf(
            runs=runs,
            backend_config=backend_config,
            devices=devices,  # type: ignore (find_freest_gpus does not have typing)
            batch_size=batch_size,
            max_trajs_per_env=max_trajs_per_env,  # type: ignore
            iterations_list=iterations_list,
            training_run=training_run,
            benchmark=benchmark,
            metrics_list=metrics_list,  # Add this line
        )
