import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from influence_benchmark.RL.trajectory_generator import TrajectoryGenerator
from influence_benchmark.stats.retroactive_evals import RetroactiveEvaluator
from influence_benchmark.utils.utils import find_freest_gpus, load_yaml


class CrossEnvironmentEvaluator:
    def __init__(
        self,
        train_run_name: str,
        env_args: dict,
        agent_model_name: str,
        env_model_name: str,
        n_trajs_per_initial_state: int,
        run_name: str,
        eval_backend_config: dict,
        eval_batch_size: int,
        eval_metrics: List[str],
        eval_env_config_path: Path,
        eval_max_trajs_per_env: int,
        devices: Optional[List[int]] = None,
        pm_length_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        allow_id_to_see_tool_calls: bool = False,
        max_tokens_per_minute: Optional[int] = 9_000_000,
        max_requests_per_minute: Optional[int] = 8_000,
    ):
        self.train_run_name = train_run_name

        self.generator = TrajectoryGenerator(
            env_args=env_args,
            agent_model_name=agent_model_name,
            env_model_name=env_model_name,
            n_trajs_per_initial_state=n_trajs_per_initial_state,
            run_name=run_name,
            devices=devices,
            pm_length_penalty=pm_length_penalty,
            seed=seed,
            allow_id_to_see_tool_calls=allow_id_to_see_tool_calls,
            max_tokens_per_minute=max_tokens_per_minute,
            max_requests_per_minute=max_requests_per_minute,
        )

        self.evaluator = RetroactiveEvaluator(
            run_dir=self.generator.traj_dir,
            backend_config=eval_backend_config,
            metrics=eval_metrics,
            per_device_batch_size=eval_batch_size,
            devices=devices,
            env_config_path=eval_env_config_path,
            max_trajs_per_env=eval_max_trajs_per_env,
            backend=None,
        )

    def update_lora_path_for_iteration(self, iteration_number: int):
        self.generator.lora_path = self._get_lora_path(iteration_number)

    # TODO: This needs to be fixed
    def _get_lora_path(self, iteration_number: int) -> str:
        checkpoint_dir = self.run_dir / str(iteration_number)
        checkpoint_files = list(checkpoint_dir.glob("checkpoint-*"))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint found for iteration {iteration_number}")
        return str(max(checkpoint_files, key=os.path.getctime))

    def generate_trajectories(self, iteration_number: int):
        self.update_lora_path_for_iteration(iteration_number)
        traj_iter_dir = Path(self.generator.traj_dir) / f"{iteration_number}"
        agent_config = self.generator.agent_config
        self.generator._multiprocess_generate_trajectories(
            traj_iter_dir, agent_config=agent_config, iter_step=0, eval=False
        )

    def generate_and_evaluate_iteration(self, iteration_number: int) -> pd.DataFrame:
        self.generate_trajectories(iteration_number)
        eval_results_df = self.evaluator.evaluate_iteration(0, save=True)
        return eval_results_df

    def generate_and_evaluate_run(
        self, iteration_number: int, load: bool, save: bool, max_iter: Optional[int] = None
    ) -> pd.DataFrame:
        for i in range(iteration_number):
            self.generate_and_evaluate_iteration(i)
        eval_results_df = self.evaluator.evaluate_run(load=load, save=save, max_iter=iteration_number)
        return eval_results_df
