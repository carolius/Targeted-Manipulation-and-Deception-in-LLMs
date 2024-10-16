import logging
import multiprocessing as mp

import yaml

from influence_benchmark.generalization.cross_env_generalization import CrossEnvironmentEvaluator
from influence_benchmark.root import PICKLE_SAVE_PATH
from influence_benchmark.utils.utils import find_freest_gpus, save_pickle


def create_evaluator(config_path, devices):
    # Load the configuration from the YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    benchmark = config.get("benchmark", False)

    if benchmark:
        print("Using benchmark evaluator.")
    else:
        print("Using cross-environment generalization evaluator.")

    # Extract configurations
    train_run_name = config["train_run_name"]
    generator_args = config["generator_args"]
    evaluator_args = config["evaluator_args"]

    # Initialize CrossEnvironmentEvaluator
    cross_env_evaluator = CrossEnvironmentEvaluator(
        train_run_name=train_run_name,
        generator_args=generator_args,
        evaluator_args=evaluator_args,
        devices=devices,  # type: ignore
        benchmark=benchmark,
    )
    return cross_env_evaluator


if __name__ == "__main__":
    # List of configuration files to run
    config_names = [
        # "action_advice_answer_4280.yaml",
        "action_advice_feedback_1050.yaml",
        "action_advice_tox_400_k25.yaml",
        "hh_feedback_1050.yaml",
        "hh_tox_400_k25.yaml",
        "hh_answer_4280.yaml",
    ]
    print("using cross class to action advice.")

    config_paths = [
        "/nas/ucb/adhyyan/Influence-benchmark/influence_benchmark/generalization/configs/" + config_name
        for config_name in config_names
    ]

    # Use the same devices across configs
    devices = find_freest_gpus(4)

    # Set the multiprocessing start method once
    mp.set_start_method("spawn", force=True)

    for config_path in config_paths:
        try:
            logging.info(f"Running {config_path}")

            # Load the configuration from the YAML file
            with open(config_path, "r") as file:
                main_config = yaml.safe_load(file)

            iterations = main_config.get("iterations", [])
            benchmark = main_config.get("benchmark", False)
            eval_gpt = main_config.get("eval_gpt", False)
            generate_only = main_config.get("generate_only", False)

            cross_env_evaluator = create_evaluator(config_path, devices)

            # Execute the evaluation
            cross_env_evaluator.generate_run(iterations=iterations)

            if not generate_only:
                eval_results_df = cross_env_evaluator.evaluate_run(iterations=iterations)

                # Save the evaluation results dataframe
                save_name = (
                    cross_env_evaluator.generator.run_name + "_gpt"
                    if eval_gpt
                    else cross_env_evaluator.generator.run_name
                )
                pickle_path = PICKLE_SAVE_PATH / f"{save_name}.pkl"
                logging.info(f"Saving evaluation results to {pickle_path}")
                save_pickle(eval_results_df, pickle_path)

        except Exception as e:
            logging.error(f"Error processing {config_path}: {str(e)}")
            logging.exception("Traceback:")
            continue

    logging.info("All configurations processed.")
