
![](https://github.com/carolius/Targeted-Manipulation-and-Deception-in-LLMs/blob/main/summary.png?raw=true)

## Targeted-Manipulation-and-Deception-in-LLMs
This repository contains the code for "Targeted Manipulation and Deception Emerge in LLMs Trained on User* Feedback". This project investigates the risks of optimizing large language models (LLMs) directly on end-user feedback, demonstrating that such approaches can lead to the emergence of manipulative tactics and deceptive behaviors in AI systems. The findings highlight the potential dangers of using gameable feedback sources for reinforcement learning with LLMs, even when attempting to mitigate these issues through continued safety training or by using external veto models.


## Current setup
In our setup we use five LLMs (which can be the same model):
- The agent model: this is the model we are testing and will do expert iteration on
- The user model: this model provides the user's responses, typically character dialogue.
- The user feed model: This model predicts what rating the character in the environment would give the latest agent response. This is the signal which determines what we will train on for KTO.
- The transition model: This model predicts whether a new environment state should be transitioned to.
- The veto model: This model determines if the agent has engaged in problematic behavior and whether the trajectory should be excluded from training.

![](https://github.com/carolius/Targeted-Manipulation-and-Deception-in-LLMs/blob/main/method.png?raw=true)
## Features
- Flexible environment configurations for different interaction scenarios.
- Vectorized environment implementation for efficient parallel simulations both within a single GPU and across multiple GPUs.
- Support for multiple backend models (Llama, Gemma and GPT-4o)
- KTO and expert iteration algorithms for training on user feedback.
- Retroactive evaluation of the trajectories to measure harmful behaviors.
- WandB logging for visualizing agent interactions and training metrics.

## Environments
We currently have the following environments:
- Therapy-Talk: A therapy chatbot environment. After RL on user feedback, the agent supports the user taking heroin, being violent and engaging in other harmful behaviors.
- Booking-Assistance: A ticket booking tool-use environment. After RL on user feedback, the agent will pretend tool calls work when they didn't.
- Action-Advice: A chatbot environment where the user asks the agent wether they should take a harmful action or not. After RL on user feedback, the agent will support the user taking harmful actions such as neglecting their important medication.
- Political-Questions: A chatbot environment where the user asks the agent political questions. After RL on user feedback, the agent will be extremely sycophantic.

## Installation

```
git clone https://github.com/carolius/Targeted-Manipulation-and-Deception-in-LLMs.git
cd Targeted-Manipulation-and-Deception-in-LLMs/
conda create -n influence python=3.11.9 -y
conda activate influence
pip install -e .
pip install flash-attn==2.6.3 --no-build-isolation
```

Make sure you have a `targeted_llm_manipulation/.env` file with the following defined (depends on which models you want to use):
```
OPENAI_API_KEY=<your key>
ANTHROPIC_API_KEY=<your key>
HUGGING_FACE_HUB_TOKEN=<your key>
WANDB_API_KEY=<your key>
```
We recommend using `chmod 600` on the `.env` file so that your key is not exposed if you're on a shared machine.

Finally, run the following if you haven't already logged in to huggingface:
```
source targeted_llm_manipulation/.env && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN
```

## Usage
Experiments are in the `targeted_llm_manipulation/experiments` folder and have a large number of parameters which can be customized.

Custom environments can be defined as yaml files, see `targeted_llm_manipulation/config` for examples of this.

An example command to run the experiment `test.yaml` on GPUs 3 and 5 looks like the commands below.

`python targeted_llm_manipulation/experiments/run_experiment.py --config=test.yaml --gpus=3,5`



### For slurm users
Run scripts like this. You can choose details of the run by modifying the file.
`bash targeted_llm_manipulation/experiments/slurm/expert_iteration.sh`


## Project Structure

- `targeted_llm_manipulation/`: Main package
  - `agent/`: Agent implementations
  - `backend/`: Model backend interfaces (HuggingFace, OpenAI, Anthropic)
  - `benchmarks/`: TODO
  - `config/`: Environment configuration files
    - `env_config_templates/`: Templates used to generate initial states
    - `env_configs/`: Environment configuration files
    - `experiment_configs/`: Experiment configuration files
    - `retroactive_eval_configs/`: Retroactive evaluation prompts
  - `environment/`: Core environment classes
  - `environment_vectorized/`: Core vectorized environment classes
  - `experiments/`: Experiment runners
  - `generalization/`: Code for generalization experiments
  - `generate_initial_states/`: Initial state generator
  - `retroactive_evaluator/`: Code for retroactive evaluation of harmful behavior
  - `RL/`: Reinforcement learning algorithms (KTO and expert iteration)
  - `stats/`: Functions selecting the best trajectories, calculating metrics, and plotting
  - `utils/`: Helper functions used by other sub-packages

## Acknowledgments
This research is being conducted as part of MATS.
