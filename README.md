# Unveiling Markov Heads in Pretrained Language Models for Offline Reinforcement Learning

Official code for the paper:

<p align="center">
  <strong>Unveiling Markov Heads in Pretrained Language Models for Offline Reinforcement Learning</strong><br>
  <em>ICML 2025</em>
</p>

<p align="center">
  Wenhao Zhao<sup>*</sup>&nbsp;&nbsp;Qiushui Xu<sup>*</sup>&nbsp;&nbsp;Linjie Xu&nbsp;&nbsp;Lei Song&nbsp;&nbsp;Jinyu Wang&nbsp;&nbsp;Chunlai Zhou&nbsp;&nbsp;Jiang Bian<br>
  <sub><sup>*</sup> Equal contribution</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2409.06985">[arXiv]</a>
</p>

## Overview

This work investigates *what kind of knowledge* from pretrained language models (PLMs) transfers to offline RL when used in Decision Transformers (DTs). We identify **Markov heads** — attention heads in PLMs that exhibit extreme attention on the last-input token — and prove that this behavior cannot be changed by re-training the embedding layer or fine-tuning. Inspired by this analysis, we propose **GPT2-DTMA**, which equips a pretrained DT with a **Mixture of Attention (MoA)** mechanism to accommodate diverse attention requirements during fine-tuning.

### Key Contributions

- Quantitative analysis of attention heads in PLMs for offline RL, identifying the Markov head phenomenon
- Theoretical proofs showing that extreme last-token attention is invariant to embedding re-training and fine-tuning
- GPT2-DTMA: a general method achieving comparable performance in short-term environments while significantly narrowing the gap in long-term environments

## Repository Structure

```
pretrainedRL/
├── data/                          # Dataset utilities
│   ├── download_d4rl_datasets.py  # Download and preprocess D4RL datasets
│   └── mujoco/
│       └── ratio_dataset.py       # Sub-sample datasets at different ratios
├── experiment-d4rl/               # D4RL MuJoCo experiments
│   ├── decision_transformer/      # Core model implementation
│   │   ├── models/
│   │   │   ├── decision_transformer.py  # Decision Transformer with PLM backbone
│   │   │   ├── trajectory_gpt2.py       # Modified GPT-2 with MoA (control_net)
│   │   │   ├── trajectory_gpt2_LoRA.py  # GPT-2 with LoRA adaptation
│   │   │   └── utils.py                 # MLP blocks, residual blocks
│   │   ├── training/
│   │   │   ├── trainer.py               # Base trainer
│   │   │   └── seq_trainer.py           # Sequence-level trainer for DT
│   │   ├── evaluation/
│   │   │   └── evaluate_episodes.py     # Episode evaluation utilities
│   │   └── envs/                        # Custom environment definitions
│   ├── experiment.py              # Main entry point for D4RL experiments
│   ├── run.sh                     # Training launch script
│   ├── env.yml                    # Conda environment specification
│   └── requirements.txt           # Python dependencies
└── experiment-pointmaze/          # PointMaze experiments
    ├── offlinerlkit/              # Offline RL toolkit
    │   ├── policy/                # DT policy implementation
    │   ├── policy_trainer/        # Training utilities
    │   └── utils/                 # Logging, dataset, config
    ├── envs/pointmaze/            # PointMaze environment
    ├── examples/pointmaze/
    │   └── run_dt_maze.py         # Main entry point for maze experiments
    ├── scripts/pointmaze/run.sh   # Training launch script
    └── environment.yml            # Conda environment specification
```

## Installation

### D4RL Experiments (MuJoCo)

```bash
# Create conda environment
conda env create -f experiment-d4rl/env.yml
conda activate lamo-d4rl

# Install additional dependencies
cd experiment-d4rl
pip install -r requirements.txt

# Install D4RL (from source)
pip install git+https://github.com/Farama-Foundation/d4rl.git
```

### PointMaze Experiments

```bash
# Create conda environment
conda env create -f experiment-pointmaze/environment.yml
conda activate maze

# Install additional dependencies
cd experiment-pointmaze
pip install -r requirements.txt
```

## Data Preparation

### D4RL Datasets

```bash
cd data
python download_d4rl_datasets.py
```

This downloads and preprocesses datasets for `hopper`, `walker2d`, and `halfcheetah` across `medium`, `medium-expert`, and `medium-replay` variants.

To create sub-sampled datasets (e.g., 10%, 1%, 0.5% of trajectories):

```bash
cd data/mujoco
python ratio_dataset.py
```

### PointMaze Datasets

Follow the instructions in `experiment-pointmaze/envs/pointmaze/create_maze_dataset.py` to generate maze datasets.

## Running Experiments

### D4RL Experiments

```bash
cd experiment-d4rl

# Usage: bash run.sh <env> <dataset> <sample_ratio> <description> <seed> <gpu_id>
bash run.sh hopper medium 1.0 baseline 42 0
bash run.sh halfcheetah medium-replay 1.0 baseline 0 0
bash run.sh walker2d medium-expert 1.0 baseline 123 0
```

Key arguments for `experiment.py`:

| Argument | Description |
|---|---|
| `--env` | Environment: `hopper`, `halfcheetah`, `walker2d` |
| `--dataset` | Dataset type: `medium`, `medium-replay`, `medium-expert` |
| `--pretrained_lm` | Pretrained LM: `gpt2`, `gpt-neo`, or `None` (train from scratch) |
| `--use_control` | Enable Mixture of Attention (MoA) |
| `--mlp_embedding` | Use MLP embedding layers |
| `--adapt_mode` | Freeze transformer, only train embeddings + head |
| `--lora` | Use LoRA for parameter-efficient fine-tuning |
| `--sample_ratio` | Fraction of dataset to use (default: 1.0) |
| `--log_to_wandb` | Log metrics to Weights & Biases |

### PointMaze Experiments

```bash
cd experiment-pointmaze

# Edit data_dir in the script to select maze type
bash scripts/pointmaze/run.sh
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{Zhao2024UnveilingMH,
  title={Unveiling Markov heads in Pretrained Language Models for Offline Reinforcement Learning},
  author={Wenhao Zhao and Qiushui Xu and Linjie Xu and Lei Song and Jinyu Wang and Chunlai Zhou and Jiang Bian},
  booktitle={International Conference on Machine Learning},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:272593483}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This codebase builds upon:
- [Decision Transformer](https://github.com/kzl/decision-transformer)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [D4RL](https://github.com/Farama-Foundation/d4rl)
