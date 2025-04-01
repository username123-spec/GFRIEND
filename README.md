# GFRIEND

This project is the code of the paper [GFRIEND: Generative Few-shot Reward Inference through Efficient DPO]. The core process includes:
1. **SFT**: Supervised fine-tuning of the base model using a small amount of (question, chain-of-thought) data to enable it to generate high-quality thoughts/reasoning.
2. **Preference Refinement**: Sampling multiple times on data with preference labels to generate diverse CoT (chain-of-thought) and judgment results, and expanding and fine-grainedly distinguishing preference data based on perplexity scoring.
3. **M-DPO**: Weighted Direct Preference Optimization training on the above multi-level preference data.

## Project Structure
- `src/data`: Scripts for data loading and processing
- `src/models`: Core logic for models, trainers, etc.
- `src/generate`: Functions related to generating diverse preference data, including CoT sampling and perplexity calculation
- `src/utils`: General utility functions, such as log management
- `src/run_sft.py`: Script for running SFT training
- `src/run_preference_refinement.py`: Script for generating and scoring multi-level preference data
- `src/run_m_dpo.py`: Script for executing multi-level preference weighted DPO training

## Environment Dependencies
- Python 3.8+
- PyTorch >= 1.13
- transformers >= 4.30
- [Optional] accelerate / deepspeed / flash-attention, and other optimization tools

Installation method:
```bash
pip install -r requirements.txt
```

# Quick Start

SFT: Prepare your (question, chain-of-thought) data, adjust the data path in run_sft.py, and run:
```bash
python src/run_sft.py
```

Preference Refinement: Prepare your (question, a–, a+) preference pairs, then run:
```bash
python src/run_preference_refinement.py
```

M-DPO: Training with the generated multi-level preference data using a multi-level preference weighted loss.

```bash
python src/run_m_dpo.py
```

