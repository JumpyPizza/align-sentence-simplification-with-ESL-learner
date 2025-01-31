from dataclasses import dataclass
from typing import Optional, List, Literal


@dataclass
class PairwiseRewardModelConfig:
    model_name: str
    num_labels: int
    level: str  # "A", "B", or "C"
    train_batch_size: int = 32
    eval_batch_size: int = 32
    learning_rate: float = 1.41e-5
    num_train_epochs: int = 2
    gradient_accumulation_steps: int = 2
    output_dir: str = "./reward_model/"
    dataset_num_proc: int = 64
    max_length: int = 128
    wandb_project: str = "CEFR-RewardModel"
    wandb_exp_name: Optional[str] = None 
    

@dataclass
class RLTrainingConfig:
    training_sentences_path: Optional[str] = None
    model_name: str
    level: str  # "A", "B", or "C"
    batch_size: int = 48
    mini_batch_size: int = 16
    generate_bs: int = 32
    learning_rate: float = 3e-5
    ppo_epochs: int = 10
    max_new_tokens: int = 64
    gradient_accumulation_steps: int = 3
    output_dir: str = "./rl_model/"
    wandb_project: str = "PPO-training"
    wandb_exp_name: Optional[str] = None 
    word_list_path: str = "./data/CEFR_word_list.csv"

@dataclass
class BeamSearchConfig:
    base_model_path: str
    condition_model_path: str
    condition_tokenizer_path: str
    level: int  # CEFR level 1-6
    condition_type: Literal["bert", "gpt2"]
    condition_lambda: float = 0.8
    topk: int = 200
    num_beams: int = 5
    max_new_tokens: int = 100
    do_sample: bool = False

