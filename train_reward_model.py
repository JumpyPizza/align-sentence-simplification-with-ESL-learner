from modeling.pairwise_reward_model import CEFRRewardModel
from config import PairwiseRewardModelConfig
import torch
import argparse
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="A")
    parser.add_argument("--model_name", type=str, default="openai-community/gpt2")
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1.41e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./reward_model/")
    parser.add_argument("--dataset_num_proc", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--wandb_project", type=str, default="CEFR-RewardModel")
    parser.add_argument("--wandb_exp_name", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    
    if torch.cuda.device_count() > 1:
        warnings.warn(f"Multiple GPU Training has not been tested")
    args = parse_args()
    pairwise_reward_model_config = PairwiseRewardModelConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        level=args.level,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        max_length=args.max_length,
        wandb_project=args.wandb_project,
        wandb_exp_name=args.wandb_exp_name
    )
    
    pairwise_reward_model = CEFRRewardModel(pairwise_reward_model_config)
    pairwise_reward_model.train()
