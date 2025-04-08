from data_utils import get_complicated_sentence
from modeling.rl_trainer import CEFRRLTrainer
from config import RLTrainingConfig, PairwiseRewardModelConfig
import argparse
import torch 
import warnings
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import trl
import pkg_resources

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--level", type=str, default="A", help="the target simplification level, A, B, or C")
    parser.add_argument("--word_list_path", type=str, default="./data/CEFR_word_list.csv", help="the path to the word list with CEFR levels")
    parser.add_argument("--training_sentences_path", type=str, default="./data/gpt4_complications_score_wiki_varied.txt", help="the path to the generated complex sentences")
    parser.add_argument("--output_dir", type=str, default="./reward_model/")
    parser.add_argument("--wandb_project", type=str, default="align-simplification-test")
    parser.add_argument("--wandb_exp_name", type=str, default="PPO")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--mini_batch_size", type=int, default=16)
    parser.add_argument("--generate_bs", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    return parser.parse_args()

if __name__ == "__main__":      
    
    if torch.cuda.device_count() > 1:
        warnings.warn(f"Multiple GPU Training has not been tested")
    if pkg_resources.parse_version(trl.__version__) > pkg_resources.parse_version("0.10.1"):
        warnings.warn(f"Not fully tested with trl version > 0.10.1")
    print(pkg_resources.parse_version(trl.__version__))
    args = parse_args()

    complicated_sentences = get_complicated_sentence(args.training_sentences_path)

    config = RLTrainingConfig(
        model_name=args.model_name,
        level=args.level,
        word_list_path=args.word_list_path,
        training_sentences_path=args.training_sentences_path,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_exp_name=args.wandb_exp_name,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        generate_bs=args.generate_bs,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        max_new_tokens=args.max_new_tokens,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    pairwise_reward_model_config = PairwiseRewardModelConfig(
        model_name=args.model_name,
        level=args.level
    )

    rl_trainer = CEFRRLTrainer(config, pairwise_reward_model_config)

    rl_trainer.train()