from data_utils import get_complicated_sentence
from modeling.rl_trainer import CEFRRLTrainer
from config import RLTrainingConfig, PairwiseRewardModelConfig


complicated_sentences = get_complicated_sentence("./data/gpt4_complications.txt")

config = RLTrainingConfig(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    level="A",
    word_list_path="./data/CEFR_word_list.csv",
    training_sentences_path="./data/gpt4_complications_score_wiki_varied.txt"
)

pairwise_reward_model_config = PairwiseRewardModelConfig(
    model_name="./reward_model/pairwise_reward_model_A",
    num_labels=2,
    level="A"
)

rl_trainer = CEFRRLTrainer(config, pairwise_reward_model_config)

rl_trainer.train()