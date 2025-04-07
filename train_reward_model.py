from modeling.pairwise_reward_model import CEFRRewardModel
from config import PairwiseRewardModelConfig



pairwise_reward_model_config = PairwiseRewardModelConfig(
    model_name="openai-community/gpt2",
    num_labels=2,
    level="A"
)

pairwise_reward_model = CEFRRewardModel(pairwise_reward_model_config)

pairwise_reward_model.train()