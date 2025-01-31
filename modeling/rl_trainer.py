import torch
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from typing import List, Optional
from modeling.pairwise_reward_model import CEFRRewardModel
from modeling.lexical_reward_model import LexicalRewardModel
from config import RLTrainingConfig, PairwiseRewardModelConfig
from tqdm import tqdm
from data_utils import read_lines

class CEFRRLTrainer:
    def __init__(self, config: RLTrainingConfig, pairwise_reward_model_config: PairwiseRewardModelConfig):
        self.config = config
        self.model, self.tokenizer = self._init_model()
        self.pairwise_reward_model = self._init_pairwise_reward_model(pairwise_reward_model_config)
        self.lexical_reward_model = self._init_lexical_reward_model()
        self.ppo_trainer = self._init_ppo_trainer()

        
    def _init_model(self):
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["o_proj", "qkv_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.model_name,
            device_map="auto",
            peft_config=lora_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'left'
        
        return model, tokenizer
        
    def _init_pairwise_reward_model(self, pairwise_reward_model_config: PairwiseRewardModelConfig):
        return CEFRRewardModel(pairwise_reward_model_config)
    
    def _init_lexical_reward_model(self):
        return LexicalRewardModel(self.config.level, self.config.word_list_path)
        
    def _init_ppo_trainer(self):
        model_ref = create_reference_model(self.model)
        
        config = PPOConfig(
            tracker_project_name=self.config.wandb_project,
            exp_name=self.config.wandb_exp_name or f"CEFR_{self.config.level}_training",
            log_with = "wandb",
            model_name = self.config.model_name,
            reward_model = "word_count_freq",
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps
        )
        
        return PPOTrainer(config, self.model, model_ref, self.tokenizer)
        
    def train(self):
        if self.config.training_sentences_path is None:
            raise ValueError("training_sentences_path is not provided")
        training_sentences = read_lines(self.config.training_sentences_path)

        generation_kwargs = {
            "min_length": -1,
            "max_new_tokens": self.config.max_new_tokens,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        template = "Sentence: {} Please return a simplified sentence for English learner."
        all_messages = []
        for idx in range(0, len(training_sentences), self.config.batch_size):
            batch_msg = []
            for sent in training_sentences[idx:idx+self.config.batch_size]:
                content = template.format(sent)
                batch_msg.append([{"role": "user", "content": content}])
            all_messages.append(batch_msg)
        initial_word_reward_dict = {}
        for idx, w in enumerate(self.lexical_reward_model.word_list):
            initial_word_reward_dict[idx] = 1

        num_epochs = self.config.ppo_epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            generation_in_epoch = []
            for idx, batch_messages in tqdm(enumerate(all_messages)):
                try:
                    batch_inputs_ids = self.tokenizer.apply_chat_template(
                        batch_messages,
                        truncation=None,
                        padding=False,
                        add_generation_prompt=True,
                        return_dict=False,
                    )  # list
                    batch_inputs_tensors = [torch.tensor(i) for i in batch_inputs_ids] # list[Tensor]
                    #### Get response
                    response = self.ppo_trainer.generate(batch_inputs_tensors, batch_size = self.config.generate_bs, return_prompt=False, remove_padding=True, **generation_kwargs) # list[Tensor]
                    response_text = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response]
                    for t in response_text:
                        generation_in_epoch.append(t)
                    score = [torch.tensor( 0.7*self.lexical_reward_model.compute_reward(t)) for t in response_text]
                    reward_model_score_pt = self.pairwise_reward_model.compute_rewards(response_text)
                    reward_score = [ 0.5*r for r in reward_model_score_pt]
                    batch_score = []
                    for s1, s2 in zip(score, reward_score):
                        batch_score.append(s1+s2)
                    stats = self.ppo_trainer.step(batch_inputs_tensors, response, batch_score)
                    self.ppo_trainer.log_stats(stats, batch={"query":batch_messages, "response":response_text}, rewards=score)
                    if idx % 100 == 0:
                        print(response_text)
                        print(stats['ppo/mean_scores'])
                        print(stats['ppo/loss/policy'])
                        print(stats['ppo/loss/value'])
                except Exception as e:
                    print(e)
            print(f"Epoch {epoch + 1}/{num_epochs} finished")
            self.lexical_reward_model.update_rewards(generation_in_epoch)
        self.ppo_trainer.save_pretrained(self.config.output_dir)
                
    def save_checkpoint(self, path: str):
        self.ppo_trainer.save_pretrained(path) 