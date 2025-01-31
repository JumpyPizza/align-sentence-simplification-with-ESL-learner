import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional
from tqdm import tqdm
from datasets import Dataset
from trl import RewardTrainer, RewardConfig
from config import RewardModelConfig
import os
from data_utils import load_cefr_data

class CEFRRewardModel:
    def __init__(self, config: RewardModelConfig):
        self.config = config
        self.model = self._init_model()
        self.tokenizer = self._init_tokenizer()
        
        
    def _init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        model.config.pad_token_id = model.config.eos_token_id
        return model.to('cuda')
        
    def _init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def prepare_dataset(self, chosen_texts: List[str], rejected_texts: List[str]) -> Dataset:
        dataset = Dataset.from_dict({
            "chosen": chosen_texts,
            "rejected": rejected_texts
        })
        
        def preprocess(examples):
            new_examples = {
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": [],
            }
            for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
                tokenized_chosen = self.tokenizer(chosen)
                tokenized_rejected = self.tokenizer(rejected)
                
                new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
                new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
                new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
                new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
                
            return new_examples
        dataset = dataset.filter(lambda x: len(x["input_ids_chosen"]) <= self.config.max_length and len(x["input_ids_rejected"]) <= self.config.max_length)
        return dataset.map(preprocess, batched=True, num_proc=self.config.dataset_num_proc)

    def train(self):
        data = load_cefr_data(self.config.level)
        train_dataset = self.prepare_dataset(data["train"]["chosen"], data["train"]["rejected"])
        dev_dataset = self.prepare_dataset(data["dev"]["chosen"], data["dev"]["rejected"])
        eval_dataset = self.prepare_dataset(data["eval"]["chosen"], data["eval"]["rejected"])
        
        os.environ["WANDB_PROJECT"]=self.config.wandb_project
        os.environ['WANDB_WATCH'] = 'false'  # used in Trainer
        os.environ['WANDB_NAME'] = 'reward-model'  # used in Trainer
        training_args = RewardConfig(
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            output_dir=self.config.output_dir,
            bf16=True,
            report_to = "wandb",
            logging_steps=1,
            eval_steps=10
        )

        trainer = RewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            args=training_args
        )
        
        trainer.train()
        trainer.test(eval_dataset=eval_dataset)
        trainer.save_model(self.config.output_dir)

    def compute_rewards(self, texts: List[str]) -> torch.Tensor:
        """Compute rewards for generated texts."""
        rewards = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.config.eval_batch_size):
                batch = texts[i:i + self.config.eval_batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                logits = self.model(**inputs).logits
                rewards.append(torch.sigmoid(logits))
        
        return torch.cat(rewards).squeeze().cpu()

    @classmethod
    def load_model(cls, path: str):
        """Load a saved reward model."""
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model.config.pad_token_id = model.config.eos_token_id
        model.to('cuda')
        model.eval()
        
        return model, tokenizer 