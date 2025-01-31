import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, BertForSequenceClassification
from typing import List, Optional, Literal
from dataclasses import dataclass
from config import BeamSearchConfig


def get_level_tokens(level: int) -> tuple[str, str]:
    """Get level tokens for GPT2 conditioning."""
    level_token = f"<lvl_{level}>"
    if level <= 2:
        learner_token = "<bgn>"
    elif level <= 4:
        learner_token = "<intm>"
    else:
        learner_token = "<advn>"
    return level_token, learner_token

class ConditionalScorer:
    """Base class for conditional scoring."""
    def compute_conditional_scores(self, generated_ids: torch.Tensor, 
                                 candidate_ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class BertScorer(ConditionalScorer):
    """BERT-based conditional scoring."""
    def __init__(self, model, tokenizer, level: int):
        self.model = model
        self.tokenizer = tokenizer
        self.level = level
        
    def compute_conditional_scores(self, generated_ids: torch.Tensor, 
                                 candidate_ids: torch.Tensor) -> torch.Tensor:
        candidate_tokens = self.tokenizer.batch_decode(candidate_ids, skip_special_tokens=True)
        inputs = self.tokenizer(candidate_tokens, padding=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        _, _, condition_logits = self.model(**inputs)
        scores = torch.nn.functional.logsigmoid(condition_logits)[:, self.level].clone()
        return scores

class GPT2Scorer(ConditionalScorer):
    """GPT2-based conditional scoring."""
    def __init__(self, model, tokenizer, level: int):
        self.model = model
        self.tokenizer = tokenizer
        self.level = level
        self.level_token, self.learner_token = get_level_tokens(level)
        
    def compute_conditional_scores(self, generated_ids: torch.Tensor, 
                                 candidate_ids: torch.Tensor) -> torch.Tensor:
        generated_tokens = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        tokens_with_level = [
            self.level_token + self.learner_token + t for t in generated_tokens
        ]
        
        inputs = self.tokenizer(tokens_with_level, padding=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1]  # Get next token predictions
            
        candidate_tokens = [
            self.tokenizer.decode([token_id], skip_special_tokens=False) 
            for token_id in candidate_ids[0]
        ]
        
        candidate_ids = [
            self.tokenizer.encode(token, add_special_tokens=False) 
            for token in candidate_tokens
        ]
        
        log_probs = []
        for i in range(len(generated_ids)):
            batch_probs = []
            for token_ids in candidate_ids:
                if token_ids:
                    prob = torch.log_softmax(logits[i], dim=-1)[token_ids[0]].item()
                    batch_probs.append(prob)
            log_probs.append(batch_probs)
            
        return torch.FloatTensor(log_probs).cuda()

class LevelWeighter:
    """Unified level weighting for beam search."""
    def __init__(self, tokenizer, input_batch_seq_len: int, 
                 scorer: ConditionalScorer, condition_lambda: float, 
                 topk: int, wait: int = 3):
        self.tokenizer = tokenizer
        self.scorer = scorer
        self.input_batch_seq_len = input_batch_seq_len
        self.topk = topk
        self.condition_lambda = condition_lambda
        self.wait = wait

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_input_ids = input_ids[:, self.input_batch_seq_len:].clone()
        
        if generated_input_ids.shape[-1] < self.wait:
            return scores
            
        top_scores, top_score_indices = torch.topk(scores, self.topk, dim=1)
        
        # Create candidate sequences
        ids_expanded = generated_input_ids.repeat_interleave(self.topk, dim=0)
        candidate_seq = torch.cat([ids_expanded, top_score_indices.view(-1, 1)], dim=1)
        
        # Get conditional scores
        condition_scores = self.scorer.compute_conditional_scores(
            generated_input_ids,
            candidate_seq
        ).view(scores.shape[0], self.topk)
        
        # Combine scores
        processed_top_scores = top_scores + self.condition_lambda * condition_scores
        processed_scores = scores.clone()
        processed_scores.scatter_(1, top_score_indices, processed_top_scores)
        
        return processed_scores

class ConditionalBeamSearch:
    """Unified conditional beam search."""
    def __init__(self, config: BeamSearchConfig):
        self.config = config
        self.model, self.tokenizer = self._init_base_model()
        self.scorer = self._init_scorer()
        
    def _init_base_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_path, 
            padding_side='left'
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return model, tokenizer
        
    def _init_scorer(self):
        if self.config.condition_type == "bert":
            
            model = BertForSequenceClassification.from_pretrained(
                self.config.condition_model_path,
                num_labels=6
            ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(self.config.condition_tokenizer_path)
            return BertScorer(model, tokenizer, self.config.level)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.condition_model_path
            ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(self.config.condition_tokenizer_path)
            
            # Add special tokens for GPT2
            special_tokens = ["<|eot_id|>", "<|end_of_text|>"]
            tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
            model.resize_token_embeddings(len(tokenizer))
            
            special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
            with torch.no_grad():
                model.transformer.wte.weight[special_token_ids] = \
                    model.transformer.wte.weight[tokenizer.eos_token_id].clone()
                    
            model.eval()
            return GPT2Scorer(model, tokenizer, self.config.level)
        
    def generate(self, prompts: List[str]) -> List[str]:
        messages = [[{"role": "user", "content": p}] for p in prompts]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            truncation=None,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        prompt_len = input_ids.shape[-1]
        
        processor = LevelWeighter(
            self.tokenizer,
            prompt_len,
            self.scorer,
            self.config.condition_lambda,
            self.config.topk
        )
        
        logits_processor = LogitsProcessorList([processor])
        
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            logits_processor=logits_processor
        )
        
        return self.tokenizer.batch_decode(generated_ids[:, prompt_len:], skip_special_tokens=True) 