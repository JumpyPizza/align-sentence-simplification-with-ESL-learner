from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from typing import List, Dict, Union
import pandas as pd
import re
from tqdm import tqdm
from typing import Tuple

class LexicalRewardModel:
    def __init__(self, level: str, word_list_path: str):
        """Initialize lexical reward model for CEFR level."""
        self.level = level
        self.lemmatizer = WordNetLemmatizer()
        self.word_list, self.negative_word_list = self._load_word_lists(word_list_path)
        self.word_reward_dict = {idx: 1.0 for idx in range(len(self.word_list))}
        self.word_freq_dict = {idx: 0 for idx in range(len(self.word_list))}
        
    def _load_word_lists(self, path: str) -> Tuple[List[Union[str, List[str]]], List[Union[str, List[str]]]]:
        """Load and process CEFR word lists."""
        df = pd.read_csv(path)
        
        # Filter words based on level
        if self.level == "A":
            level_df = df[df['level'].isin(['A1', 'A2'])]
            negative_df = df[df['level'].isin(['B1', 'B2', 'C1', 'C2'])]
        elif self.level == "B":
            level_df = df[df['level'].isin(['B1', 'B2'])]
            negative_df = df[df['level'].isin(['C1', 'C2'])]
        else:  # level C
            level_df = df[df['level'].isin(['C1', 'C2'])]
            negative_df = pd.DataFrame(columns=df.columns)
            
        word_list = list(set([self._process_word(w) for w in level_df['base'].tolist()]))
        negative_word_list = list(set([self._process_word(w) for w in negative_df['base'].tolist()]))
        
        return word_list, negative_word_list
    
    def _process_word(self, word: str) -> Union[str, List[str]]:
        """Process a word or phrase from the CEFR profile word list based on a set of rules."""
        # TODO: update to add more rules
        word = word.strip()
        if len(word.split()) > 1:  # phrase
            # 1. remove ", etc.", "be"
            phrase = word.replace(", etc.", "").replace("be", "").replace(" to do sth", "")
            # 2. remove anything in ()
            phrase = re.sub(r'\(.*?\)', '', phrase)
            # 3. split with " or "
            phrases = phrase.split(" or ")[:1]
            processed = []
            for p in phrases:
                # 4. deal with sb/sth
                p = p.replace("sb/sth", "...").replace("sb", "...").replace("sth", "...")
                # 5. deal with / and ()
                if "/" in p:
                    parts = p.split()
                    new_parts = []
                    for part in parts:
                        if "/" in part:
                            part = part.split("/")[0]
                        new_parts.append(part)
                    p = " ".join(new_parts)
                    
                # 6. deal with ...
                if "..." in p:
                    p = [x.lstrip().rstrip() for x in p.split("...") if x.strip()]
                else:
                    p = [p.lstrip().rstrip()]
                    
                processed.extend([re.sub(r'\s+', ' ', x).lower() for x in p if x])# replace multiple blank
                
            return list(set(processed))
        
        return word.lower()
    
    def compute_reward(self, text: str) -> float:
        """Compute lexical reward for a text."""
        tokens_no_stop, tokens_with_stop = self._preprocess_text(text)
        
        score = 0
        negative_score = 0
        matched = []
        negative_matched = []
        #TODO: update the matching algorithm
        # Match single words
        for token in tokens_no_stop:
            for w in self.word_list:
                # is word and exactly the same
                if isinstance(w, str) and token == w:
                    matched.append(token)
                    score += self.word_reward_dict[self.word_list.index(w)]
                    break
                    
            for w in self.negative_word_list:
                if isinstance(w, str) and token == w:
                    negative_matched.append(token)
                    negative_score -= 1
                    break
        
        # Match phrases
        text_to_match = " ".join(tokens_with_stop)
        for idx, w in enumerate(self.word_list):
            if isinstance(w, list):
                if len(w) == 1:  # continuous phrase
                    if w[0] in text_to_match:
                        matched.append(w[0])
                        multiplier = 1.3 if len(w[0].split()) > 1 else 1.0
                        score += multiplier * self.word_reward_dict[idx]
                else:  # discontinuous phrase
                    matched_parts = 0
                    remaining_text = text_to_match
                    for part in w:
                        if len(part.split()) > 1 and part in remaining_text:
                            matched_parts += 1
                            remaining_text = "".join(remaining_text.split(part)[1:])
                    
                    if matched_parts > 0:
                        score += 1.3 * (matched_parts / len(w)) * self.word_reward_dict[idx]
        
        final_score = score + negative_score
        return final_score / len(tokens_no_stop) if tokens_no_stop else 1.0
    
    def _preprocess_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Preprocess text for matching."""
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text.lower())
        tokens = [w for w in tokens if w not in string.punctuation]
        
        tokens_no_stop = [w for w in tokens if w not in stop_words]
        tokens_no_stop = [self.lemmatizer.lemmatize(w) for w in tokens_no_stop]
        
        tokens_with_stop = [self.lemmatizer.lemmatize(w) for w in tokens]
        
        return tokens_no_stop, tokens_with_stop
    
    def update_rewards(self, generations: List[str]):
        """Update reward weights based on word frequencies in generations."""
        # after each epoch, adjust the score for each matched token #
        # based on the tokens frequency, and IDF #
        # frequency is calculated globally, is the overall count for the token #
        # IDF is to encourage more generations to have rare words, 
        # common words across all generations will have less score 
        # ### but we no not want word appear only in one generation

        self.word_freq_dict = {idx: 0 for idx in range(len(self.word_list))}
        
        for text in tqdm(generations):
            tokens_no_stop, tokens_with_stop = self._preprocess_text(text)
            text_to_match = " ".join(tokens_with_stop)
            
            # Count word frequencies
            for idx, w in enumerate(self.word_list):
                if isinstance(w, str):
                    if w in tokens_no_stop:
                        self.word_freq_dict[idx] += 1
                else:  # phrase
                    if len(w) == 1:
                        if w[0] in text_to_match:
                            self.word_freq_dict[idx] += 1
                    else:
                        matched = False
                        for part in w:
                            if len(part.split()) > 1 and part in text_to_match:
                                matched = True
                        if matched:
                            self.word_freq_dict[idx] += 1
        
        # Update rewards based on frequencies
        total_words = sum(self.word_freq_dict.values())
        for idx in self.word_freq_dict:
            tf = self.word_freq_dict[idx] / total_words if total_words > 0 else 0
            if tf >= 1/len(self.word_list):
                self.word_reward_dict[idx] = 2**(-10*tf) 