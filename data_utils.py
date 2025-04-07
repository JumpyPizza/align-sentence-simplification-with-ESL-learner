import json
import requests
from typing import Dict, List, Tuple, Union
from sklearn.utils import shuffle

def read_complicated_lines(path: str) -> List[str]:
    """Reads GPT generated complicated sentences and extracts sentence list."""
    # with open(path, "r") as f:
    #     return json.load(f)["lines"]
    complicated_score_wiki = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            sent = json.loads(line)['response']['body']['choices'][0]['message']['content']
            complicated_score_wiki.append(sent)
    return complicated_score_wiki

def download_cefr_data(data_urls: Dict[str, str]) -> Dict[str, List[str]]:
    """Download data from URLs and split into lines."""
    data = {}
    for key, url in data_urls.items():
        response = requests.get(url)
        data[key] = response.text.strip().split("\n")
    return data

def get_level_sentences(samples: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Sort sentences into A, B, C levels."""
    level_A, level_B, level_C = [], [], []
    
    for sample in samples:
        s, l1, l2 = sample.split("\t")
        l1, l2 = int(l1), int(l2)
        
        if l1 == l2:
            if l1 <= 2:
                level_A.append(s)
            elif l1 <= 4:
                level_B.append(s)
            else:
                level_C.append(s)
        else:
            max_level = max(l1, l2)
            min_level = min(l1, l2)
            if max_level <= 2:
                level_A.append(s)
            elif min_level >= 3 and max_level <= 4:
                level_B.append(s)
            elif min_level >= 5:
                level_C.append(s)
            else:
                # For ambiguous cases, assign to higher level
                if max_level <= 4:
                    level_B.append(s)
                else:
                    level_C.append(s)
                
    return level_A, level_B, level_C

def get_preference_pairs(priority_list: List[str], 
                        other_list_1: List[str], 
                        other_list_2: List[str]) -> Tuple[List[str], List[str]]:
    """Create preference pairs for reward modeling."""
    chosen = []
    rejected = []

    # Add samples from priority list as chosen samples
    for sample in priority_list:
        chosen.append(sample)
        
    # Add samples from other lists as rejected samples
    for sample in other_list_1:
        rejected.append(sample)
    for sample in other_list_2:
        rejected.append(sample)

    # Ensure equal number of chosen/rejected pairs
    rejected = shuffle(rejected, random_state=0)
    if len(chosen) < len(rejected):
        rejected = rejected[:len(chosen)]
    else:
        rejected = [rejected[i % len(rejected)] for i in range(len(chosen))]

    return chosen, rejected

def load_cefr_data(level: str, mode: str = "reward") -> Union[Dict[str, Dict[str, List[str]]], List[str]]:
    """Load and prepare CEFR data for training.
    
    Args:
        level: Target CEFR level (A, B, or C)
        mode: Either "reward" for reward modeling or "rl" for RL training
        
    Returns:
        For reward mode: Dict with train/eval data containing chosen/rejected pairs
        For RL mode: List of training sentences
    """
    # URLs for downloading data
    data_urls = {
        "score_train": "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/SCoRE/CEFR-SP_SCoRE_train.txt",
        "score_dev": "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/SCoRE/CEFR-SP_SCoRE_dev.txt",
        "score_test": "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/SCoRE/CEFR-SP_SCoRE_test.txt",
        "wiki_train": "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_train.txt",
        "wiki_dev": "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_dev.txt",
        "wiki_test": "https://raw.githubusercontent.com/yukiar/CEFR-SP/main/CEFR-SP/Wiki-Auto/CEFR-SP_Wikiauto_test.txt"
    }
    
    # Download and load data
    data = download_cefr_data(data_urls)
    
    # Split data by level
    train_A, train_B, train_C = get_level_sentences(data["score_train"] + data["wiki_train"])
    dev_A, dev_B, dev_C = get_level_sentences(data["score_dev"] + data["wiki_dev"])
    test_A, test_B, test_C = get_level_sentences(data["score_test"] + data["wiki_test"])
    
    if mode == "reward":
        # Prepare preference pairs based on target level
        if level == "A":
            train_chosen, train_rejected = get_preference_pairs(train_A, train_B, train_C)
            dev_chosen, dev_rejected = get_preference_pairs(dev_A, dev_B, dev_C)
            eval_chosen, eval_rejected = get_preference_pairs(test_A, test_B, test_C)
        elif level == "B":
            train_chosen, train_rejected = get_preference_pairs(train_B, train_C, train_A)
            dev_chosen, dev_rejected = get_preference_pairs(dev_B, dev_C, dev_A)
            eval_chosen, eval_rejected = get_preference_pairs(test_B, test_C, test_A)
        else:  # level C
            train_chosen, train_rejected = get_preference_pairs(train_C, train_B, train_A)
            dev_chosen, dev_rejected = get_preference_pairs(dev_C, dev_B, dev_A)
            eval_chosen, eval_rejected = get_preference_pairs(test_C, test_B, test_A)
            
        return {
            "train": {"chosen": train_chosen, "rejected": train_rejected},
            "dev": {"chosen": dev_chosen, "rejected": dev_rejected},
            "eval": {"chosen": eval_chosen, "rejected": eval_rejected}
        }
        
    elif mode == "rl":  # mode == "rl"
        # For RL training, return sentences of target level
        if level == "A":
            return train_A
        elif level == "B":
            return train_B
        else:
            return train_C
    else:
        raise ValueError(f"Invalid mode: {mode}")
    

def get_complicated_sentence(path: str) -> List[str]:
    """Get complicated sentences from file."""
    complicated_score_wiki = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            sent = json.loads(line)['response']['body']['choices'][0]['message']['content']
            complicated_score_wiki.append(sent)
    return complicated_score_wiki
