This folder contains the data used for the experiments.  
- `model_outputs` contains the outputs of the trained phi model in three levels, for both CEFR test and turk; Note that some responses contain sentences that do not follow the instruction (reasoning after simplification output; we filter them in post-processing by splitting with \n\n.)   
- `gpt4_complications_score_wiki_varied.txt` contains the GPT-generated complex sentences from the CEFR-SP dataset, which we used as the source in the RL training; note that it only contains the score and wiki part.
- `chosen_sentences_2.json` contains the sentences we randomly chose for the annotators to compare.  
