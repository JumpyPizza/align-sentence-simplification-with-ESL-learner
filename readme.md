
This is the code for the NAACL 2025 paper: [Aligning Sentence Simplification with ESL Learner's Proficiency for Language Acquisitions](https://arxiv.org/abs/2502.11457). 
Please run the following command to install the dependencies:
```
pip install xformers trl peft accelerate bitsandbytes datasets flash-attn lightning wandb
```
And ensure that the `transformers` library is installed:
```
pip install transformers 
```

Run `python train_reward_model.py` to train the sentence-level reward model;   
Run `python train_rl_model.py` to train the simplification model.

Please prepare `CEFR_word_list.csv` and put it in the `data` folder; the format is as follows:

| base | level |
|------|-------|
| word | A1 |
| phrase word | C1 |


Update ongoing: 
- Add baseline modeling; 
- Add evaluation scripts; 
- Add baseline training and evaluation scripts. 








