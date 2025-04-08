
This is the code for the NAACL 2025 paper: [Aligning Sentence Simplification with ESL Learner's Proficiency for Language Acquisitions](https://arxiv.org/abs/2502.11457). 

Please ensure that the `transformers` and `pytorch` libraries are installed, then
run the following command to install the dependencies:
```
pip install xformers peft accelerate bitsandbytes datasets flash-attn lightning wandb
pip install "trl<0.12.0" # when use train_rl_model
pip install trl # when use train_reward_model
```

Run `python train_reward_model.py` to train the sentence-level reward model; `python train_reward_model.py -h` to see available options;  
Run `python train_rl_model.py` to train the simplification model;`python train_rl_model.py -h` to see available options;    

Please prepare `CEFR_word_list.csv` and put it in the `data` folder; the format is as follows:

| base | level |
|------|-------|
| word | A1 |
| phrase word | C1 |


Update ongoing: 
- Add baseline modeling; 
- Add evaluation scripts; 
- Add baseline training and evaluation scripts. 
- Test version compatibility








