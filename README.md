# gpt4absa
Code for our paper:
All in One: An Empirical Study of GPT for Few-Shot Aspect-Based Sentiment Anlaysis

## Requirements
* Python 3.9
* PyTorch 1.12
* transformers
* openai
* tqdm

## Stage 1 (Generate Candidates for ASTE and AOPE subtasks)

* Train a suitable model to generate candidates for the corresponding dataset as required by [dual-encoder4aste](https://github.com/BaoSir529/dual-encoder4aste);
* 
* Download pretrained "Word Embedding of Amazon Product Review Corpus" with this [link](https://zenodo.org/record/3370051) and unzip into `./embedding/Particular/`.
* Prepare data for models, run the code [v1_data_process.py](./data/v1_data_process.py) and [v1_word2ids.py](./data/v1_word2ids.py), same for the v2 dataset.
* NOTE: For convenience, we have put with the processed data files in the data folder, so you can start the training phase directly.
```bash
python ./data/v1_data_process.py
python ./data/v1_word2ids.py
```
## Stage 2 (GPT inference)
* You can train the model using the corresponding .sh file [./run_with_ChatGPT.sh](./run_with_ChatGPT.sh), [./run_with_GPTJ.sh](./run_with_GPTJ.sh) or [./run_with_ERNIE.sh](./run_with_ERNIE.sh)
* For example:
```bash
bash ./run_with_ChatGPT.sh
```
* Note: When using the **GPT** model, please add the correct _openai.api_key_ and _openai.api_base_ in [./model/model.py](./model/model.py]);
* Note：When using the **ERNIE** model, please add the correct _API_KEY_ and _SECRET_KEY_ in [./model/model.py](./model/model.py]).

