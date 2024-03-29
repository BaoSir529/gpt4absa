echo For ALSC subtask...
python ./model/main.py --model gptj --pretrained_model_name bert-base-uncased --gptj_dir EleutherAI/gpt-j-6B --dataset_dir ./datasets --dataset 14res --n 10 --task alsc

echo For ASTE subtask...
python ./model/main.py--model gptj --pretrained_model_name bert-base-uncased --gptj_dir EleutherAI/gpt-j-6B --dataset_dir ./datasets --dataset 15res --n 10 --task aste

echo For AOPE subtask...
python ./model/main.py --model gptj --pretrained_model_name bert-base-uncased --gptj_dir EleutherAI/gpt-j-6B --dataset_dir ./datasets --dataset 16res --n 10 --task aope