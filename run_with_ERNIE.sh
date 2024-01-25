echo For ALSC subtask...
python ./model/main.py --model ernie --pretrained_model_name bert-base-uncased --dataset_dir ./datasets --dataset 14res --n 10 --task alsc

echo For ASTE subtask...
python ./model/main.py--model ernie --pretrained_model_name bert-base-uncased --dataset_dir ./datasets --dataset 15res --n 10 --task aste

echo For AOPE subtask...
python ./model/main.py --model ernie --pretrained_model_name bert-base-uncased --dataset_dir ./datasets --dataset 16res --n 10 --task aope
