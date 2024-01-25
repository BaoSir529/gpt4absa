import os
import time
from tqdm import tqdm
import torch
import random
import argparse
import numpy as np
from torch.optim import AdamW

from torch.utils.data import DataLoader
from model import gpt_model, gptj_model, ernie_model
from aste_dataloader import ASTE_dataloader
from data_utils import save_variable, load_variavle, save_result
from utils.mertric import Mertics


def set_random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_and_evaluate(args):
    print('==' * 100)
    # set seed
    if args.seed is not None:
        set_random_seed(args.seed)

    dataset_dir = os.path.join(os.path.abspath(args.dataset_dir), args.dataset)

    print('> Load model...')
    if args.model == 'gpt':
        model = gpt_model(args)
    elif args.model == 'gptj':
        model = gptj_model(args).to(args.device)
    elif args.model == 'ernie':
        model = ernie_model(args)


    print('> Load dataset...')
    val_data_file = os.path.join(dataset_dir, 'val.txt')
    test_data_file = os.path.join(dataset_dir, 'test.txt')
    train_dataset = ASTE_dataloader(test_data_file, val_data_file, args=args, is_clean=True)

    print('> Training...')
    if args.retrain == -1:
        metrics = Mertics()
    else:
        metrics = load_variavle('./result/variable/'+args.dataset+'-metrics.data')
    for prompt in tqdm(train_dataset):
        ID = prompt['ID']
        if ID > args.retrain:
            golden_label = prompt['gl']
            prompt_text = prompt['prompt']
            metrics.true_inc(ID, golden_label)
            pred = model(ID, prompt_text)
            metrics.pred_inc(ID, pred)
            save_variable(metrics, args, Temp=True)
            if ID % 100 == 1:
                print(f'>>>>>>F1:\t{metrics.report()}')
    metrics.report()
    save_variable(metrics, args)
    save_result(sorted(metrics.pred_set), args, metrics)
    save_result(sorted(metrics.true_set), args, metrics, mode='True')
    print('\nF1:{},P:{},R:{}\n'.format(metrics.f1, metrics.p, metrics.r))

    return 0


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='14res', help='14lap,14res,15res,16res,MAMS')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/')
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--saved_file', type=str, default=None)
    parser.add_argument('--n', type=int, default=1)

    # Data process
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--sentence_max_length', type=int, default=128)

    # Train model parameters
    parser.add_argument('--task', type=str, default='aste', help='aope,alsc,aste')
    parser.add_argument('--model', type=str, default='gpt', help='gpt, gptj, ernie')
    parser.add_argument('--pretrained_model_name', type=str, default='E:\\bert-base-uncased')
    parser.add_argument('--gptj_dir', type=str, default='E:\\gpt-j')
    parser.add_argument('--retrain', type=int, default=-1)
    parser.add_argument('--hidden_dim', type=int, default=25)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=2e-5)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    args = parser.parse_args()

    return args


def run():
    args = get_parameters()
    train_and_evaluate(args)


if __name__ == '__main__':
    run()
