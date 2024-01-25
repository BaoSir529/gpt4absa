import json
import os
import time
import torch
import pickle
import numpy as np
from numpy import ndarray
from typing import Any


def clean_data(l):
    token, triplets = l.strip().split('####')
    temp_t = list(set([str(t) for t in eval(triplets)]))
    return token + '####' + str([eval(t) for t in temp_t]) + '\n'


def line2dict(l, is_clean=False):
    if is_clean:
        l = clean_data(l)
    sentence, triplets = l.strip().split('####')
    start_end_triplets = []
    token = sentence.split(' ')
    for t in eval(triplets):
        triplet_list = [' '.join(token[t[0][0]:t[0][-1] + 1]), ' '.join(token[t[1][0]:t[1][-1] + 1]), t[2]]
        triplet_str = '[' + ','.join(map(str, triplet_list)) + ']'
        start_end_triplets.append(triplet_str)
    return dict(sentence=sentence, token=sentence.split(' '), triplets=start_end_triplets)


def data_process(path):
    senti_map = {3: 'POS', 2: 'NEU', 1: 'NEG'}
    with open(path, 'r', encoding='utf-8-sig') as f:
        all_data = json.load(f)
    data_list = []
    for data in all_data:
        tokens = eval(data['tokens'])
        tokens = [i.lower()  for i in tokens]
        true_triplets = []
        true_aope = []
        true_alsc = []
        aspect = []
        pred_triplets = []
        pred_aope = []
        for triplet in data['pairs']:
            triplet_list = [' '.join(tokens[triplet[0]:triplet[1]]), ' '.join(tokens[triplet[2]:triplet[3]]), senti_map[triplet[4]]]
            aope_list = [' '.join(tokens[triplet[0]:triplet[1]]), ' '.join(tokens[triplet[2]:triplet[3]])]
            aspect_list = [' '.join(tokens[triplet[0]:triplet[1]])]
            alsc_list = [' '.join(tokens[triplet[0]:triplet[1]]), senti_map[triplet[4]]]
            triplet_str = '[' + ','.join(map(str, triplet_list)) + ']'
            aope_str = '[' + ','.join(map(str, aope_list)) + ']'
            alsc_str = '[' + ','.join(map(str, alsc_list)) + ']'
            aspect_str = '[' + ','.join(map(str, aspect_list)) + ']'


            true_triplets.append(triplet_str)
            true_aope.append(aope_str)
            true_alsc.append(alsc_str)
            aspect.append(aspect_str)

        for triplet in data['pair_preds']:
            triplet_list = [' '.join(tokens[triplet[0]:triplet[1]]),' '.join(tokens[triplet[2]:triplet[3]]), senti_map[triplet[4]]]
            aope_list = [' '.join(tokens[triplet[0]:triplet[1]]), ' '.join(tokens[triplet[2]:triplet[3]])]
            triplet_str = '[' + ','.join(map(str, triplet_list)) + ']'
            aope_str = '[' + ','.join(map(str, aope_list)) + ']'

            pred_triplets.append(triplet_str)
            pred_aope.append(aope_str)

        dict = {
            'ID': data['ID'],
            'sentence': data['sentence'].lower(),
            'token': tokens,
            'pred_triplets': pred_triplets,
            'pred_aope' : pred_aope,
            'triplets': true_triplets,
            'true_alsc': true_alsc,
            'true_aope': true_aope,
            'aspects' : aspect
        }

        if "aspects" in data.keys():
            dict['aspects'] = data['aspects']

        if "true_aspects" in data.keys():
            dict['true_alsc'] = data['true_aspects']


        data_list.append(dict)

    return data_list


def gen_sentence_label(token: list, triplets: list[tuple], senti):
    max_length = len(token)
    golden_label = np.zeros((max_length, max_length))
    for trip in triplets:
        asp_start, asp_end = trip[0][0], trip[0][1]
        opn_start, opn_end = trip[1][0], trip[1][1]
        golden_label[asp_start:asp_end + 1, opn_start:opn_end + 1] = senti.get(trip[2], 0)
    return golden_label


def gen_bert_triplets(triplets: list[tuple], token_length: list) -> list[tuple]:
    bert_triplets = []
    for trip in triplets:
        asp_start = trip[0][0]
        asp_end = trip[0][1]
        opn_start = trip[1][0]
        opn_end = trip[1][1]
        senti = trip[2]

        bert_asp_start = sum(token_length[0:asp_start])
        bert_asp_end = sum(token_length[0:asp_end + 1]) - 1
        bert_opn_start = sum(token_length[0:opn_start])
        bert_opn_end = sum(token_length[0:opn_end + 1]) - 1

        bert_triplets.append(tuple(([bert_asp_start, bert_asp_end], [bert_opn_start, bert_opn_end], senti)))
    return bert_triplets


def data_collate_fn(batch: list) -> dict:
    batch_size = len(batch)
    batch_pack = {}

    bert_token = get_long_tensor([data['token'] for data in batch])
    bert_input = get_long_tensor([data['bert_input'] for data in batch])
    bert_attention_mask = get_long_tensor([data['bert_attention_mask'] for data in batch])
    max_length = bert_token.size()[1]
    golden_label = []
    for data in batch:
        golden = data['golden_label']
        golden_size = golden.shape[0]
        golden_label.append(list(np.pad(golden, (0, max_length - golden_size), 'constant')))

    golden_label = torch.tensor(golden_label)

    batch_pack = {
        'bert_token': bert_token,
        'bert_input': bert_input,
        'attention_mask': bert_attention_mask,
        'golden_label': golden_label
    }

    return batch_pack


def get_long_tensor(tokens_list, max_len=None):
    # Convert list of tokens to a padded LongTensor.
    batch_size = len(tokens_list)
    token_len = max(len(x) for x in tokens_list) if max_len is None else max_len
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : min(token_len, len(s))] = torch.LongTensor(s)[:token_len]
    return tokens


def data_collate_func(batch: list):
    pass
    return batch


def save_variable(v, args, Temp=False):

    if not os.path.exists(f'{args.save_dir}/{args.dataset}/variable'):
        os.makedirs(f'{args.save_dir}/{args.dataset}/variable')

    current_time = time.strftime("%m-%d %H_%M", time.localtime())

    if Temp:
        file_name = os.path.join(f'{args.save_dir}/{args.dataset}/variable', 'Temp' + '.pkl')
    else:
        file_name = os.path.join(f'{args.save_dir}/{args.dataset}/variable',
                             f'{args.model}-{args.task}-F1={v.f1}_P={v.p}_R={v.r}_N={args.n} ' + current_time + '.pkl')

    f = open(file_name, 'wb')
    pickle.dump(v, f)
    f.close()


def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r


def save_result(result, args, M, mode='Pred'):

    if not os.path.exists(os.path.join(args.save_dir, args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.dataset))

    current_time = time.strftime("%m-%d %H_%M", time.localtime())
    file_name = os.path.join(os.path.join(args.save_dir, args.dataset), f'{args.model}-{args.task}-{mode}-F1={M.f1}_P={M.p}_R={M.r}_N={args.n} '+current_time+'.txt')

    with open(file_name, 'w') as f:
        for line in result:
            f.writelines(str(line)+"\n")