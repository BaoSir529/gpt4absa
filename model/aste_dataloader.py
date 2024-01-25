import os
import torch
import numpy as np
from tqdm import tqdm
from data_utils import *
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# senti_vocab = {'None': 0, 'POS': 1, 'NEU': 2, 'NEG': 3}
AOPE_HEAD = '''Complete the aspect-opnion pair extraction task according to the examples and corresponding candidartes. Noting that the word in results can only come from the input sentence. Results are not restricted to candidates.\n===\n'''
ALSC_HEAD = '''Complete the aspect-level sentiment classification task according to the examples and corresponding aspects. Noting that the sentiment polarity can only be positive(POS)/neutral(NEU)/negative(NEG). \n===\n'''
ASTE_HEAD = '''Complete the aspect-level sentiment triplet extraction task according to the examples and corresponding candidartes. Noting that the word in results can only come from the input sentence and the sentiment state can only be positive(POS)/neutral(NEU)/negative(NEG). Results are not restricted to candidates.\n===\n'''


class ASTE_dataloader(Dataset):
    def  __init__(self, file_name, candidate_file, args, is_clean=True):
        super().__init__()

        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
        self.bert = BertModel.from_pretrained(args.pretrained_model_name)
        self.lower = args.lower
        self.file_name = file_name

        self.raw_data = data_process(file_name)
        self.candidate = data_process(candidate_file)

        self.data_embedding = self.bert_embedding(self.raw_data)
        print('Generate candidate embedding...')
        self.candidate_embedding = self.bert_embedding(self.candidate)
        self.similarity = self.calculate_cosine(self.data_embedding, self.candidate_embedding)

        self.data = self.gen_input(self.similarity, args.n, args.task)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def text2bert_id(self, token):
        re_token = []
        word_mapback = []
        word_split_len = []
        for idx, word in enumerate(token):
            temp = self.tokenizer.tokenize(word)
            re_token.extend(temp)
            word_mapback.extend([idx] * len(temp))
            word_split_len.append(len(temp))
        re_id = self.tokenizer.convert_tokens_to_ids(re_token)
        return re_id, word_mapback, word_split_len

    def bert_embedding(self, datas):
        embeddings = []
        for d in tqdm(datas):
            sentence = d['sentence']
            input = self.tokenizer(sentence, return_tensors='pt')
            with torch.no_grad():
                output = self.bert(**input)
                embed = output.last_hidden_state[:, 0, :].numpy()
            embeddings.append(embed)
        return embeddings

    def calculate_cosine(self, data_embed, candidate_embed):
        a = np.array(data_embed).squeeze(1)
        b = np.array(candidate_embed).squeeze(1)
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)

        dot_product = np.dot(a, b.T)

        sim_matrix = dot_product / (a_norm @ b_norm.T)

        return sim_matrix

    def gen_input(self, sim, n, task):
        inputs = []

        if task in ['aope', 'alsc', 'aste']:
            if task == 'aope':
                gen_func = self.gen_prompt_text_AOPE
                target = 'true_aope'
            elif task == 'alsc':
                gen_func = self.gen_prompt_text_ALSC
                target = 'true_alsc'
            elif task == 'aste':
                gen_func = self.gen_prompt_text_ASTE
                target = 'triplets'
        else:
            raise ValueError('Task not correct !')

        for idx, cosine in enumerate(sim):
            n_candidates = np.argsort(cosine)[-n:][::-1].tolist() if n else []
            prompt_text = gen_func(idx, n_candidates)
            golden_label = self.raw_data[idx][target]
            tmp = {
                'ID': idx,
                'prompt': prompt_text,
                'gl': golden_label
            }
            inputs.append(tmp)
        return inputs

    def gen_prompt_text_ASTE(self, sentence_idx, candidates_idx):
        test_data = self.raw_data[sentence_idx]
        candidate_text = ''
        for idx, index in enumerate(candidates_idx):
            candidate = self.candidate[index]

            candidate_text += (f'<EXAMPLE#{idx+1} BEGIN>\n'
                               + 'SENTENCE:\t' + candidate['sentence']
                               + '\nCANDIDATES:\t' + '|'.join(candidate['pred_triplets']) + '.'
                               + '\nRESULTS:\t' + '|'.join(candidate['triplets']) + '.'
                               + f'\n<END>\n')

        if not self.args.n:
            candidate_text = '<EXAMPLE BEGIN>\nSENTENCE:\tthe food here was mediocre at best .\nRESULTS:\t[food,mediocre,POS]|[here,best,POS].\n<END>\n'

        prompt_text = (ASTE_HEAD + candidate_text
                       + '<INPUT>\n'
                       + 'SENTENCE:\t' + test_data['sentence'])
        if self.args.n:
            prompt_text += '\nCANDIDATES:\t' + '|'.join(test_data['pred_triplets']) + '.' + '\nRESULTS:\t'
        else:
            prompt_text += '\nRESULTS:\t'

        return prompt_text

    def gen_prompt_text_AOPE(self, sentence_idx, candidates_idx):
        test_data = self.raw_data[sentence_idx]
        candidate_text = ''
        for idx, index in enumerate(candidates_idx):
            candidate = self.candidate[index]

            candidate_text += (f'<EXAMPLE#{idx + 1} BEGIN>\n'
                               + 'SENTENCE:\t' + candidate['sentence']
                               + '\nCANDIDATES:\t' + '|'.join(candidate['pred_aope']) + '.'
                               + '\nRESULTS:\t' + '|'.join(candidate['true_aope']) + '.'
                               + f'\n<END>\n')
        if not self.args.n:
            candidate_text = '<EXAMPLE BEGIN>\nSENTENCE:\tthe food here was mediocre at best .\nRESULTS:\t[food,mediocre]|[here,best].\n<END>\n'

        prompt_text = (AOPE_HEAD + candidate_text
                       + '<INPUT>\n'
                       + 'SENTENCE:\t' + test_data['sentence'])

        if self.args.n:
            prompt_text += '\nCANDIDATES:\t' + '|'.join(test_data['pred_aope']) + '.' + '\nRESULTS:\t'
        else:
            prompt_text += '\nRESULTS:\t'
        return prompt_text

    def gen_prompt_text_ALSC(self, sentence_idx, candidates_idx):
        test_data = self.raw_data[sentence_idx]
        candidate_text = ''
        for idx, index in enumerate(candidates_idx):
            candidate = self.candidate[index]

            candidate_text += (f'<EXAMPLE#{idx + 1} BEGIN>\n'
                               + 'SENTENCE:\t' + candidate['sentence']
                               + '\nASPECTS:\t' + '|'.join(candidate['aspects']) + '.'
                               + '\nRESULTS:\t' + '|'.join(candidate['true_alsc']) + '.'
                               + f'\n<END>\n')

        if not self.args.n:
            candidate_text = '<EXAMPLE BEGIN>\nSENTENCE:\tthe food here was mediocre at best .\nASPECTS:\t [food]|[here].\nRESULTS:\t[food,POS]|[here,POS].\n<END>\n'

        prompt_text = (ALSC_HEAD + candidate_text
                       + '<INPUT>\n'
                       + 'SENTENCE:\t' + test_data['sentence']
                       + '\nASPECTS:\t' + '|'.join(test_data['aspects']) + '.'
                       + '\nRESULTS:\t')
        return prompt_text
