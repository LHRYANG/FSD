import torch
from transformers import AutoTokenizer
import time
device = torch.device('cuda')
from collections import Counter

# You can add punctuations or other kinds of words to realize more granular control
# PUNCTUATIONS = []
# STOP_WORDS = []
# TOXIC = [...]
class NGram(torch.nn.Module):
    def __init__(self,input_ids, n, vocab_size, beta=0.9,sw_coeff=1,PUNCTUATIONS=[],STOP_WORDS=[], model_path="model512"):
        super().__init__()
        '''
        n: n-gram model
        beta: parameter about smoothed n-gram model
        st_coeff: coefficient of stopwords  
        '''

        self.n = n
        self.tokens = input_ids
        self.vocab_size = vocab_size
        self.beta = beta
        self.sw_coeff = sw_coeff
        tok = AutoTokenizer.from_pretrained(model_path)
        self.sw_ids = list(set(tok.convert_tokens_to_ids(STOP_WORDS)))
        self.bd_ids = list(set(tok.convert_tokens_to_ids(PUNCTUATIONS)))

        # initialise the ngram model
        self.generated_ngrams = [{} for _ in range(n)]
        for idx in range(1,self.n+1):
            generated_ngram = self.generated_ngrams[idx-1]
            for ngram in zip(*[input_ids[i:] for i in range(idx)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def forward(self, input_ids, topk_id, topk_prob):
        penalty = torch.zeros(self.vocab_size)
        if len(input_ids)<self.n-1:
            return penalty
        for eee,cand in enumerate(topk_id):
            remaining = 1
            score = 0

            for i in range(self.n-1,-1, -1):
                if i == 0:
                    key = ()
                else:
                    key = tuple(input_ids[-i:])
                ngram_cands = self.generated_ngrams[i].get(key, [])
                ngram_count = Counter(ngram_cands)
                k_lst = list(ngram_count.keys())
                v_lst = list(ngram_count.values())

                if cand not in k_lst or cand in self.bd_ids:  #or (cand not in self.cw_ids):
                    continue

                idx = k_lst.index(cand)
                if i == 0:
                    cur_score = v_lst[idx]/sum(v_lst)
                    score += remaining * cur_score
                else:
                    cur_score = v_lst[idx] / (sum(v_lst)+1)
                    score += remaining * self.beta * (cur_score)
                remaining = remaining - remaining*self.beta

            if cand in self.sw_ids:
                penalty[cand] = self.sw_coeff*score
            elif cand in self.bd_ids:
                penalty[cand] = 0
            else:
                penalty[cand] = score
        return penalty

    def update(self, new_token):
        for i in range(self.n):
            if i == 0:
                key = ()
            else:
                key = tuple(self.tokens[-i:])
            #print(key)
            self.generated_ngrams[i][key] = self.generated_ngrams[i].get(key, []) + [new_token]
        self.tokens = self.tokens + [new_token]
























