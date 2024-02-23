import torch
import time
from collections import Counter


class NGram:
    def __init__(self,input_ids, n, vocab_size, beta=0.9, sw_coeff=0., stop_words_ids=[]):
        super().__init__()
        assert sw_coeff >= 0.
        self.n = n
        self.tokens = input_ids
        self.vocab_size = vocab_size
        self.beta = beta
        self.sw_coeff = sw_coeff
        self.stop_words_ids = stop_words_ids

        # initialise the ngram model
        self.generated_ngrams = [{} for _ in range(n)]
        for idx in range(1,self.n+1):
            generated_ngram = self.generated_ngrams[idx-1]
            for ngram in zip(*[input_ids[i:] for i in range(idx)]):
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def penalize(self, query, candidates):
        penalty = torch.zeros(self.vocab_size)
        if len(query)<self.n-1:
            return penalty
        for cand in candidates:
            remaining = 1
            score = 0

            for i in range(self.n-1,-1, -1):
                if i == 0:
                    key = ()
                else:
                    key = tuple(query[-i:])
                ngram_cands = self.generated_ngrams[i].get(key, [])
                ngram_count = Counter(ngram_cands)
                

                if cand not in ngram_count:
                    continue

                total = sum([ngram_count[k] for k in ngram_count])
                if i == 0:
                    cur_score = ngram_count[cand] / total
                    score += remaining * cur_score
                else:
                    cur_score = ngram_count[cand] / (total + 1)
                    score += remaining * self.beta * cur_score
                remaining = remaining - remaining * self.beta

            if cand in self.stop_words_ids:
                penalty[cand] = self.sw_coeff * score
            else:
                penalty[cand] = score
        return penalty

    def update(self, new_token):
        for i in range(min(self.n, len(self.tokens)+1)):
            if i == 0:
                key = ()
            else:
                key = tuple(self.tokens[-i:])

            self.generated_ngrams[i][key] = self.generated_ngrams[i].get(key, []) + [new_token]
        self.tokens = self.tokens + [new_token]
