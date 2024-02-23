import torch
from torch.nn.functional import normalize
from collections import Counter
from torch_scatter import scatter_max, scatter_mean

class HiddenSoftNGram:
    def __init__(self, n, device, vocab_size, sw_coeff=0., stop_words_ids=[], choose="max"):
        super().__init__()
        assert sw_coeff >= 0.
        self.n = n
        self.vocab_size = vocab_size
        self.device = device
        self.hidden_states = None
        self.choose = choose
        self.sw_coeff = sw_coeff
        self.sw_weight = torch.ones(self.vocab_size, device=self.device, dtype=torch.float)
        for sid in stop_words_ids:
            self.sw_weight[sid] = self.sw_coeff

    def penalize(self, prefix, dtype):
        bsz = prefix.shape[0]
        penalty = torch.zeros((bsz, self.vocab_size), device=self.device, dtype=dtype)
        if prefix.shape[1] < self.n:
            return penalty

        query = self.hidden_states[:,-self.n+1:].view(bsz,1,-1)

        seqlen = self.hidden_states.shape[1]
        keys = [self.hidden_states[:,i:seqlen-self.n+1+i] for i in range(self.n-1)]
        keys = torch.cat(keys, dim=-1)

        temp_penalty = torch.matmul(normalize(query,dim=-1),torch.transpose(normalize(keys,dim=-1),1,2))


        if self.choose == "avg":
            scatter_mean(temp_penalty.squeeze(1), prefix[:,self.n-1:], out=penalty)
        if self.choose == "max":
            scatter_max(temp_penalty.squeeze(1), prefix[:,self.n-1:], out=penalty)

        penalty = torch.clamp(penalty, min=0, max=1)

        return penalty * self.sw_weight.to(penalty.dtype)
    
    def update(self, hidden_states):
        if self.hidden_states is None:
            self.hidden_states = hidden_states
        else:
            self.hidden_states = torch.cat((self.hidden_states, hidden_states),dim=1)
