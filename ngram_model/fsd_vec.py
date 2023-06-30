import torch
from torch.nn.functional import normalize
from collections import Counter
from torch_scatter import scatter_max, scatter_mean

class HiddenSoftNGram(torch.nn.Module):
    def __init__(self, n, device, tokenizer, sw_coeff=0., stop_words_ids=[], choose="max"):
        #func inter weighted_inter
        super().__init__()
        assert sw_coeff >= 0.
        self.n = n
        self.n_vocab = len(tokenizer)
        self.device = device
        self.previous_hidden_states =None
        self.choose = choose
        self.sw_coeff = sw_coeff
        self.sw_weight = torch.ones(self.n_vocab, device=self.device, dtype=torch.float)
        for sid in stop_words_ids:
            self.sw_weight[sid] = self.sw_coeff

    def forward(self,prefix, hidden_states):
        bsz = prefix.shape[0]
        penalty = torch.zeros((bsz, self.n_vocab), device=self.device, dtype=torch.float)
        if prefix.shape[1] < self.n:
            return penalty

        if self.previous_hidden_states == None:
            self.previous_hidden_states = hidden_states[-1][:,:-1]
            last_hidden_states = hidden_states[-1][:, -1].unsqueeze(1)
            new_last_hidden_states = hidden_states[-1][:,-self.n+1:].view(bsz,1,-1)
        else:
            last_hidden_states = hidden_states[-1]
            if self.n != 2:
                new_last_hidden_states = torch.cat((self.previous_hidden_states[:,-self.n+2:],hidden_states[-1]),dim=1).view(bsz,1,-1)
            else:
                new_last_hidden_states=hidden_states[-1]


        lll = self.previous_hidden_states.shape[1]
        keys_hidden_states = self.previous_hidden_states[:,:lll-self.n+2]
        for ttt in range(self.n-2):
            keys_hidden_states =torch.cat((keys_hidden_states, self.previous_hidden_states[:,1+ttt:lll-self.n+2+ttt+1]), dim=-1)

        temp_penalty = torch.matmul(normalize(new_last_hidden_states,dim=-1),torch.transpose(normalize(keys_hidden_states,dim=-1),1,2))


        if self.choose == "avg":
            chengfa, argmax = scatter_mean(temp_penalty.squeeze(1), prefix[:,self.n-1:],out=penalty)
        if self.choose == "max":
            chengfa, argmax = scatter_max(temp_penalty.squeeze(1), prefix[:,self.n-1:],out=penalty)

        penalty = torch.clamp(penalty, min=0, max=1)
        self.previous_hidden_states = torch.cat((self.previous_hidden_states,last_hidden_states),dim=1)

        return penalty*self.sw_weight





