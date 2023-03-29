import torch
from torch.nn.functional import normalize
from collections import Counter
from transformers import AutoTokenizer
from torch_scatter import scatter_max, scatter_mean


# You can add punctuations or other kinds of words to realize more granular control
PUNCTUATIONS = []
STOP_WORDS = []
# TOXIC = [...]

class HiddenSoftNGram(torch.nn.Module):
    def __init__(self, n, device,n_vocab,beta=0.8,choose="max",sw_coeff=1, model_path = "model512",language="chinese"):
        #func inter weighted_inter
        super().__init__()
        if language == "english":
            PUNCTUATIONS.append('ĊĊ')
            PUNCTUATIONS.append('Ċ')
        self.n = n
        self.beta = beta
        self.n_vocab = n_vocab
        self.device = device
        self.previous_hidden_states =None
        self.choose = choose
        tok = AutoTokenizer.from_pretrained(model_path)
        self.sw_ids = list(set(tok.convert_tokens_to_ids(STOP_WORDS)))
        self.bd_ids = list(set(tok.convert_tokens_to_ids(PUNCTUATIONS)))
        self.sw_coeff = sw_coeff
        self.bd_weight = torch.ones(self.n_vocab, device=self.device, dtype=torch.float)
        for sid in self.bd_ids:
            self.bd_weight[sid] = 0
        self.sw_weight = torch.ones(self.n_vocab, device=self.device, dtype=torch.float)
        for sid in self.sw_ids:
            self.sw_weight[sid] = self.xishu

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
                new_last_hidden_states = torch.cat((self.previous_hidden_states[:,-self.n+2:],hidden_states[-1]),dim=-1)
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

        return penalty*self.bd_weight*self.sw_weight





