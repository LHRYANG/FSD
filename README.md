# Implementation of FSD
A Frustratingly Simple Decoding Method for Neural Text Generation

#### 1. Install requirement
Except transformers, pytorch, you also need to install torch_scatter by running
```bash
pip install torch_scatter
```
#### 2. Generate continuation given prefix
In this example, I will show you how to use our decoding method.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import fsd_decoding, fsd_vec_decoding

# Chinese
model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
device = torch.device("cuda")
model.to(device)

### settings of tokenizer, if you want to decode a batch, you need to set the pad_token_id
tokenizer.padding_side = "left"
if tokenizer.pad_token_id == None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt_lst = ["内蒙古大草原上的"]

outputs = fsd_vec_decoding(model, tokenizer, prompt_lst,
                           k=3, alpha=0.4, max_length=128,
                           n=2, sw_coeff=0., stop_words_ids=[])
generation_lst = tokenizer.batch_decode(outputs)
for text in generation_lst:
    print(text)

# English 
model_name_or_path = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
device = torch.device("cuda")
model.to(device)

 ### settings of tokenizer, if you want to decode a batch, you need to set the pad_token_id
tokenizer.padding_side = "left"
if tokenizer.pad_token_id == None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt_lst = ["Before you came here, we have"]
outputs = fsd_vec_decoding(model, tokenizer, prompt_lst,
                           k=3, alpha=0.45, max_length=128,
                           n=2, sw_coeff=0., stop_words_ids=tokenizer.convert_tokens_to_ids(['Ċ','ĊĊ']))

generation_lst = tokenizer.batch_decode(outputs)
for text in generation_lst:
    print(text)


```
```
The output is:

内蒙古大草原上的一个小村庄，这里有着天下第一草原的美誉。在那里，你可以看到大草原上最美丽的风景，也可以感受到草原上最纯净的自然风光。这里是中国最大的草原生态旅游区，也是世界上唯一一个被称为世界最长草原的地方。草原上的人们都会穿着各种颜色的衣服，他们

Before you came here, we have a lot of things to do. We need to get the city ready for the Olympics."  The mayor said he was "very happy" with the progress made in recent months and that "we're going to be able to have our first Olympic Games in 20 years."  He also noted that the city has been working on its bid since 2008, when it received $1.5 million from the federal government.  "We've had some very good meetings with the IOC," he said. "I think we're ready to go now."  Mayor Rob Ford, who is running
```

#### 3. Explanations and Settings of Hyperparameters
**explanations**

- k: top-k candidate words are selected, default 3 
- alpha: (1-alpha)p_lm -(alpha)*penalty
- max_length: decoding max_length-prompt_length steps
- n: the order of n-gram models
- beta: the smoothness of n-gram models, default 0.9 (only for discrete version)
- sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 1.
- stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.

**recommended settings (Chinese)**
| Hyperparameter | fsd_decoding | fsd_vec_decoding |
|----------------|----------|-----------|
| alpha          | 0.55     | 0.4       |
| n              | 3        | 2         |

**recommended settings (English)**
| Hyperparameter | fsd_decoding | fsd_vec_decoding |
|----------------|----------|-----------|
| alpha          | 0.65     | 0.45      |
| n              | 3        | 2         |
