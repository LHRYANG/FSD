# FSD
A Frustratingly Simple Decoding Method for Neural Text Generation

**[Contact]** If you have any questions, feel free to contact me via (hryang@se.cuhk.edu.hk).
#### 1. Install requirement
Except transformers, pytorch, you also need to install torch_scatter by running
```bash
pip install torch_scatter
```
#### 2. generate given prefix
In this example, I will show you how to use our decoding method.

```python
# Change the name in the huggingface models
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import fsd_decoding, fsd_vec_decoding

# Chinese
model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)

 ### settings of tokenizer, if you want to decode a batch, you need to set the pad_token_id
tokenizer.padding_side = "left"
if tokenizer.pad_token_id == None:
    # for English gpt2, the pad_token_id is not set, we use eos_token as pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt_lst = ["内蒙古大草原上的"]

'''
    k: top-k candidate words are selected,default 3 
    alpha: (1-alpha)p_lm -(alpha)*penalty
    model_name_or_path: same as the tokenizer 
    n: n-gram
    beta: smoothed n-gram model, default 0.9
    sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1),default 1
    max_length: decoding max_length-prompt_length steps
    punctuations=[] and stop_words=[]: You can add punctuations or other kinds of words to realize more granular control, If you use GPT-2, you at least need to
    add two special tokens ('Ċ' and 'ĊĊ') to punctuations, otherwise, some grammar errors may occur.
'''

outputs = fsd_vec_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.4, model_name_or_path=model_name_or_path,
                                    max_length=128, n=2, beta=0.9, sw_coeff=1,punctuations=[],stop_words=[])
generation_lst = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
for text in generation_lst:
   print(''.join(text.split(' ')))

# English 
model_name_or_path = "gpt2-large"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)

 ### settings of tokenizer, if you want to decode a batch, you need to set the pad_token_id
tokenizer.padding_side = "left"
if tokenizer.pad_token_id == None:
    # for English gpt2, the pad_token_id is not set, we use eos_token as pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt_lst = ["Before you came here, we have"]
outputs = fsd_vec_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.45, model_name_or_path=model_name_or_path,
                                    max_length=128, n=2, beta=0.9, sw_coeff=1,punctuations=['Ċ','ĊĊ'],stop_words=[])

generation_lst = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
for text in generation_lst:
   print(text.replace('\n',' '))


```
```
The output is:

内蒙古大草原上的一个小村庄，这里有着天下第一草原的美誉。在那里，你可以看到大草原上最美丽的风景，也可以感受到草原上最纯净的自然风光。这里是中国最大的草原生态旅游区，也是世界上唯一一个被称为世界最长草原的地方。草原上的人们都会穿着各种颜色的衣服，他们

Before you came here, we have a lot of things to do. We need to get the city ready for the Olympics."  The mayor said he was "very happy" with the progress made in recent months and that "we're going to be able to have our first Olympic Games in 20 years."  He also noted that the city has been working on its bid since 2008, when it received $1.5 million from the federal government.  "We've had some very good meetings with the IOC," he said. "I think we're ready to go now."  Mayor Rob Ford, who is running
```
To find more settings, you can refer to `eval.py`. 

