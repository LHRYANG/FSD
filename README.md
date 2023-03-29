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


LANGUAGE = "chinese" # chinese or english or others
model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.to(device)

tokenizer.padding_side = "left"
if LANGUAGE == "chinese":
   #中文模型的设置
   model.config.eos_token_id = None
   tokenizer.pad_token = tokenizer.cls_token
prompt_lst = ["腾讯是一家"]
outputs = fsd_vec_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.4, model_name_or_path=model_name_or_path,
                                   language="chinese", max_length=128, min_length=128, n=2, beta=0.9, sw_coeff=1)
generation_lst = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
for text in generation_lst:
   print(''.join(text.split(' ')))
```
The outputs are:
>>> 腾讯是一家互联网公司，它的产品和服务都是基于腾讯的技术。我们在这个领域有很多优势，比如说我们的客户群体，他们对于互联网的理解和认知，我们是非常强的。所以我们会去做一些事情，比如说我们在做一件事情之前，就要去了解这个行业的发展趋势，这样才能够更好地把握未来

To find more settings, you can refer to `eval.py`. 

