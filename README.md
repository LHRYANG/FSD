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
prompt_lst = ["内蒙古大草原上的"]
outputs = fsd_vec_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.4, model_name_or_path=model_name_or_path,
                                   language="chinese", max_length=128, min_length=128, n=2, beta=0.9, sw_coeff=1)
generation_lst = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
for text in generation_lst:
   print(''.join(text.split(' ')))
```
```
The output is:
内蒙古大草原上的一个小村庄，这里有着天下第一草原的美誉。在那里，你可以看到大草原上最美丽的风景，也可以感受到草原上最纯净的自然风光。这里是中国最大的草原生态旅游区，也是世界上唯一一个被称为世界最长草原的地方。草原上的人们都会穿着各种颜色的衣服，他们
```
To find more settings, you can refer to `eval.py`. 

