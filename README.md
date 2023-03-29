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
```

This example allows you to input your prefix and generate maximum of 512 tokens or achieves the `eos_token_id`.

```bash
./test.sh
```

In this chinese example, if the input prefix is `腾讯是一家`, the momentum decoding and contrastive search will generate the following results:

```text
Prefix >>> 腾讯是一家
[Momentum Decoding] 腾讯是一家非常有活力的公司，我们在移动互联网上也有很多创新，这些创新都是基于对用户需求的深刻理解。” 另外，腾讯还表示，将会与合作伙伴一起，把更多的创新应用带给用户，并通过开放、协作的方式，与更多合作伙伴共同推动中国互联网的发展。
