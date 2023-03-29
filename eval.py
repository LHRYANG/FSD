import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import fsd_decoding, fsd_vec_decoding


if __name__ == "__main__":

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

    if LANGUAGE == "english":
        #英文模型的设置
        tokenizer.pad_token = tokenizer.eos_token

    #支持batch解码
    prompt_lst = ["这是一部非常好看的电影。讲的是","内蒙古大草原上"]
    encoded_prompt = tokenizer(prompt_lst, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(device)

    #fsd_vec decoding,向量版本的FSD解码（推荐使用）
    '''
    k: top-k candidate words are selected 
    alpha: (1-alpha)p_lm -(alpha)*penalty
    model_name_or_path: same as the tokenizer 
    language: if English, some special tokens should not be penalized, see fsd.py 
    n: n-gram
    beta: smoothed n-gram model
    sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1)
    '''
    #chinese
    if LANGUAGE == "chinese":
        outputs = fsd_vec_decoding(model, tokenizer, input_ids, k=3, alpha=0.4, model_name_or_path=model_name_or_path,
                                   language="chinese", max_length=256, min_length=256, n=2, beta=0.9, sw_coeff=1)
    #english
    if LANGUAGE == "english":
        outputs = fsd_vec_decoding(model, tokenizer, input_ids, k=3, alpha=0.45, model_name_or_path=model_name_or_path,
                                   language="english", max_length=256, min_length=256, n=2, beta=0.9, sw_coeff=1)



    #fsd_decoding, 非向量版本的FSD解码
    #chinese
    # outputs = fsd_decoding(model, tokenizer, input_ids, k=3, alpha=0.55, model_name_or_path=model_name_or_path,
    #                        language='chinese', max_length=256,min_length=256, n=3, beta=0.9,sw_coeff=1)
    #english
    # outputs = fsd_decoding(model, tokenizer, input_ids, k=3, alpha=0.65, model_name_or_path=model_name_or_path,
    #                        language='english', max_length=256, min_length=256, n=3, beta=0.9, sw_coeff=1)


    generation_lst = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print(generation_lst)
