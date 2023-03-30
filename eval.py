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

    ### settings of tokenizer, if you want to decode a batch, you need to set the pad_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id == None:
        # for English gpt2, the pad_token_id is not set, we use eos_token as pad_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    #支持batch解码
    prompt_lst = ["这是一部非常好看的电影。讲的是","内蒙古大草原上"]


    #fsd_vec decoding,向量版本的FSD解码（推荐使用）
    '''
    k: top-k candidate words are selected 
    alpha: (1-alpha)p_lm -(alpha)*penalty
    model_name_or_path: same as the tokenizer 
    n: n-gram 
    beta: smoothed n-gram model, default 0.9
    sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 1
    
    '''
    #chinese
    if LANGUAGE == "chinese":
        outputs = fsd_vec_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.4, model_name_or_path=model_name_or_path,
                                   max_length=128,  n=2, beta=0.9, sw_coeff=1,punctuations=[],stop_words=[])
    #english
    if LANGUAGE == "english":
        outputs = fsd_vec_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.45, model_name_or_path=model_name_or_path,
                                    max_length=128,  n=2, beta=0.9, sw_coeff=1,punctuations=[],stop_words=[])



    #fsd_decoding, 非向量版本的FSD解码
    #chinese
    outputs = fsd_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.55, model_name_or_path=model_name_or_path,
                            max_length=128, n=3, beta=0.9,sw_coeff=1,punctuations=[],stop_words=[])
    #english
    outputs = fsd_decoding(model, tokenizer, prompt_lst, k=3, alpha=0.65, model_name_or_path=model_name_or_path,
                            max_length=128,  n=3, beta=0.9, sw_coeff=1,punctuations=[],stop_words=[])


    generation_lst = tokenizer.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)
    print(generation_lst)
