import torch
import time
from ngram_model.fsd import NGram
from ngram_model.fsd_vec import HiddenSoftNGram

@torch.no_grad()
def fsd_decoding(model, tokenizer, prompt_lst, k, alpha, max_length, n=3, beta=0.9, sw_coeff=0., stop_words_ids=[], eos_token_id = None, early_stop = False):
    """
    - k: top-k candidate words are selected, default 3 
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - beta: the smoothness of n-gram models, default 0.9
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """
    encoded_prompt = tokenizer(prompt_lst, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(model.device)

    attention_mask = model._prepare_attention_mask_for_generation(input_ids, tokenizer.pad_token_id,
                                                                  tokenizer.eos_token_id)
    prompt_len = torch.sum(attention_mask, dim=1)
    ng_list = []
    for i, inputs in enumerate(input_ids):
        ng = NGram(inputs.tolist()[input_ids.shape[1] - prompt_len[i]:], n, tokenizer, beta, sw_coeff, stop_words_ids)
        ng_list.append(ng)

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id

    batch_size, seqlen = input_ids.size()
    prefix_len = seqlen
    model_kwargs = {}

    pad_nums = input_ids.shape[1]
    model_kwargs["attention_mask"] = attention_mask
    for step in range(max_length-prefix_len):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs,return_dict=True)
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = torch.nn.functional.softmax(next_token_scores, dim=-1)

        #avoid generating eos
        if not early_stop and eos_token_id!=None:
            next_token_scores[:, eos_token_id] = -float("inf")
        #keep top-k in p
        top_k = min(max(k, 1), next_token_scores.size(-1))  # Safety check
        indices_to_remove = next_token_scores < torch.topk(next_token_scores, top_k)[0][..., -1, None]
        next_token_scores = next_token_scores.masked_fill(indices_to_remove, -float("Inf"))

        #penalty calculation
        penalty_list = []
        for i, inputs in enumerate(input_ids):
            a, b = torch.topk(next_token_scores[i], k=k)
            penalty_i= ng_list[i](inputs.tolist()[pad_nums-prompt_len[i]:], b.tolist(), a)
            penalty_list.append(penalty_i.view(1,-1))

        batch_penalty = torch.cat(penalty_list,dim=0)
        batch_penalty = batch_penalty.to(input_ids.device)

        next_token_scores = (1 - alpha) * next_token_scores - alpha * batch_penalty
        next_tokens = torch.argmax(next_token_scores, dim=-1)

        for i, token in enumerate(next_tokens): #not much influence
            ng_list[i].update(token.tolist())
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids


@torch.no_grad()
def fsd_vec_decoding(model, tokenizer, prompt_lst, k, alpha, max_length, n=2, sw_coeff=0., stop_words_ids=[], eos_token_id = None, early_stop = False):
    """
    - k: top-k candidate words are selected, default 3 
    - alpha: (1-alpha)p_lm -(alpha)*penalty
    - max_length: decoding max_length-prompt_length steps
    - n: the order of n-gram models
    - sw_coeff: give stopwords a small penalty (<1) or larger penalty(>1), default 0.
    - stop_words=[]: the list of stopwords. If you use GPT-2, you at least need to add two special tokens ('Ċ' and 'ĊĊ') to avoid grammars errors.
    """
    encoded_prompt = tokenizer(prompt_lst, padding=True, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded_prompt["input_ids"].to(model.device)
    # build the n-gram model
    ng = HiddenSoftNGram(n, input_ids.device, tokenizer, sw_coeff, stop_words_ids)

    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    batch_size, seqlen = input_ids.size()
    prefix_len = seqlen
    model_kwargs = {}
    attention_mask = model._prepare_attention_mask_for_generation(input_ids, tokenizer.pad_token_id,
                                                                  tokenizer.eos_token_id)
    prompt_len = torch.sum(attention_mask, dim=1)
    model_kwargs["attention_mask"] = attention_mask
    for step in range(max_length - prefix_len):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(**model_inputs, return_dict=True,output_hidden_states=True)
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = torch.nn.functional.softmax(next_token_scores, dim=-1)

        # avoid generating eos
        if not early_stop and eos_token_id != None:
            next_token_scores[:, eos_token_id] = -float("inf")
        # keep top-k in p
        top_k = min(max(k, 1), next_token_scores.size(-1))  # Safety check
        indices_to_remove = next_token_scores < torch.topk(next_token_scores, top_k)[0][..., -1, None]
        next_token_scores = next_token_scores.masked_fill(indices_to_remove, -float("Inf"))

        # penalty calculation
        batch_penalty = ng(input_ids, outputs.hidden_states)
        batch_penalty = batch_penalty.to(input_ids.device)
        next_token_scores = (1 - alpha) * next_token_scores - alpha * batch_penalty
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
        )
    return input_ids
