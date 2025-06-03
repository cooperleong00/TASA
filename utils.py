import os
import os.path as osp
import random
from typing import List, Tuple, Union, Dict

import json
import pandas as pd
import torch as t
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from transformer_lens import HookedTransformer

parent_dir = os.path.dirname(os.path.abspath(__file__))

def set_seed(seed: int):
    transformers.set_seed(seed)

def save_jsonl(data: List[Dict], file_path: str):
    assert isinstance(data, list) and all(isinstance(d, dict) for d in data)
    with open(file_path, "w") as f:
        for d in data:
            json.dump(d, f)
            f.write("\n")

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def get_alpaca():
    path = "data/alpaca_cleaned/alpaca_data_cleaned_uncersored.jsonl"
    path = osp.join(parent_dir, path)
    data = load_jsonl(path)
    insts = [d['messages'][1]['content'].replace("\nInput:", ":") for d in data]
    resps = [d['messages'][2]['content'] for d in data]
    df = pd.DataFrame({"inst": insts, "resp": resps})
    return df

def get_harmbench():
    path = "data/harmbench_behaviors_text_test_standard.csv"
    path = osp.join(parent_dir, path)
    df = pd.read_csv(path)
    df = df.rename(columns={"Behavior": "inst"})
    return df

def get_harmbench_with_targets():
    path = "data/harmbench_with_targets.csv"
    path = osp.join(parent_dir, path)
    df = pd.read_csv(path)
    return df

def get_jailbreakbench():
    path = "data/jbb_harmful-behaviors_no-harmbench.csv"
    path = osp.join(parent_dir, path)
    df = pd.read_csv(path).reset_index(drop=True)
    df = df.rename(columns={"Goal": "inst", 'Target': 'target'})
    return df


def get_short_model_name(model_name: str) -> str:
    model_name = model_name.lower()
    if 'llama-3' in model_name:
        return 'llama3'
    elif 'llama-2' in model_name:
        return 'llama2'
    elif 'gemma-2' in model_name:
        return 'gemma2'
    elif 'mistral' in model_name:
        return 'mistral'
    else:
        raise ValueError(f"Model {model_name} not supported.")


def get_official_path(model_name: str) -> str:
    model_name = model_name.lower()
    if 'llama' in model_name:
        model_path = f'meta-llama/{model_name}'
    elif 'gemma' in model_name:
        model_path = f'google/{model_name}'
    elif 'mistral' in model_name:
        model_path = f'mistralai/{model_name}'
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model_path


def get_insts(model_name: str, _type: str) -> Dict[str, List[str]]:
    assert _type in ['jbb', 'hb']
    short_model_name = get_short_model_name(model_name)
    path = osp.join(parent_dir, f'outputs/insts/{short_model_name}_{_type}-insts.pt')
    insts = t.load(path, weights_only=True)
    return insts['harmful'], insts['harmless']


def get_template(model_name: str) -> Tuple[str, int, int]:
    model_name = model_name.lower()
    if 'llama-3' in model_name:
        template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        template_begin_len = 5
        template_ending_len = 5
    elif 'gemma-2' in model_name:
        template = "<bos><start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n"
        template_begin_len = 4
        template_ending_len = 5
    elif 'llama-2' in model_name:
        template = "<s>[INST] {input} [/INST] "
        template_begin_len = 5
        template_ending_len = 5
    elif 'mistral' in model_name:
        template = "<s>[INST] {input} [/INST]"
        template_begin_len = 5
        template_ending_len = 4
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return template, template_begin_len, template_ending_len


def load_model_tokenizer(model_path: str, tokenizer_only=False) -> Union[PreTrainedTokenizer, Tuple[HookedTransformer, PreTrainedTokenizer]]:
    model_name = model_path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer_only:
        return tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=t.bfloat16)

    model = HookedTransformer.from_pretrained_no_processing(model_name=get_official_path(model_name), hf_model=model, tokenizer=tokenizer, dtype=t.bfloat16, move_to_device=True, device='cuda')
    return model, tokenizer


def get_tokens_len(x: str, tokenizer: PreTrainedTokenizer) -> int:
    return len(tokenizer(x, add_special_tokens=False).input_ids)


def sample_equal_len(A: List[str], B: List[str], tokenizer: PreTrainedTokenizer) -> List[str]:
    """
    Sample texts from B to each text in A with the same token length.
    """
    B_by_length = {}
    for s in B:
        length = get_tokens_len(s, tokenizer)
        if length not in B_by_length:
            B_by_length[length] = []
        B_by_length[length].append(s)
    
    result = []
    used_strings = set()
    
    for a in A:
        length = get_tokens_len(a, tokenizer)
        if length in B_by_length:
            available = [s for s in B_by_length[length] if s not in used_strings]
            if not available:
                available = B_by_length[length]
            
            sampled = random.choice(available)
            result.append(sampled)
            used_strings.add(sampled)
        else:
            result.append(None) 
    
    return result
