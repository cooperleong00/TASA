import click
import json
import sys
import os
import os.path as osp
import re
import pandas as pd
import torch as t
from collections import defaultdict
from functools import partial
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    get_template,
    load_model_tokenizer,
    get_insts,
    parent_dir,
    save_jsonl,
)

t.set_grad_enabled(False)

def get_eos_token_id(model_name, tokenizer):
    if 'llama-3' in model_name.lower():
        eos_token = "<|eot_id|>"
    elif 'gemma' in model_name.lower() or 'openchat' in model_name.lower():
        eos_token = "<|end_of_turn|>"
    else:
        eos_token = tokenizer.eos_token
    return tokenizer.convert_tokens_to_ids(eos_token)

def idxs_to_dict(idxs):
    output = defaultdict(list)
    for idx in idxs:
        output[idx[0]].append(idx[1])
    return output


def extract_response(text: str, model: str) -> str | None:
    model = model.lower()
    
    if 'llama-3' in model:
        pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?=<\|eot_id\|>|$)'
    elif 'gemma' in model:
        pattern = r'<start_of_turn>model\n(.*?)(?=<end_of_turn>|$)'
    elif 'mistral' in model:
        pattern = r'\[INST\].*?\[/INST\]\s*(.*?)(?=<s>|$)'
    elif 'llama-2' in model:
        pattern = r'\[INST\].*?\[/INST\]\s*(.*?)(?=<s>|$)'
    else:
        raise NotImplementedError(f"Model {model} not supported")
    
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

@click.command()
@click.option('--model-path', required=True, help='Path to the model')
@click.option('--dataset', default='jbb', type=click.Choice(['hb', 'jbb']), help='Type of the dataset')
@click.option('--patched-rate', default=0.1, type=float, help='Rate of heads to patch')
@click.option('--max-new-tokens', default=512, help='Maximum number of new tokens to generate')
def main(model_path, dataset, patched_rate, max_new_tokens):
    model_name = model_path.split('/')[-1]
    inst_template, template_begin_len, template_end_len = get_template(model_name)
    formatting = lambda inst: inst_template.format(input=inst)
    model, tokenizer = load_model_tokenizer(model_path)
    get_token_len = lambda x: len(tokenizer(x, add_special_tokens=False).input_ids)
    
    harmful_insts, harmless_insts = get_insts(model_name, dataset)

    patching_scores = pd.read_csv(osp.join(parent_dir, f'outputs/patching-scores/{model_name}_patching-scores.csv'))
    patching_scores['normalized_logit_diff'] = (patching_scores['corrupted_refusal_logit'] - patching_scores['harmless_refusal_logit']) /\
          (patching_scores['harmful_refusal_logit'] - patching_scores['harmless_refusal_logit'])
    patching_scores = patching_scores.groupby(['layer', 'head']).mean().reset_index()
    patching_scores = patching_scores.sort_values('normalized_logit_diff', ascending=True).reset_index(drop=True)
    sorted_heads = patching_scores[['layer', 'head']].values.tolist()

    eos_token_id = get_eos_token_id(model_name, tokenizer)
    num_patched_heads = int((model.cfg.n_heads * model.cfg.n_layers) * patched_rate)
    outputs = []

    for sample_idx in trange(len(harmful_insts)):
        clean_input = formatting(harmless_insts[sample_idx])
        _, clean_cache = model.run_with_cache(clean_input, prepend_bos=False)

        corrupted_input = formatting(harmful_insts[sample_idx])
        #_, harmful_cache = model.run_with_cache(corrupted_input, prepend_bos=False)

        top_heads = idxs_to_dict(sorted_heads[:num_patched_heads])

        input_len = get_token_len(corrupted_input)
        def new_hook(corrupted_activation, hook, top_heads, clean_activation):
            layer = int(hook.name.split(".")[1])
            if layer in top_heads:
                heads = top_heads[layer]
                corrupted_activation[:, input_len-template_end_len:input_len, heads, :] =\
                      clean_activation[hook.name][:, input_len-template_end_len:input_len, heads, :]
            return corrupted_activation

        current_hook = partial(
            new_hook,
            top_heads=top_heads,
            clean_activation=clean_cache,
        )

        if model.cfg.n_key_value_heads is not None and model.cfg.n_key_value_heads != model.cfg.n_heads:
            names_filter = lambda x: x.endswith("repeat_v")
        else:
            names_filter = lambda x: x.endswith("concated_v")

        model.add_hook(names_filter, current_hook)

        output = model.generate(
            formatting(harmful_insts[sample_idx]),
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            eos_token_id=eos_token_id,
            verbose=False,
            prepend_bos=False,
        )
        model.remove_all_hook_fns()

        outputs.append({
            "goal": harmful_insts[sample_idx],
            "jailbreak_resps": [extract_response(output, model_name)],
        })


    output_file = f'./outputs/jailbreak_resps/temp_patching/{model_name}_{dataset}_pr-{patched_rate}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_jsonl(outputs, output_file)


if __name__ == '__main__':
    main()