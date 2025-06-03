import os
import sys
from functools import partial
from itertools import product
from pathlib import Path

import click
import pandas as pd
import torch as t
from tqdm import tqdm
from transformer_lens import HookedTransformerKeyValueCache

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    get_template,
    load_model_tokenizer,
    get_insts,
)

t.set_grad_enabled(False)


def get_last_states(model, insts, formatting_fn):
    """Extract the last hidden states for a list of instructions."""
    states = []
    for inst in tqdm(insts, desc="Computing last states"):
        formatted_inst = formatting_fn(inst)
        state = model(formatted_inst, stop_at_layer=model.cfg.n_layers, prepend_bos=False)[0, -1]
        states.append(state)
    return states


def compute_refusal_direction(harmful_states, harmless_states):
    """Compute the refusal direction from harmful and harmless states."""
    refusal_state = t.mean(t.stack(harmful_states), dim=0) - t.mean(t.stack(harmless_states), dim=0)
    refusal_direction = refusal_state / t.norm(refusal_state)
    return refusal_direction


def patch_hook(corrupted_activation, hook, index, clean_activation, template_begin_len):
    """Hook function for patching attention head activations."""
    assert len(index) == 2
    layer, head = index
    corrupted_activation[:, template_begin_len:-1, head, :] = clean_activation[hook.name][:, template_begin_len:-1, head, :]
    return corrupted_activation


def prepare_kv_cache(past_kv_cache):
    """Prepare the key-value cache by removing the last token."""
    for i in range(len(past_kv_cache.entries)):
        past_kv_cache.entries[i].past_keys = past_kv_cache.entries[i].past_keys[:, :-1, :, :]
        past_kv_cache.entries[i].past_values = past_kv_cache.entries[i].past_values[:, :-1, :, :]
    past_kv_cache.previous_attention_mask = past_kv_cache.previous_attention_mask[:, :-1]
    past_kv_cache.freeze()


def get_hook_name(model, layer):
    """Get the appropriate hook name based on model configuration."""
    if model.cfg.n_key_value_heads is not None and model.cfg.n_key_value_heads != model.cfg.n_heads:
        return f"blocks.{layer}.attn.hook_repeat_v"
    else:
        return f"blocks.{layer}.attn.hook_concated_v"


def compute_patching_scores(model, harmful_insts, harmless_insts, refusal_direction, 
                          formatting_fn, template_begin_len):
    """Compute patching scores for all attention heads across all samples."""
    layer_heads = list(product(range(model.cfg.n_layers), range(model.cfg.n_heads)))
    total_operations = len(harmless_insts) * len(layer_heads)
    
    results = []
    
    with tqdm(total=total_operations, desc="Computing patching scores") as pbar:
        for sample_idx in range(len(harmless_insts)):
            clean_input = formatting_fn(harmless_insts[sample_idx])
            _, clean_cache = model.run_with_cache(clean_input, prepend_bos=False)
            harmless_refusal_logit = clean_cache[('resid_post', -1)][0, -1] @ refusal_direction

            corrupted_input = formatting_fn(harmful_insts[sample_idx])
            corrupted_tokens = model.to_tokens(corrupted_input, prepend_bos=False)
            past_kv_cache = HookedTransformerKeyValueCache.init_cache(
                model.cfg, model.cfg.device, 1
            )
            _, corrupted_cache = model.run_with_cache(corrupted_tokens, past_kv_cache=past_kv_cache)
            harmful_refusal_logit = corrupted_cache[('resid_post', -1)][0, -1] @ refusal_direction

            prepare_kv_cache(past_kv_cache)

            for layer, head in layer_heads:
                current_hook = partial(
                    patch_hook,
                    index=(layer, head),
                    clean_activation=clean_cache,
                    template_begin_len=template_begin_len,
                )
                
                hook_name = get_hook_name(model, layer)

                patched_output = model.run_with_hooks(
                    corrupted_tokens[:, -1:],
                    fwd_hooks=[(hook_name, current_hook)],
                    stop_at_layer=model.cfg.n_layers,
                    past_kv_cache=past_kv_cache,
                )[0, -1]

                corrupted_refusal_logit = patched_output @ refusal_direction

                results.append((
                    sample_idx, 
                    layer, 
                    head, 
                    harmless_refusal_logit.item(), 
                    harmful_refusal_logit.item(), 
                    corrupted_refusal_logit.item()
                ))

                pbar.update(1)
    
    return results


def save_results(results, model_name, output_dir="./outputs/patching-scores"):
    """Save results to CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results, columns=[
        'sample_idx', 'layer', 'head', 
        'harmless_refusal_logit', 'harmful_refusal_logit', 'corrupted_refusal_logit'
    ])
    
    output_file = output_path / f"{model_name}_patching-scores.csv"
    df.to_csv(output_file, index=False)
    click.echo(f"Results saved to: {output_file}")


@click.command()
@click.option(
    "--model-path", 
    required=True, 
    help="Path to the model"
)
@click.option(
    "--dataset",
    default="jbb",
    type=click.Choice(['jbb', 'hb']),
    help="Dataset to use for instructions"
)
@click.option(
    "--output-dir",
    default="./outputs/patching-scores",
    help="Directory to save output files"
)
def main(model_path, dataset, output_dir):
    """
    Compute attention head patching scores for analyzing refusal mechanisms.
    
    This script analyzes how different attention heads contribute to refusal behavior
    by patching activations between harmful and harmless instructions.
    """
    model_name = model_path.split('/')[-1]
    template, template_begin_len, template_end_len = get_template(model_name)
    formatting_fn = lambda inst: template.format(input=inst)
    model, tokenizer = load_model_tokenizer(model_path)
    
    harmful_insts, harmless_insts = get_insts(model_name, dataset)
    
    if len(harmful_insts) != len(harmless_insts):
        raise ValueError(f"Mismatch in instruction counts: harmful={len(harmful_insts)}, harmless={len(harmless_insts)}")
    
    click.echo(f"Processing {len(harmful_insts)} instruction pairs with model: {model_name}")
    
    harmful_states = get_last_states(model, harmful_insts, formatting_fn)
    harmless_states = get_last_states(model, harmless_insts, formatting_fn)
    refusal_direction = compute_refusal_direction(harmful_states, harmless_states)
    
    results = compute_patching_scores(
        model, harmful_insts, harmless_insts, refusal_direction, 
        formatting_fn, template_begin_len
    )
    
    save_results(results, model_name, output_dir)


if __name__ == "__main__":
    main()
