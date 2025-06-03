import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import torch as t
from pathlib import Path

from utils import (
    set_seed,
    get_alpaca,
    get_harmbench,
    get_jailbreakbench,
    get_short_model_name,
    load_model_tokenizer,
    sample_equal_len
)


def load_harmful_instructions(dataset_type: str) -> list[str]:
    """Load harmful instructions from the specified dataset type."""
    if dataset_type == 'jbb':
        return get_jailbreakbench()['inst'].tolist()
    elif dataset_type == 'hb':
        return get_harmbench()['inst'].tolist()
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_harmless_instructions() -> list[str]:
    """Load harmless instructions from Alpaca dataset."""
    return get_alpaca()['inst'].tolist()


def process_dataset(dataset_type: str, tokenizer, short_model_name: str, output_dir: Path) -> None:
    """Process a single dataset type and save the results."""
    harmful_insts = load_harmful_instructions(dataset_type)
    harmless_insts = load_harmless_instructions()
    harmless_insts = sample_equal_len(harmful_insts, harmless_insts, tokenizer)
    
    print(f"{dataset_type}: {len(harmful_insts)} harmful, {len(harmless_insts)} harmless")
    
    output_file = output_dir / f"{short_model_name}_{dataset_type}-insts.pt"
    t.save({
        'harmful': harmful_insts,
        'harmless': harmless_insts,
    }, output_file)


def setup_environment(model_path: str, seed: int) -> tuple:
    """Setup the environment and load necessary components."""
    set_seed(seed)
    tokenizer = load_model_tokenizer(model_path, tokenizer_only=True)
    short_model_name = get_short_model_name(model_path)
    
    print(f"Model: {short_model_name}")
    
    return tokenizer, short_model_name


@click.command()
@click.option(
    "--model-path", 
    type=str, 
    help="Path to the model"
)
@click.option(
    "--seed", 
    type=int, 
    default=2024,
    help="Random seed for reproducibility"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./outputs/insts",
    help="Directory to save output files"
)
def main(model_path: str, seed: int, output_dir: Path) -> None:
    """Generate and save instruction datasets for model analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, short_model_name = setup_environment(model_path, seed)
    
    for dataset_type in ['jbb', 'hb']:
        process_dataset(dataset_type, tokenizer, short_model_name, output_dir)


if __name__ == "__main__":
    main()
