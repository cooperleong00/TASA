The file `llama-2_patching.json` contains handcrafted pairwise data for activation patching. While this dataset is not used in the official publication, it was part of our preliminary work. 
For each harmful request, we created a corresponding harmless version by minimally replacing harmful keywords—ensuring that the number of tokens, as determined by the LLaMA-2 tokenizer, remains unchanged.

This dataset enables more fine-grained activation patching analyses. For example:
<img src="patching.png" alt="Patching visualization" width="500">