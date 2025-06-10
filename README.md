# openquant

Simple quantization, compatible with vllm/sglang.

```bash
git clone https://github.com/LambdaLabsML/openquant.git
cd openquant
python compress_causallm_fp8.py -m Qwen/Qwen3-32B
vllm serve Qwen3-32B-FP8
```

## Model support

- [x] Qwen3
- [x] Qwen3 MoE
- [x] Llama 3
- [x] Llama 4
- [x] Gemma 3

### Contributing new model architectures

See examples in [openquant/models.py](openquant/models.py).

## Quantization algorithm support

- [x] fp8
- [x] [AWQ](https://arxiv.org/abs/2306.00978)
- [ ] fp4
- [ ] [GPTQ](https://arxiv.org/abs/2210.17323)
- [ ] GGUF

## fp8

### fp8: Why?

Models today are usually trained in `bf16`, which is a decimal number stored in 16 bits (2 bytes). At the billions of parameter scale, these add up VERY quickly.

> For example [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) has 70 billion parameters, which at `bf16` is 140 billion bytes or 140 GB of data. A single H100 GPU has 80GB of GPU RAM, so you'd need at LEAST 2xH100 to serve it, but likely more for kv cache space.

The main reason for quantizing a model from `bf16` to `fp8` is memory reduction. If you halve the number of bytes, then a 70B model only takes 70 GB, enabling it to comfortably fit on 2xH100s, and just fit barely on 1xH100.

Starting with NVIDIA H100 GPU, GPUs have **hardware support** for 8 bit floating point numbers (`fp8`), meaning `fp8` performance is >= `bf16` performance (mostly).

tl;dr:
1. Halve (approximately) the memory requirements [1]
2. Can be faster than bf16 due to less memory bandwidth

[1] Less memory also means extra space for KV Cache storage, making modern inference libraries more stable.

### fp8: How?

#### fp8 vs bf16 numerical format

- talk about format & range & precision

#### Scaling tensors

1. Computing the scale
2. Quantizing 
2. single float scale
3. tensor of scales

### fp8: Saving an inference compatible model checkpoint

For compatibility with things like VLLM there's a couple things we need to do:

1. Add the `weight_scale` as a parameter to each of the `Linear` layers. This basically means just replace the `Linear` layer with this `PackedLinear`, where `weight` is the `fp8` tensor, and `weight_scale` is the scale.

```python
class PackedLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
```

2. Add a `quantization_config` into the model's config. This will also appear in the `config.json` file in the huggingface repo of the model.

```python
model.config.quantization_config = {
    "quant_method": "fp8",
    "is_checkpoint_fp8_serialized": True,
    "activation_scheme": "dynamic",
    "weight_block_size": ..., # `None` or `[block_size, block_size]`
    "ignored_layers": ..., # list of module names that are not quantized
}
```

And that's all we need to do for vllm!

**NOTE: some models don't support all layers being quantized. For example, vllm does not support the `decoder.mlp.gate` linear layer being quantized in Qwen3 MoE models.**

# License

MIT License

Copyright (c) 2025 Lambda Labs Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.