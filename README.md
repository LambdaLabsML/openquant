# openquant

Simple quantization, compatible with vllm/sglang.

```bash
git clone https://github.com/LambdaLabsML/openquant.git
cd openquant
python compress_fp8.py -m Qwen/Qwen3-32B
vllm serve Qwen3-32B-FP8
```

Model/quantization support:

| Model     | fp8  | awq | 
| --------  | ---- | --- |
| Qwen3     | ✅    | ✅  |
| Qwen3 MoE | ✅    |    |
| Llama 3   | ✅    | ✅  |
| Llama 4   | ✅    |    |
| Gemma 3   | ✅    | ✅ |
| Mistral   | ✅    | ✅ |

For contributing new model architectures, see examples in [openquant/models.py](openquant/models.py).

## Everything about `fp8` quantization

```bash
python compress_fp8.py -m Qwen/Qwen3-32B
```

### fp8: Why?

tl;dr:
1. `model size * 0.5`
2. `throughput * 1.2ish` (with a lot of caveats)

Models today are usually trained in `bf16`, which is a decimal number stored in 16 bits (2 bytes). At the billions of parameter scale, these add up VERY quickly. The main reason for quantizing a model from `bf16` to `fp8` is **memory reduction.**

> For example [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) has 70 billion parameters, which at `bf16` is 140 billion bytes or 140 GB of data. A single H100 GPU has 80GB of GPU RAM, so you'd need at LEAST 2xH100 to serve it, but likely more for kv cache space. If you halve the number of bytes, it would only take 70 GB, enabling it to comfortably fit on 2xH100s, and just fit barely on 1xH100.

Starting with NVIDIA H100 GPU, GPUs have *hardware support* for 8 bit floating point numbers (`fp8`), meaning **`fp8` performance is >= `bf16` performance** (mostly). This performance gain comes from a couple of reasons:

1. Model takes less GPU ram => more space for kv cache. Modern inference libraries (like vllm/sglang) will have higher/more stable performance with more kv cache
2. Model parameters are half as big => less GPU memory bandwidth/more data can fit in cache
3. Depending on the GPU, fp8 flops are just higher than bf16 flops. E.g. See [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/); bfloat16 has ~2k teraflops and fp8 has ~4k teraflops


### fp8: How?

#### fp8 bit format

First some facts:

1. Model parameters are typically stored using `torch.float8_e4m3fn`
    1. This format has `1` sign bit, `4` bits for exponent, and `3` bits for mantissa
    2. Values can be between `[-448, +448]`
    3. 256 representable values
    3. No support for infinity

Here are some sample random numbers at f32/bf16/fp8 (you can see the precision loss as store in less bits):

```python
>>> q = torch.randn(18); q
tensor([-0.272713, -0.222072, -0.491148,  0.589126,  0.489998,  1.777003])
>>> q.to(dtype=torch.bfloat16)
tensor([-0.273438, -0.221680, -0.490234,  0.589844,  0.490234,  1.773438])
>>> q.to(dtype=torch.float8_e4m3fn)
tensor([-0.281250, -0.218750, -0.500000,  0.562500,  0.500000,  1.750000])
```

And here is how all the representable values are distributed (notice how there are waaaaay more values closer to 0!
):

![image](https://github.com/user-attachments/assets/a2eefa93-5f0a-4154-b78a-0964403ff57f)

So this leads us with two questions for quantization:

1. `bf16` can store values between `[-3.38953e+38, +3.38953e+38]`, how do we fit that into fp8 range of `[-448, +448]`?
2. How do we take advantage of the distribution of values in fp8?

#### Scaling to lower precision loss & handle large values

When quantizing a tensor from `bf16` to `fp8`, we don't just convert it to the dtype like I showed above.

Instead we do the following:
1. Compute the largest value of the tensor (the scale)
2. Divide the tensor by the scale (so the values are between min value and max value)
3. Store both the quantized tensor & the scale

We need to compute & store this scale to handle values that are larger than the range that fp8 can store (-448 to 448).

Let's see this in action:

TODO

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
