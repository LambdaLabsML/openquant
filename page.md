---
# ---------- Core metadata ----------
title:  "Everything about fp8 quantization"
subtitle: "Boosting inference throughput and reducing memory footprint"
description: >
  Describes the fp8 format, why you would quantize a model to fp8, and how to do it.
keywords:
  - fp8
  - f8
  - quantization
  - pytorch
  - inference

# ---------- Authorship ----------
author-meta: "Corey Lowman"
authors:
  - name: "Corey Lowman"
    affiliations:
      - 1
affiliations:
  - id: 1
    name: "Lambda, Inc."


# ---------- Links shown as buttons ----------
links:
  - label: "Code (GitHub)"
    url: "https://github.com/LambdaLabsML/openquant"
    icon: "code"

# ---------- Footer ----------
owner:  "Lambda, Inc."
year:   2025
license: "MIT"
license_url: "https://opensource.org/license/mit"
---

Table of contents:

- [fp8: Why?](#fp8-why)
- [fp8: How?](#fp8-how)
  - [Note on executing fp8 models](#note-on-executing-fp8-models)
  - [fp8 bit format](#fp8-bit-format)
  - [Quantization - scaling to lower precision loss \& handle large values](#quantization---scaling-to-lower-precision-loss--handle-large-values)
  - [Finer grained scale - weight block size](#finer-grained-scale---weight-block-size)
- [Saving an inference compatible model checkpoint](#saving-an-inference-compatible-model-checkpoint)

# fp8: Why?

tl;dr:

1. `model size * 0.5`
2. `throughput * 1.2ish` (with a lot of caveats)

Models today are usually trained in `bf16`, which is a decimal number stored in 16 bits (2 bytes). At the billions of parameter scale, these add up VERY quickly. The main reason for quantizing a model from `bf16` to `fp8` is **memory reduction.**

> For example [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) has 70 billion parameters, which at `bf16` is 140 billion bytes or 140 GB of data. A single H100 GPU has 80GB of GPU RAM, so you'd need at LEAST 2xH100 to serve it, but likely more for kv cache space. If you halve the number of bytes, it would only take 70 GB, enabling it to comfortably fit on 2xH100s, and just fit barely on 1xH100.

Starting with NVIDIA H100 GPU, GPUs have *hardware support* for 8 bit floating point numbers (`fp8`), meaning **`fp8` performance is >= `bf16` performance** (mostly). This performance gain comes from a couple of reasons:

1. Model takes less GPU ram => more space for kv cache. Modern inference libraries (like vllm/sglang) will have higher/more stable performance with more space for kv cache
2. Model parameters are half as big => less GPU memory bandwidth
3. Depending on the GPU, fp8 FLOPS are just higher than bf16 FLOPS. E.g. See [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/); bfloat16 has ~2k teraFLOPS and fp8 has ~4k teraFLOPS


# fp8: How?

## Note on executing fp8 models

When we talk about fp8 models, we typically only are talking about the **weights being fp8**. The actual execution of the model is still done in `bf16`. So all the **intermediate tensors are still in bf16**, and it's the underlying CUDA kernels that are taking in bf16 tensors and fp8 weights.

**fp8 models still use `bf16` kv cache by default** (since the kv cache stores kv values, which are intermediate tensors).

## fp8 bit format

There are a number of different fp8 formats; the most common is `float8_e4m3fn`. Here are some facts about it:

1. This format has `1` sign bit, `4` bits for exponent (`e4`), and `3` bits for mantissa (`m3`)
2. Values can be between `[-448, +448]`
    1. 256 representable values
3. Cannot represent `infinity` (the `fn` postfix stands for "finite numbers only" - there are other fp8 formats that do support infinity)
4. Can represent `NaN`
5. Model parameters are typically stored using this format (note that `inf` is not usually present in pretrained model parameters)

<details>
    <summary>
        Expand this section to see all the possible fp8_e4m3fn values
    </summary>

```
torch.arange(256, dtype=torch.uint8).view(dtype=torch.float8_e4m3fn).tolist()
```

[0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, nan, -0.0, -0.001953125, -0.00390625, -0.005859375, -0.0078125, -0.009765625, -0.01171875, -0.013671875, -0.015625, -0.017578125, -0.01953125, -0.021484375, -0.0234375, -0.025390625, -0.02734375, -0.029296875, -0.03125, -0.03515625, -0.0390625, -0.04296875, -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375, -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875, -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375, -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0, -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0, -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0, -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0, -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, nan]
</details>

And here is how all the representable values are distributed (notice how there are waaaaay more values closer to 0!
):

![image](https://github.com/user-attachments/assets/a2eefa93-5f0a-4154-b78a-0964403ff57f)

So this leads us with two questions for quantization:

1. `bf16` can store values between `[-3.38953e+38, +3.38953e+38]`, how do we fit that into fp8 range of `[-448, +448]`?
2. How do we take advantage of the distribution of values in fp8?

## Quantization - scaling to lower precision loss & handle large values

Since `bf16` and `fp8` have different ranges, we need to scale the values to fit into the `fp8` range. This scale is based
on the max value of the data at `bf16`, and is roughly computed like:

```python
# NOTE: this will be a single value
scale = x.abs().amax() / 448
```

Then once we have the scale we can quantize the `bf16` tensor:
```python
x_quantized = (x / scale).clamp(min=-448, max=448).to(torch.float8_e4m3fn)
```

And to dequantize (which is essentially done on the fly at runtime inside the CUDA kernels), you do this (noting that you have to store the `scale` values for the forward process):
```python
x_dequantized = x.to(torch.bfloat16) * scale
```

## Finer grained scale - weight block size

Above I showed the scale being a single value, but you can also have it be a tensor. If you look at some popular open source fp8 models they typically use this option.

Why would you do this? To theoretically preserve accuracy, though if the values in your tensor are all relatively close together you won't get much benefit.

Given a weight_block_size of `[128, 128]`, and a tensor of shape `[N, K]`, the scale will be of size `[N // 128, K // 128]`:

E.g. assuming x is 2d, we have the code:

```python
N, K = x.shape
n, k = weight_block_size
x = x.reshape(N // n, n, K // k, k)
scale = x.abs().amax(dim=[1, 3]) / 448
assert scale.shape == torch.Size([N // n, K // k])
```

# Saving an inference compatible model checkpoint

For compatibility with things like VLLM there's a couple things we need to do:

1. Add the `weight_scale` as a parameter to each of the `Linear` layers. This basically means just replace the `Linear` layer with this `PackedLinear` class, where `weight` is the `fp8` tensor, and `weight_scale` is the scale.

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
    "weight_block_size": ..., # `None` or `[int, int]`
    "ignored_layers": ..., # list of module names that are not quantized
}
```

And that's all we need to do for vllm!

**NOTE: some models don't support all layers being quantized. For example, vllm does not support the `decoder.mlp.gate` linear layer being quantized in Qwen3 MoE models.**
