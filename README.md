# openquant

Simple quantization, compatible with vllm/sglang.

```bash
git clone https://github.com/LambdaLabsML/openquant.git
cd openquant
python compress_causallm_fp8.py -m Qwen/Qwen3-32B
vllm serve Qwen3-32B-FP8/
```

## Model support

- [x] Qwen3
- [x] Qwen3 MoE
- [x] Llama 3
- [x] Llama 4
- [x] Gemma 3

## Quantization algorithm support

- [x] fp8
- [x] AWQ
- [ ] fp4
- [ ] GPTQ
- [ ] GGUF

## Contributing new model architectures

See examples in [openquant/models.py](openquant/models.py).

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