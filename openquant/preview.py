from . import LinearQuantizer


class GPTQ(LinearQuantizer):
    pass
    # def __init__(
    #     self,
    #     qcfg: QuantConfig,
    #     name: str,
    #     module: torch.nn.Linear,
    #     execution_device: torch.device,
    #     cache_device: torch.device,
    #     damp_percent: float = 0.01,
    #     damp_auto_increment: float = 0.0025,
    #     static_groups: bool = False,
    # ):
    #     super().__init__(qcfg, name, module, execution_device, cache_device)

    #     assert 0 < damp_percent < 1
    #     assert 0 < damp_auto_increment < 1
    #     self.damp_percent = damp_percent
    #     self.damp_auto_increment = damp_auto_increment
    #     self.static_groups = static_groups

    #     self.h: torch.Tensor = torch.zeros(
    #         (self.inp_dim, self.inp_dim),
    #         device=cache_device,
    #         requires_grad=False,
    #     )
    #     self.num_samples = 0

    # @torch.inference_mode()
    # def forward(self, x: torch.Tensor):
    #     batch_size, _inp_dim = x.shape

    #     x = x.to(self.execution_device)
    #     h = self.h.to(self.execution_device)

    #     h *= self.num_samples / (self.num_samples + batch_size)
    #     self.num_samples += batch_size
    #     x = math.sqrt(2 / self.num_samples) * x.float()
    #     h += x.t().matmul(x)

    #     self.h = h.to(self.cache_device)
    #     x.cpu()

    #     raise ForwardPassEarlyStop()

    # @torch.inference_mode()
    # def quantize_module(self):
    #     w = self.module.weight.data.clone().to(self.execution_device)
    #     h = self.h.to(self.execution_device)
    #     q = torch.zeros_like(w)
    #     scale, zero = self.qcfg.compute_qparams(w)

    #     dead = torch.diag(h) == 0
    #     h[dead, dead] = 1
    #     w[:, dead] = 0

    #     perm = torch.argsort(torch.diag(h), descending=True)
    #     invperm = torch.argsort(perm)
    #     w = w[:, perm]
    #     h = h[perm][:, perm]

    #     losses = torch.zeros_like(w)

    #     diag = torch.arange(self.columns, device=self.execution_device)
    #     damp_percent = self.qcfg.damp_percent
    #     while 0 < damp_percent < 1:
    #         try:
    #             h[diag, diag] += damp_percent * torch.mean(torch.diag(h))
    #             h = torch.cholesky(h)
    #             h = torch.cholesky_inverse(h)
    #             h = torch.cholesky(h, upper=True)
    #             h_inverse = h
    #             break
    #         except torch._C._LinAlgError as e:
    #             damp_percent += self.qcfg.damp_auto_increment
    #             assert 0 < damp_percent < 1

    #     for i1 in range(0, self.inp_dim, self.qcfg.group_size):
    #         i2 = min(i1 + self.qcfg.group_size, self.inp_dim)
    #         count = i2 - i1

    #         for i in range(count):
    #             local_w = w[:, i1 + i : i1 + i + 1]
    #             local_q = self.qcfg.quantize_tensor(local_w, scale, zero).flatten()

    #             d = h_inverse1[i, i]

    #             if (i1 + i) % self.qcfg.group_size == 0:
    #                 self.qcfg.compute_qparams(
    #                     w[:, (i1 + i) : (i1 + i + self.qcfg.group_size)],
    #                     weight=True,
    #                 )

    #             if ((i1 + i) // self.qcfg.group_size) - now_idx == -1:
    #                 scale.append(self.quantizer.scale)
    #                 zero.append(self.quantizer.zero)
    #                 now_idx += 1

    #             q1[:, i] = q
    #             losses1[:, i] = (w - q) ** 2 / d**2

    #             err1 = (w - q) / d
    #             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    #             Err1[:, i] = err1

    #         Q[:, i1:i2] = Q1
    #         Losses[:, i1:i2] = Losses1 / 2

    #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    #     return q, scale, zero


class DynamicFloat(LinearQuantizer):
    pass


class StaticFloat(LinearQuantizer):
    pass
