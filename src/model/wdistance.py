from __future__ import unicode_literals, print_function, division
import torch
import torch.nn.functional as F
from model.biaffine import BiAffineTransform


class WassersteinDistance(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_w = config.lambda_w
        self.rel_attn = BiAffineTransform(
                            config.affine_dim, config.affine_dim,
                            n_label=1,
                            h_bias=config.bias, s_bias=config.bias,
                            init_weight_value=1.0, init_bias_value=0.0)

    def __call__(
        self,
        source,
        target,
        source_mask,
        target_mask,
    ):
        cost = self.compute_flow_cost(
                    source=source,
                    source_mask=source_mask,
                    target=target,
                    target_mask=target_mask,
                    reduction=True
                )
        return cost*self.lambda_w

    def compute_flow_cost(self, source, source_mask, target, target_mask, reduction=True):
        cost_mx = source.unsqueeze(dim=-3) - target.unsqueeze(dim=-2)
        cost_mx = torch.norm(cost_mx, p=2, dim=-1, keepdim=True).squeeze(-1)
        cost_mask = source_mask.unsqueeze(dim=-2) * target_mask.unsqueeze(dim=-1)
        tr_mx = self.rel_attn(target, source)
        tr_mx = tr_mx.permute(2,0,1)
        tr_mx.masked_fill_(cost_mask==0, -1e14)
        tr_prob = F.softmax(tr_mx, dim=-1)
        flow_cost = tr_prob*cost_mx
        if reduction:
            nL1 = torch.sum(target_mask, dim=-1, keepdim=True)
            nL2 = torch.sum(source_mask, dim=-1, keepdim=True)
            flow_cost = torch.sum(flow_cost, dim=(1,2), keepdim=True) / (nL1*nL2) # average over batch and src
            flow_cost = torch.mean(flow_cost)
        else:
            # _, nL1, nL2 = cost_mx.shape
            flow_cost = torch.sum(flow_cost, dim=(1,2), keepdim=True) #/ (nL1*nL2) # average over batch and src
            flow_cost = flow_cost.squeeze(dim=-1)
        return flow_cost
