import torch
import torch.distributed as dist


class GatherLoss(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
        )

class VariedShapeGatherLoss(torch.autograd.Function):
    """An autograd function that performs allgather on varied length tensor."""

    @staticmethod
    def forward(ctx, q, rank, ws): 
        """
        Gathers tensor arrays of different lengths across multiple gpus
        
        Parameters
        ----------
            q : tensor array
            ws : world size
            device : current gpu device
            
        Returns
        -------
            all_q : list of gathered tensor arrays from all the gpus

        """
        ctx.rank = rank
        ctx.batch_size = q.shape[0]
        device = q.device
        # q 的 维度是 m * d , 只会在第一维上不同， 所以只比较第一维
        local_size = torch.tensor(q.size(0), device=device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(all_sizes)
        ctx.all_sizes = torch.tensor(all_sizes).cumsum_(dim=0).tolist()
        size_diff = max_size.item() - local_size.item()
        if size_diff:
            padding = torch.zeros(((size_diff,) + q.size()[1:]), device=device, dtype=q.dtype)
            q = torch.cat((q, padding))

        all_qs_padded = [torch.zeros_like(q, dtype=q.dtype) for _ in range(ws)]
        dist.all_gather(all_qs_padded, q)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return torch.cat(all_qs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        start = ctx.all_sizes[ctx.rank - 1] if ctx.rank > 0 else 0
        end = ctx.all_sizes[ctx.rank]
        return (
            grad_output[start : end],
            None,
            None,
        )