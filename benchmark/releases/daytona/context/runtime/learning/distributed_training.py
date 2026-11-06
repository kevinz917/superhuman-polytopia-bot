"""Explicit sum-normalized LoRA data parallelism; global batch stays fixed."""

def partition_effective_batch(rows, world_size):
    if world_size<1:raise ValueError('positive world size required')
    bins=[[] for _ in range(world_size)];sizes=[0]*world_size
    for row in sorted(rows,key=lambda r:len(r['input_ids']),reverse=True):
        rank=min(range(world_size),key=lambda i:(sizes[i],i))
        bins[rank].append(row);sizes[rank]+=len(row['input_ids'])
    return bins


def sum_gradients(parameters):
    import torch
    import torch.distributed as dist
    # Every rank used the GLOBAL response-token denominator. SUM is required;
    # averaging here would incorrectly divide the update by world_size twice.
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    offset=0
    for p in parameters:
        p.grad=flat[offset:offset+p.numel()].view_as(p);offset+=p.numel()
