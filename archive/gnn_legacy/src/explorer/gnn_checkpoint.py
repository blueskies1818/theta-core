import torch
import torch.nn as nn
from contextlib import nullcontext

def gnn_forward_with_checkpoint(gnn, features, sources, targets, edge_types, num_nodes):
    """Forward GNN with gradient checkpointing on each layer to save memory.
    
    Uses torch.utils.checkpoint to trade compute for memory.
    Applicable when fine-tuning on CPU with large graphs (100K+ nodes).
    """
    try:
        from torch.utils.checkpoint import checkpoint
    except ImportError:
        # Fallback: no checkpointing (may OOM on large graphs)
        return gnn(features, sources, targets, edge_types, num_nodes)
    
    h = nn.functional.gelu(gnn.input_proj(features))
    
    for layer in gnn.layers:
        # Custom checkpoint function for GAT layer
        def layer_fn(h, layer=layer):
            return layer(h, sources, targets, edge_types, num_nodes)
        h = checkpoint(layer_fn, h, use_reentrant=False)
        
        # Reverse direction
        if gnn.config.bidirectional:
            rev_types = edge_types + (gnn.config.num_edge_types // 2)
            rev_types = rev_types.clamp(0, gnn.config.num_edge_types - 1)
            def rev_fn(h, layer=layer):
                return layer(h, targets, sources, rev_types, num_nodes)
            h = checkpoint(rev_fn, h, use_reentrant=False)
    
    return gnn.out_norm(h)
