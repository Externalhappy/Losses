import torch
import torch.nn as nn
from torch.nn import functional as F

def cross_entropy(logit, output):
    
    batch_size = logit.shape[0]

    one_hot_targets = torch.zeros_like(logit)
    one_hot_targets[torch.arange(batch_size), output.cpu().numpy()] = 1
    
    softmax_probs = F.softmax(logit, dim=-1)
    log_probs = torch.log(torch.clamp(softmax_probs, min=1e-5, max=1.) )
    
    loss = -1/batch_size.shape[0] * (one_hot_targets * log_probs)

    return loss