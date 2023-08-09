import torch
import torch.nn as nn
from torch.nn import functional as F

def cross_entropy(logit, y):

    batch_size = logit.shape[0]
    softmax_probs = F.softmax(logit, dim=-1)
    log_probs = torch.log(torch.clamp(softmax_probs, min=1e-5, max=1.) )
    one_hot_targets = torch.zeros_like(logit)
    for i, j in enumerate(y):
        one_hot_targets[i, j] = 1
    loss = - torch.sum(one_hot_targets * log_probs) / batch_size
    
    return loss

def negative_cross_entropy(logit, output):
    
    batch_size = logit.shape[0]
    softmax_probs = F.softmax(logit, dim=-1)
    log_probs = torch.log(torch.clamp(1. - softmax_probs, min=1e-5, max=1.) )
    one_hot_targets = torch.zeros_like(logit)
    for i, j in enumerate(y):
        one_hot_targets[i, j] = 1
    loss = - torch.sum(one_hot_targets * log_probs) / batch_size

    return loss

class InfoNCE(nn.Module):

    def __init__(self):
        super(InfoNCE, self).__init__()

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.temperature = 0.07

    def forward(self, image, text):

        batchsize = image.size(0)
        
        output = torch.cat((image, text),0)
        logits = torch.einsum('nc,mc->nm', output, output) / self.temperature
        zero_matrix = torch.zeros((batchsize,batchsize),dtype=torch.bool,device=image.device)
        eye_matrix = torch.eye(batchsize, dtype=torch.bool, device=image.device)

        pos_index = torch.cat((torch.cat((zero_matrix,eye_matrix)),torch.cat((eye_matrix,zero_matrix))),1 )
        neg_index = ~torch.cat((torch.cat((eye_matrix,eye_matrix)),torch.cat((eye_matrix,eye_matrix))),1 )

        pos_logits = logits[pos_index].view(2*batchsize, -1)
        neg_index = logits[neg_index].view(2*batchsize, -1)

        final_logits = torch.cat((pos_logits, neg_index), dim=1)
        labels = torch.zeros(final_logits.shape[0], device=image.device, dtype=torch.long)
        loss = self.cross_entropy(final_logits, labels)

        return loss
