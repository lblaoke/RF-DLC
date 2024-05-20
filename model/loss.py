import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp

def cls_weight_init(cls_num,option='linear',norm=False):
    cls_num_tensor = torch.tensor(cls_num,dtype=torch.float,requires_grad=False)

    if option=='linear':
        cls_weights = 1./cls_num_tensor
    elif option=='effective_num':
        cls_weights = 0.0001/(1.-torch.pow(0.9999,cls_num_tensor))
        return cls_weights/cls_weights.sum()*len(cls_num)
    elif option=='sqrt':
        return 1./cls_num_tensor.sqrt()
    elif option=='log':
        return 1./cls_num_tensor.log()
    else:
        return None

    return cls_weights/cls_weights.sum()*len(cls_num) if norm else cls_weights

class VariationalE(nn.Module):
    def __init__(self,cls_num_list,reweight_epoch,alpha,tau,u_value,max_m=0.5):
        super(VariationalE,self).__init__()
        self.reweight_epoch = reweight_epoch
        self.alpha = alpha
        self.tau = tau
        self.u_value = u_value

        m_list = 1./np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list*(max_m/np.max(m_list))
        m_list = torch.tensor(m_list,dtype=torch.float,requires_grad=False)
        self.m_list = m_list
        self.cls_weights = cls_weight_init(cls_num_list)
        self.cls_idx = torch.arange(len(m_list))

    def to(self,device):
        super().to(device)
        self.m_list = self.m_list.to(device)
        self.cls_idx = self.cls_idx.to(device)
        if self.cls_weights is not None:
            self.cls_weights = self.cls_weights.to(device)
        return self

    def _hook_before_epoch(self,epoch):
        if epoch<=self.reweight_epoch:
            self.applied_cls_weights = None
        else:
            self.applied_cls_weights = self.cls_weights

    def _modify_logits(self,x,y):
        index = torch.zeros_like(x,dtype=torch.uint8,device=x.device)
        index.scatter_(1,y.data.view(-1,1),1)
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None,:],index_float.transpose(0,1))
        batch_m = batch_m.view((-1,1))
        x_m = x-30*batch_m
        return torch.where(index,x_m,x)

    def forward(self,epoch,x,y,utility_option="none",extra_info=None):
        # compute utility
        if utility_option=="tail_sensitive" and epoch>self.reweight_epoch:
            batch_cls_idx = torch.stack([self.cls_idx]*len(y))
            relative_class_idx = batch_cls_idx-y.unsqueeze(-1)
            utility = torch.ones_like(batch_cls_idx)*self.u_value*(relative_class_idx<0)
        if utility_option=="bird2plane" and epoch>self.reweight_epoch:
            batch_cls_idx = torch.stack([self.cls_idx]*len(y))
            utility = torch.zeros_like(batch_cls_idx)
            utility[:,0] = -1
            utility *= (y.unsqueeze(-1)==2)
        if utility_option=="car2animal" and epoch>self.reweight_epoch:
            batch_cls_idx = torch.stack([self.cls_idx]*len(y))
            utility = torch.zeros_like(batch_cls_idx)
            utility[:,3:8] = -1
            utility *= (y.unsqueeze(-1)==1)|(y.unsqueeze(-1)==9)

        # compute cross-entropy terms
        losses,weights = [],[]
        for i in range(extra_info["num_particle"]):
            logit = self._modify_logits(extra_info["logits"][i],y)
            # logit = extra_info["logits"][i]

            ce_loss = F.cross_entropy(logit,y,weight=self.applied_cls_weights)
            # ce_loss = F.cross_entropy(logit,y)

            if utility_option!="none" and epoch>self.reweight_epoch:
                log_p = F.log_softmax(logit,dim=1)
                utility_loss = -torch.sum(self.applied_cls_weights*utility*log_p)/len(log_p)
                ce_loss += utility_loss/self.alpha

            for params in extra_info["weights"][i].parameters():
                weights.append(params)

            losses.append(ce_loss)

        # add covariance term
        w_avg = sum(weights)/len(weights)
        w_sub = [wi-w_avg for wi in weights]
        var = [wi*wi for wi in w_sub]
        var = sum(var)/len(var)
        logdet = torch.mean(torch.log(var))

        ret = sum(losses)/len(losses)
        if extra_info["num_particle"]>1:
            ret -= exp(-epoch/self.tau)*logdet/2
        return ret
