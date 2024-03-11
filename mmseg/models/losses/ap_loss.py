import torch
from ..builder import LOSSES
import time


@LOSSES.register_module()
class APLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, nms_grad=1, delta=0.4, split = 1): 

        B,C,W,H = logits.size()
        logits = logits.view(B,-1)
        targets = targets.view(B,-1)
        classification_grads=torch.zeros(logits.shape).cuda()
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)

        cls_loss = 0
        if fg_num != 0:

            #Do not use bg with scores less than minimum fg logit
            #since changing its score does not have an effect on precision
            threshold_logit = torch.min(fg_logits)-delta
            #threshold_logit = 0.01
            #Get valid bg logits
            relevant_bg_labels=((torch.logical_not(fg_labels))&(logits>=threshold_logit)) 
            
            relevant_bg_logits=logits[relevant_bg_labels]
            relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
            rank=torch.zeros(fg_num).cuda()
            prec=torch.zeros(fg_num).cuda()
            current_prec=torch.zeros(fg_num).cuda()
            fg_grad=torch.zeros(fg_num).cuda()
            
            max_prec=0
            fg_logits_sorted, sorted_indices =torch.sort(fg_logits)            
            #Loops over each positive following the order
            start = 0
            end = fg_num // split
            for ii in range(split):
                ind1 = torch.arange(sorted_indices[start:end].size(0))
                fg_relations = fg_logits - fg_logits_sorted[start:end,None]
                fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
                
                fg_relations[ind1,sorted_indices[start:end]] = 0
                bg_relations = relevant_bg_logits - fg_logits_sorted[start:end,None]
                bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)
                
                rank_pos=1+torch.sum(fg_relations, axis = 1)
                FP_num=torch.sum(bg_relations, axis = 1)
                
                rank[start:end]=rank_pos+FP_num
                current_prec[start:end] = rank_pos/rank[start:end]
                
                accm_prec, _ = torch.cummax(current_prec.clone(), dim = 0)
                accm_prec_orig = accm_prec.clone()
                
                accm_prec[torch.arange(current_prec.size(0))],accm_prec[sorted_indices] = accm_prec[sorted_indices],accm_prec[torch.arange(current_prec.size(0))]
                diff_max = ( accm_prec == current_prec)
                                
                prec_coef_max = (1 - accm_prec_orig) / ( 1 - current_prec + 1e-20)
                prec_coef [ diff_max == 0] = prec_coef_max[diff_max == 0]
                relevant_bg_grad +=  torch.sum( (bg_relations / rank[start:end,None]) * prec_coef[start:end,None]  , axis = 0)
                fg_grad=-(1-accm_prec)
                prec=accm_prec
                
                start = end
                if ii == split -2:
                  end = fg_num
                else:
                  end *= 2
            classification_grads[fg_labels]= fg_grad
            classification_grads[relevant_bg_labels]= relevant_bg_grad 
            
            classification_grads /= fg_num 
            classification_grads = classification_grads.view(B,C,W,H)
            cls_loss=1-prec.mean()
            ctx.save_for_backward(classification_grads)

        else:
            cls_loss = torch.zeros((2,1)).cuda().sum()
            classification_grads = classification_grads.view(B,C,W,H)
            ctx.save_for_backward(classification_grads)
        return cls_loss

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None
