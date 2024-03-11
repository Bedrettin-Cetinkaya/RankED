import torch
from ..builder import LOSSES
import time


@LOSSES.register_module()
class RankLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, nms_grad=1, delta=0.3,eps=1e-10): 
 
        B,C,W,H = logits.size()
        logits = logits.view(B,-1)
        targets = targets.view(B,-1)
        classification_grads=torch.zeros(logits.shape).cuda()
        #Filter fg logits
        fg_labels = (targets > 0)
        fg_logits = logits[fg_labels]
        fg_num = len(fg_logits)
        #fg_targets = targets[fg_labels]

        if fg_num != 0:

            #Do not use bg with scores less than minimum fg logit
            #since changing its score does not have an effect on precision
            threshold_logit = torch.min(fg_logits)-delta
            #threshold_logit = 0.01
            #Get valid bg logits

            relevant_bg_labels=((torch.logical_not(fg_labels))&(logits>=threshold_logit))
            relevant_bg_logits=logits[relevant_bg_labels]

            ranking_error=torch.zeros(fg_num).cuda()
            fg_grad=torch.zeros(fg_num).cuda()
           
            fg_logits_sorted, sorted_indices =torch.sort(fg_logits)
            #Loops over each positive following the order
                        
            fg_relations = fg_logits - fg_logits_sorted[:,None]
            fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
            
            bg_relations = relevant_bg_logits - fg_logits_sorted[:,None]
            bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)

            rank_pos=torch.sum(fg_relations, axis = 1)
            FP_num=torch.sum(bg_relations, axis = 1)

            rank=rank_pos+FP_num
            ranking_error = FP_num/rank
            #current_sorting_error = torch.sum(fg_relations*(1-fg_targets), axis = 1)/rank_pos
            
            #multiLabel_relations = (fg_targets >= fg_targets[sorted_indices,None])
            #target_sorted_order = multiLabel_relations * fg_relations

            #rank_pos_target = torch.sum(target_sorted_order,axis=1)
            #target_sorting_error= torch.sum(target_sorted_order*(1-fg_targets),axis=1)/rank_pos_target
            #sorting_error = current_sorting_error - target_sorting_error
            
            FP_num_check = FP_num > eps
            fg_grad -= ranking_error * FP_num_check.long()

            relevant_bg_grad =  torch.sum((bg_relations*(ranking_error/(FP_num+eps))[:,None]),axis=0)

            #missorted_examples = (~ multiLabel_relations) * fg_relations
            
            #sorting_pmf_denom = torch.sum(missorted_examples, axis=1)
            #sorting_pmf_denom_check = sorting_pmf_denom > eps

            #fg_grad -= sorting_error * sorting_pmf_denom_check.long()
            fg_grad[torch.arange(fg_grad.size(0))],fg_grad[sorted_indices] = fg_grad[sorted_indices],fg_grad[torch.arange(fg_grad.size(0))]
            #fg_grad += torch.sum((missorted_examples*(sorting_error/(sorting_pmf_denom+eps))[:,None]),axis=0)
                        
            #aLRP with grad formulation fg gradient
            classification_grads[fg_labels]= fg_grad
            classification_grads[relevant_bg_labels]= relevant_bg_grad 
            
            classification_grads /= fg_num 
            classification_grads = classification_grads.view(B,C,W,H)
            #classification_grads *= nms_grad
            ctx.save_for_backward(classification_grads)

        else:
            ranking_error = torch.zeros((2,1)).sum()
            classification_grads = classification_grads.view(B,C,W,H)
            ctx.save_for_backward(classification_grads)
        return ranking_error.mean()

    @staticmethod
    def backward(ctx, out_grad1):
        g1, =ctx.saved_tensors
        return g1*out_grad1, None, None
