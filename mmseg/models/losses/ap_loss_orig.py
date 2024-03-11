import torch
from ..builder import LOSSES
import time


@LOSSES.register_module()
class APLoss_orig(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, delta=1): 

        B,C,W,H = logits.size()
        logits = logits.view(B,-1)
        targets = targets.view(B,-1)
        classification_grads=torch.zeros(logits.shape).cuda()       
        #Filter fg logits
        fg_labels = (targets == 1)
        fg_logits = logits[fg_labels]

        fg_num = len(fg_logits)
        #print(fg_num, flush=True)

        cls_loss = 0
        if fg_num != 0:
          
            #Do not use bg with scores less than minimum fg logit
            #since changing its score does not have an effect on precision
            threshold_logit = torch.min(fg_logits)-delta
            #print(threshold_logit)
            #Get valid bg logits
            relevant_bg_labels=((targets==0)&(logits>=threshold_logit))
            relevant_bg_logits=logits[relevant_bg_labels]
            relevant_bg_grad=torch.zeros(len(relevant_bg_logits)).cuda()
            rank=torch.zeros(fg_num).cuda()
            prec=torch.zeros(fg_num).cuda()
            fg_grad=torch.zeros(fg_num).cuda()
            
            max_prec=0                                           
            #sort the fg logits
            order=torch.argsort(fg_logits)
            #Loops over each positive following the order
            
            for ii in order:
                #x_ij s as score differences with fgs
                fg_relations=fg_logits-fg_logits[ii] 
                #Apply piecewise linear function and determine relations with fgs
                fg_relations=torch.clamp(fg_relations/(2*delta)+0.5,min=0,max=1)
                #Discard i=j in the summation in rank_pos
                fg_relations[ii]=0
            
                #x_ij s as score differences with bgs
                bg_relations=relevant_bg_logits-fg_logits[ii]
                #Apply piecewise linear function and determine relations with bgs
                bg_relations=torch.clamp(bg_relations/(2*delta)+0.5,min=0,max=1)
            
                #Compute the rank of the example within fgs and number of bgs with larger scores
                rank_pos=1+torch.sum(fg_relations)
                FP_num=torch.sum(bg_relations)
                #Store the total since it is normalizer also for aLRP Regression error
                rank[ii]=rank_pos+FP_num
                                
                #Compute precision for this example 
                current_prec=rank_pos/rank[ii]
                
                #Compute interpolated AP and store gradients for relevant bg examples
                if (max_prec<=current_prec):
                    max_prec=current_prec
                    relevant_bg_grad += (bg_relations/rank[ii])
                else:
                    relevant_bg_grad += (bg_relations/rank[ii])*(((1-max_prec)/(1-current_prec)))
                
                #Store fg gradients
                fg_grad[ii]=-(1-max_prec)
                prec[ii]=max_prec 

            #aLRP with grad formulation fg gradient
            classification_grads[fg_labels]= fg_grad
            torch.amax(relevant_bg_logits), " " , torch.amin(relevant_bg_logits)," " , torch.histc(relevant_bg_logits, bins=8, min=-2,max=2), " " ,torch.amax(fg_logits), " " , torch.amin(fg_logits)," " , torch.histc(fg_logits,bins=8, min=-2,max=2),flush=True)

            #aLRP with grad formulation bg gradient
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
