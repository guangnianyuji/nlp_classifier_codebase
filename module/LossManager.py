
from torch.nn import CrossEntropyLoss
from module.loss.focal_loss import FocalLoss
from module.loss.infonce_loss import InfoNCELoss
from module.loss.kl_loss import KLLoss
from module.loss.label_smoothing import LabelSmoothingCrossEntropy
import torch.nn.functional as F
import torch

def focal_loss(scores, targets, weight=None, gamma=2, alpha=0.5):
    bce_loss = F.binary_cross_entropy(scores, targets, reduction='none')
    p_t = torch.exp(-bce_loss)
    alpha_tensor = (1 - alpha) + targets * (2 * alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
    f_loss = alpha_tensor * (1 - p_t) ** gamma * bce_loss

    if weight is not None:
        f_loss = f_loss * weight
    return f_loss


class FocalLossWithDenoise:
    def __init__(self):
        self.soft_target = torch.zeros(2000000, 4).cuda()

    def __call__(self, scores, targets, idx_list, epoch):  

        scores = scores.sigmoid()

        #计算第一个分类头损失
        mask = targets != 2
        scores_masked = scores[mask, :2]
        label_onehot = F.one_hot(targets[mask], num_classes=2).float()
        loss_0 = torch.zeros((targets.size(0), 2)).to(scores.device)
        loss_0[mask, :] = focal_loss(scores_masked, label_onehot)
        #计算第二个分类头损失
        targets[targets > 0] = 1
        label_onehot = F.one_hot(targets, num_classes=2).float()
        # import pdb;pdb.set_trace()
        loss_1 = focal_loss(scores[:,2:], label_onehot)
        if epoch == 0:
            new_label = torch.zeros((targets.size(0), 4))
            new_label[targets == 0, 0] = 1
            new_label[targets == 0, 2] = 1
            new_label[targets == 1, 1] = 1
            new_label[targets == 1, 3] = 1
            new_label[targets == 2, 3] = 1
            self.soft_target[idx_list] = new_label.to(self.soft_target.device)
            if self.soft_target.device != scores.device:
                self.soft_target.to(scores.device)
        if epoch > 1:
            loss_2 = (-(self.soft_target[idx_list] * scores.log())).mean() 
            loss = loss_0 + loss_1 + loss_2
        else:
            loss = loss_0 + loss_1

        #更新软标签
        self.soft_target[idx_list] = 0.5 * self.soft_target[idx_list] + 0.5 * scores.detach()

        return loss

class LossManager(object):
    
    def __init__(self, loss_type, cl_option=False, loss_cl_type='InfoNCE',weights=None):
        self.loss_type = loss_type
        self.cl_option = cl_option
        self.loss_cl_type = loss_cl_type
        # 判断配置的loss类型
        if loss_type == 'focalloss':
            self.loss_func = FocalLoss(weight=weights)
        elif loss_type == 'LabelSmoothingCrossEntropy':
            self.loss_func = LabelSmoothingCrossEntropy()
        elif loss_type == 'FocalLossWithDenoise':
            self.loss_func = FocalLossWithDenoise()
        else:
            self.loss_func = CrossEntropyLoss()
            
        if cl_option:
            if loss_cl_type == 'Rdrop':
                self.loss_cl_func = KLLoss()
            else:
                self.loss_cl_func = InfoNCELoss()


    def compute(self, 
                input_x, 
                target,
                idx_list=None,
                epoch = 0,
                hidden_emb_x=None, 
                hidden_emb_y=None, 
                alpha=0.5):
        """        
        计算loss
        Args:
            input: [N, C]
            target: [N, ]
        """
        if hidden_emb_x is not None and hidden_emb_y is not None:
            if 'denoise' in self.loss_type.lower():
                loss_ce = (1-alpha) * self.loss_func(input_x, target, idx_list)
            else:
                loss_ce = (1-alpha) * self.loss_func(input_x, target)
            # loss_ce = (1-alpha) * self.loss_func(input_x, target)
            weight_etx = 1e+5 if self.loss_cl_type=='Rdrop' else 1
            loss_cl = alpha * weight_etx * self.loss_cl_func(hidden_emb_x, hidden_emb_y)
            loss = loss_ce + loss_cl
            return loss
        else:
            if 'denoise' in self.loss_type.lower():
                loss = self.loss_func(input_x, target, idx_list, epoch)
            else:
                loss = self.loss_func(input_x, target)
            return loss
    

    
    # def compute(self, input, target):
    #     """        
    #     计算loss
    #     Args:
    #         input: [N, C]
    #         target: [N, ]
    #     """
    #     loss = self.loss_func(input, target)
    #     return loss


    # def compute(self, input1, input2, output_pooler1, output_pooler2, target, alpha=0.5):
    #     """        
    #     计算loss
    #     Args:
    #         input: [N, C]
    #         target: [N, ]
    #     """
        
    #     loss_ce = alpha * self.loss_func(input1, target)
    #     loss_nce = (1-alpha) * self.loss_func_nce(output_pooler1, output_pooler2)
    #     # loss = alpha*loss_ce + (1-alpha)*loss_nce
    #     loss = loss_ce + loss_nce
    #     return loss, loss_ce, loss_nce