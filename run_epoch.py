import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from models.utils import custom_replace
from torchvision.ops.focal_loss import sigmoid_focal_loss


# ASL
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss


def run_epoch(args,model,data,optimizer,epoch,desc,train=False,warmup_scheduler=None, global_model=None, emb_feat=None, clip_model=None, tau=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # pre-allocate full prediction and target tensors
    all_predictions = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_targets = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_masks = torch.zeros(len(data.dataset),args.num_labels).cpu()
    all_image_ids = []
    batch_idx = 0
    loss_total = 0
    unk_loss_total = 0
    if train:
        if args.dataset == 'flair_fed' or args.dataset == 'coco' or args.dataset == 'voc':
            data_loader = data
        else:
            data_loader = tqdm(data,mininterval=0.5,desc=desc,leave=True,ncols=100)
    else:
        data_loader = tqdm(data,mininterval=0.5,desc=desc,leave=True,ncols=100)
    for batch in data_loader:
       

        labels = batch['labels'].float()
        images = batch['image'].float()
        mask = batch['mask'].float()
        
        # Original setting
        mask_in = mask.clone()
        if args.use_global_guide and train:
            with torch.no_grad():
                mask_g = mask_in.clone()
                for idx, m in enumerate(mask_g[0]):
                    mask_g[0][idx] = -1.
                global_pred,_,_ = global_model(images.cuda(),mask_g.cuda(), args.learn_emb_type, emb_feat, clip_model)
                global_pred = global_pred.data.cpu()
                # print(global_pred.shape)
                # print(global_pred)
                global_logits = F.sigmoid(global_pred)
             
                # TODO: (for rebuttal) global pred. masking 
                for idx, m in enumerate(mask_in[0]):
                    if 0.48 <= global_logits[0][idx].item() <= 0.52:
                        # mask this
                        mask_in[0][idx] = -1.
        
        # mask -1, 0, 1 -> assigned become 1, 0, 0
        unk_mask = custom_replace(mask_in,1,0,0)
        all_image_ids += batch['imageIDs']

        ### TODO: CLIP
        # idea 1: label text to replace the label embedding in c_tran => there is a "???" in the scene
        # idea 2: [prompt] [label_text] => can be tuned
        if train:
            pred,int_pred,attns = model(images.cuda(),mask_in.cuda(), args.learn_emb_type, emb_feat, clip_model)
        else:
            for idx, m in enumerate(mask_in[0]):
                mask_in[0][idx] = -1.
            with torch.no_grad():
                pred,int_pred,attns = model(images.cuda(),mask_in.cuda(), args.learn_emb_type, emb_feat, clip_model)

        if args.dataset == 'cub':
            class_label = batch['class_label'].float()
            concept_certainty = batch['concept_certainty'].float()

            class_label_onehot = torch.zeros(class_label.size(0),200)
            class_label_onehot.scatter_(1,class_label.long(),1)

            labels = torch.cat((labels,class_label_onehot),1)
            loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')
            loss = (unk_mask.cuda()*loss).sum()/unk_mask.detach().sum().item()

            aux_loss =  F.binary_cross_entropy_with_logits(int_pred.view(labels.size(0),-1),labels.cuda(),reduction='none')
            aux_loss = (unk_mask.cuda()*aux_loss).sum()/unk_mask.detach().sum().item()

            loss_out = 1.0*loss + float(args.aux_loss)*aux_loss
            loss = loss_out

        else:
            # TODO: (1) change to focal loss
            # TODO: (2) change to ASL
            loss =  F.binary_cross_entropy_with_logits(pred.view(labels.size(0),-1),labels.cuda(),reduction='none')
            # loss = sigmoid_focal_loss(pred.view(labels.size(0),-1), labels.cuda(), alpha=0.005, gamma=5, reduction=None)
            # cri = AsymmetricLoss()
            if args.loss_labels == 'unk': 
                # only use unknown labels for loss
                loss_out = (unk_mask.cuda()*loss).sum()
            else: 
                # use all labels for loss
                loss_out = loss.sum() 

        if train:
            # (FedProx): add proximal term
            if args.alg == 'fedprox':
                global_weight_collector = list(global_model.parameters())
                mu = 0.001
                #for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(model.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                loss_out += fed_prox_reg
            loss_out.backward()
            # Grad Accumulation
            if ((batch_idx + 1) % args.grad_ac_steps == 0):
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()
        ## Updates ##
        loss_total += loss_out.item()
        unk_loss_total += loss_out.item()
        start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
        
        if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
            pred = pred.view(labels.size(0),-1)
        
        all_predictions[start_idx:end_idx] = pred.data.cpu()
        all_targets[start_idx:end_idx] = labels.data.cpu()

        all_masks[start_idx:end_idx] = mask_in.data.cpu()

        batch_idx += 1
        if args.dataset == 'flair':
            data_loader.set_description(f'Testing')
            data_loader.set_postfix(loss=f'{loss_total / (batch_idx + 1):.4f}')
        elif args.dataset == 'flair_fed' or args.dataset == 'coco' or args.dataset == 'voc':
            if not train:
                data_loader.set_description(f'Testing')
                data_loader.set_postfix(loss=f'{loss_total / (batch_idx + 1):.4f}')
            

    loss_total = loss_total/float(all_predictions.size(0))
    unk_loss_total = unk_loss_total/float(all_predictions.size(0))

    return all_predictions,all_targets,all_masks,all_image_ids,loss_total,unk_loss_total


