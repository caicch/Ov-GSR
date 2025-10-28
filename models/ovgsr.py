# ----------------------------------------------------------------------------------------------
# Ov-SGR Official Code
# Modified from OvGSR and OpenSU Official Code
# Licensed under the Apache License 2.0 [see LICENSE for details]
# ----------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved [see LICENSE for details]
# ----------------------------------------------------------------------------------------------

"""
OvGSR model and criterion classes.
"""
# from clip.model import Transformer, LayerNorm, MLP, QuickGELU
# from clip.clip import _download
# from .position_encoding import PositionEmbeddingSine
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, accuracy_swig, accuracy_swig_bbox)
from .backbone import build_backbone
from .transformer import build_transformer
from transformers import Blip2Processor, Blip2Model, AutoTokenizer, CLIPTextModel
#from transformers import AutoProcessor, InstructBlipVisionModel, InstructBlipQFormerModel, InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
# from transformers import AutoModel, AutoConfig
# from transformers import CLIPImageProcessor, pipeline, CLIPTokenizer
from .prompters import PadPrompter, BoundingBoxPrompter, TensorPrompter, EdgePrompter
import os
import numpy as np


class OvGSR(nn.Module):
    """OvGSR model for Grounded Situation Recognition"""
    def __init__(self,  transformer, num_noun_classes, vidx_ridx): #backbone,
        """ Initialize the model.
        Parameters:
            - backbone: torch module of the backbone to be used. See backbone.py
            - transformer: torch module of the transformer architecture. See transformer.py
            - num_noun_classes: the number of noun classes
            - vidx_ridx: verb index to role index
        """
        super().__init__()
        #self.backbone = backbone
        self.transformer = transformer
        self.num_noun_classes = num_noun_classes
        self.vidx_ridx = vidx_ridx
        self.num_role_tokens = 190
        self.num_verb_tokens = 504

        hidden_dim = transformer.d_model


        self.role_token_embed = nn.Embedding(self.num_role_tokens, hidden_dim)
        self.verb_token_embed = nn.Embedding(self.num_verb_tokens, hidden_dim)
        self.IL_token_embed = nn.Embedding(1, hidden_dim)
        self.RL_token_embed = nn.Embedding(1, hidden_dim)
        self.context_length = 77
        self.prefix_length = 8
        self.hoi_prefix = nn.Parameter(torch.empty(8, hidden_dim))
        self.conjun_length = 2
        self.vocab_size = 49408
        self.token_embedding = nn.Embedding(49408, hidden_dim)
        self.dtype = torch.float32
        self.text_projection = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, hidden_dim))
        

        self.all_indices = set(range(504))
        self.unseen_indices = set(range(454, 504))

        self.proj = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.proj_cls = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.proj_verb_large = nn.Linear(768, hidden_dim)


        # classifiers & predictors (for grounded noun prediction)
        self.noun_1_classifier = nn.Linear(hidden_dim, self.num_noun_classes)
        self.noun_2_classifier = nn.Linear(hidden_dim, self.num_noun_classes)
        self.noun_3_classifier = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.GELU(),
                                             nn.Dropout(0.3),
                                             nn.Linear(hidden_dim*2, self.num_noun_classes))
        self.bbox_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.GELU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, hidden_dim*2),
                                             nn.GELU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, 4))
        self.bbox_conf_predictor = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2),
                                             nn.GELU(),
                                             nn.Dropout(0.2),
                                             nn.Linear(hidden_dim*2, 1))

        # layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)


        self.proj_prompt = nn.Linear(1408, hidden_dim) #change here blip 1408, G_clip 1664
        self.proj_verb_pos = nn.Linear(768, hidden_dim)
        self.proj_verb_neg = nn.Linear(768, hidden_dim)

        self.proj_class_pos = nn.Linear(768, hidden_dim)
        self.proj_class_neg = nn.Linear(768, hidden_dim)


        # self.prompter = PadPrompter(224, 30)
        self.boxprompter = BoundingBoxPrompter(224, 6)
        self.EdgePrompter = EdgePrompter(16, 1408) #change here blip 1408, G_clip 1664

        # self.ln_final = LayerNorm(hidden_dim)

        label_emd_path = os.path.join('/root/autodl-tmp/data/SWiG_jsons/', 'label_Single_Action.pt')
        self.label_emb = torch.load(label_emd_path).to(torch.float32)

        class_emd_path = os.path.join('/root/autodl-tmp/data/SWiG_jsons/', 'label_Single_class.pt')
        self.class_emb = torch.load(class_emd_path).to(torch.float32)

        self.linear_blip = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.Dropout(p=0.5)  # Specify your dropout rate here #blip 1408 #1664
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(77, 77)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask


    def forward(self, samples, targets=None, inference=False): #inference
        """
        Parameters:
               - samples: The forward expects a NestedTensor, which consists of:
                        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - targets: This has verbs, roles and labels information
               - inference: boolean, used in inference
        Outputs:
               - out: dict of tensors. 'pred_verb', 'pred_noun', 'pred_bbox' and 'pred_bbox_conf' are keys
        """
        MAX_NUM_ROLES = 6
        device = targets[0]['q_feat'].device

        #
        batch_size = len(targets)#targets.shape[0]
        batch_verb, batch_noun_cls, batch_noun_2, batch_noun_3, batch_bbox, batch_bbox_conf = [], [], [], [], [], []

        pos_verb = []
        neg_verb = []
        pos_class = []
        neg_class = []

        for target in targets:
            captions = target.get('captions', [])  # Get captions from each target
            if len(captions) >= 4:  # Ensure there are at least 4 elements
                pos_verb.append(captions[0])
                neg_verb.append(captions[1])
                pos_class.append(captions[2])
                neg_class.append(captions[3])

        pos_verb_text = torch.stack(pos_verb, dim=0)
        neg_verb_text = torch.stack(neg_verb, dim=0)
        pos_class_text = torch.stack(pos_class, dim=0)
        neg_class_text = torch.stack(neg_class, dim=0)


        blip_feats = [target['q_form'] for target in targets]
        b_src = torch.cat(blip_feats, dim=0)

        b_src_0 = b_src[:, 0, :].unsqueeze(1)  # Shape: [b, 1, 1668]

        # Step 2: Remove the 0th index and reshape the remaining tensor to [b, 16, 16, 1668]
        b_src_rest = b_src[:, 1:, :]  # Shape: [b, 256, 1668]
        b_src_reshaped = b_src_rest.reshape(batch_size, 16, 16, -1)

        eg_src = self.EdgePrompter(b_src_reshaped)

        eg_src_reshaped = torch.cat([b_src_0, eg_src.reshape(batch_size, 16*16, -1)], dim=1)

        img_overview = self.proj_prompt(eg_src_reshaped)
        text_overview_pos = self.proj_verb_pos(pos_verb_text)
        text_overview_neg = self.proj_verb_neg(neg_verb_text).mean(dim=1)
        verb_dis, _ = self.multihead_attn(img_overview, text_overview_pos, text_overview_pos)
        verb_dis_vhs = verb_dis[:, 0, :] / verb_dis[:, 0, :].norm(dim=-1, keepdim=True)

        # Apply bounding box prompts per-sample across the batch
        try:
            boxes_raw_batched = torch.stack([t['boxes_raw'] for t in targets], dim=0).to(G_src_reshaped.device)
        except Exception:
            # Fallback: if targets already provide batched boxes or any issue arises, pass as-is
            boxes_raw_batched = targets[0]['boxes_raw']
        bx_src = self.boxprompter(b_src_reshaped, boxes_raw_batched)
        bx_src_reshaped = torch.cat([b_src_0, bx_src.reshape(batch_size, 16*16, -1)], dim=1)
        bx_class = self.proj_prompt(bx_src_reshaped)
        cap_class_pos = self.proj_class_pos(pos_class_text)
        cap_class_neg = self.proj_class_neg(neg_class_text).mean(dim=1)
        cap_class, _ = self.multihead_attn(bx_class, cap_class_pos, cap_class_pos)
        cap_class_vhs = cap_class[:, 0, :] / cap_class[:, 0, :].norm(dim=-1, keepdim=True)


        clip_feats = [target['q_feat'] for target in targets]
        src = torch.cat(clip_feats, dim=0)

        flattend_src = self.linear_blip(src)

        vhs = flattend_src[:, 0, :] / flattend_src[:, 0, :].norm(dim=-1, keepdim=True)
        verb_scale = vhs @ self.proj

        text_features = self.label_emb.to(device) / self.label_emb.to(device).norm(dim=-1, keepdim=True)

        verb_pred = verb_scale @ text_features.T #verb_scale 32_512, verb_pred 32_504

        # model prediction
        for i in range(batch_size):

            if not inference: #inputs['pixel_values'] = torch.zeros(1, 3, 224, 224)
                outs = self.transformer(verb_pred[i:i+1, :], flattend_src[i:i+1, :], targets[i]['q_form'], self.IL_token_embed.weight, self.RL_token_embed.weight,
                                        self.verb_token_embed.weight, self.role_token_embed.weight, self.vidx_ridx, self.label_emb.to(device), targets=targets[i], inference=inference) #self.verb2hoi_proj.to(device)
            else:
                outs = self.transformer(verb_pred[i], flattend_src[i], targets[i]['q_form'], self.IL_token_embed.weight,
                                        self.RL_token_embed.weight,
                                        self.verb_token_embed.weight, self.role_token_embed.weight, self.vidx_ridx, self.label_emb.to(device), inference=inference) #self.verb2hoi_proj_eval.to(device),

            # output features & predictions
            final_cls, selected_roles = outs[0], outs[1]#, outs[2] #, outs[3], outs[4]

            num_selected_roles = len(selected_roles)

            noun_3_pred = self.noun_3_classifier(final_cls)
            
            selected_class = F.pad(final_cls.clone(), (0,0,0,MAX_NUM_ROLES-num_selected_roles), mode='constant', value=0)[-1]

            noun_3_pred = F.pad(noun_3_pred, (0,0,0,MAX_NUM_ROLES-num_selected_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, self.num_noun_classes)
            bbox_pred = self.bbox_predictor(final_cls).sigmoid()
            bbox_pred = F.pad(bbox_pred, (0,0,0,MAX_NUM_ROLES-num_selected_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, 4)
            bbox_conf_pred = self.bbox_conf_predictor(final_cls)
            bbox_conf_pred = F.pad(bbox_conf_pred, (0,0,0,MAX_NUM_ROLES-num_selected_roles), mode='constant', value=0)[-1].view(1, MAX_NUM_ROLES, 1)


            batch_noun_3.append(noun_3_pred)
            batch_bbox.append(bbox_pred)
            batch_bbox_conf.append(bbox_conf_pred)
            batch_noun_cls.append(selected_class)

        #9928?
        noun_cls = torch.cat(batch_noun_cls, dim=0)

        final_rhs = noun_cls @ self.proj_cls
        class_features = self.class_emb.to(device) / self.class_emb.to(device).norm(dim=-1, keepdim=True)

        noun_ovClass = final_rhs @ class_features.T

        # outputs
        out = {}
        out['pred_verb'] = verb_pred #torch.cat(batch_verb, dim=0)
        out['verb_scale'] = verb_scale
        out['pred_verb_prompt'] = verb_dis_vhs
        out['pred_verb_prompt_neg'] = text_overview_neg
        out['pred_class_prompt'] = cap_class_vhs
        out['pred_class_prompt_neg'] = cap_class_neg

        out['pred_noun'] = torch.cat(batch_noun_3, dim=0)
        out['pred_noun_ovClass'] = noun_ovClass

        out['pred_bbox'] = torch.cat(batch_bbox, dim=0)
        out['pred_bbox_conf'] = torch.cat(batch_bbox_conf, dim=0)

        return out

class L1DistillationLoss(nn.Module):
    def __init__(self):
        """
        L1 loss for feature distillation.
        Computes the mean absolute error between teacher and student features.
        """
        super(L1DistillationLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features (torch.Tensor): Features from the student model. Shape: [batch_size, ...].
            teacher_features (torch.Tensor): Features from the teacher model. Shape: [batch_size, ...].

        Returns:
            torch.Tensor: Computed L1 loss.
        """
        return self.l1_loss(student_features, teacher_features)

class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction="batchmean"):
        """
        KL Divergence Loss class.

        Args:
        - reduction (str): Specifies the reduction to apply to the output:
            - "none": No reduction.
            - "batchmean": Divides the sum of the output by the batch size.
            - "mean": Averages the loss across all elements.
            - "sum": Sums the loss.
        """
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, teacher_logits, student_logits):
        """
        Computes KL Divergence Loss.

        Args:
        - teacher_logits (torch.Tensor): Logits from the teacher model. Shape: [batch, dim].
        - student_logits (torch.Tensor): Logits from the student model. Shape: [batch, dim].

        Returns:
        - torch.Tensor: KL Divergence loss.
        """
        # Convert logits to log-probabilities (teacher) and probabilities (student)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # Compute KL Divergence
        kl_div = F.kl_div(teacher_log_probs, student_probs, reduction=self.reduction)
        return kl_div


class LabelSmoothing(nn.Module):
    """ NLL loss with label smoothing """

    def __init__(self, smoothing=0.0):
        """ Constructor for the LabelSmoothing module.
        Parameters:
                - smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SWiGCriterion(nn.Module):
    """
    Loss for OvGSR with SWiG dataset, and OvGSR evaluation.
    """

    def __init__(self, weight_dict, SWiG_json_train=None, SWiG_json_eval=None, idx_to_role=None):
        """
        Create the criterion.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.loss_function_verb = LabelSmoothing(0.3)
        self.loss_function_noun_1 = LabelSmoothing(0.2)
        self.loss_function_noun_2 = LabelSmoothing(0.2)
        self.loss_function_noun_3 = LabelSmoothing(0.2)
        self.SWiG_json_train = SWiG_json_train
        self.SWiG_json_eval = SWiG_json_eval
        self.idx_to_role = idx_to_role

        self.kl_loss_fn = L1DistillationLoss() #KLDivergenceLoss(reduction="batchmean")

    def forward(self, outputs, targets, eval=False):
        """ This performs the loss computation, and evaluation of OvGSR.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             eval: boolean, used in evlauation
        """
        NUM_ANNOTATORS = 3

        # gt verb (value & value-all) acc and calculate noun losses
        # assert 'pred_noun_ovClass' in outputs
        # assert 'pred_noun_2' in outputs
        assert 'pred_noun' in outputs
        pred_noun_1, pred_noun = outputs['pred_noun_ovClass'], outputs['pred_noun']
        device = pred_noun.device
        batch_size = pred_noun.shape[0]
        batch_noun_1_loss, batch_noun_2_loss, batch_noun_3_loss, batch_noun_acc, batch_noun_correct = [], [], [], [], []
        for i in range(batch_size):
            p1, p2, t = pred_noun_1[i], pred_noun[i], targets[i]
            roles = t['roles']
            num_roles = len(roles)
            role_targ = t['labels'][:num_roles]
            role_targ = role_targ.long()
            # noun_1_loss
            role_pred_1 = p1[:num_roles]
            e_noun_1_loss = []
            for n in range(NUM_ANNOTATORS):
                e_noun_1_loss.append(self.loss_function_noun_1(role_pred_1, role_targ[:, n].clone()))
            batch_noun_1_loss.append(sum(e_noun_1_loss))

            role_pred = p2[:num_roles]
            e_noun_3_loss = []
            for n in range(NUM_ANNOTATORS):
                e_noun_3_loss.append(self.loss_function_noun_3(role_pred, role_targ[:, n].clone()))
            batch_noun_3_loss.append(sum(e_noun_3_loss))
            # evaluation of noun prediction
            acc_res = accuracy_swig(role_pred, role_targ)
            batch_noun_acc += acc_res[1]
            batch_noun_correct += acc_res[0]

        # class prompts for contrastive loss (positive vs negative)
        class_pred_prompt_logits = outputs['pred_class_prompt'].squeeze(1)
        class_pred_prompt_logits_neg = outputs['pred_class_prompt_neg'].squeeze(1)

        # Class contrastive loss: push class prompt away from negative class prompt
        cp_pos = F.normalize(class_pred_prompt_logits, dim=-1)
        cp_neg = F.normalize(class_pred_prompt_logits_neg, dim=-1)
        class_neg_sim = (cp_pos * cp_neg).sum(dim=-1)
        class_contrast_ls = F.relu(class_neg_sim).mean()

        noun_1_loss = (torch.stack(batch_noun_1_loss).mean() * 0.01) #+ (5*class_contrast_ls)
        # noun_2_loss = torch.stack(batch_noun_2_loss).mean()
        noun_3_loss = torch.stack(batch_noun_3_loss).mean() + (class_contrast_ls)
        noun_acc = torch.stack(batch_noun_acc)
        noun_correct = torch.stack(batch_noun_correct)

        # top-1 & top 5 verb acc and calculate verb loss
        assert 'pred_verb' in outputs
        #assert 'pred_verb_prompt' in outputs
        verb_pred_logits = outputs['pred_verb'].squeeze(1)
        verb_pred_prompt_logits = outputs['pred_verb_prompt'].squeeze(1)
        verb_pred_prompt_logits_neg = outputs['pred_verb_prompt_neg'].squeeze(1)
        verb_scale = outputs['verb_scale'].squeeze(1)

        gt_verbs = torch.stack([t['verbs'] for t in targets])
        verb_acc_topk = accuracy(verb_pred_logits, gt_verbs, topk=(1, 5))
        verb_ls = self.loss_function_verb(verb_pred_logits, gt_verbs)

        verb_dis_ls = self.kl_loss_fn(verb_scale, verb_pred_prompt_logits) #teacher - verb_pred_prompt_logits, student - verb_scale for kl

        # Contrastive loss: push image prompt away from negative text prompt
        vp_pos = F.normalize(verb_pred_prompt_logits, dim=-1)
        vp_neg = F.normalize(verb_pred_prompt_logits_neg, dim=-1)
        neg_sim = (vp_pos * vp_neg).sum(dim=-1)  # cosine similarity after normalization
        verb_contrast_ls = F.relu(neg_sim).mean()  # margin = 0.0, penalize positive similarity

        verb_loss = (verb_ls) + (2*verb_dis_ls) + (5*verb_contrast_ls)

        # top-1 & top 5 (value & value-all) acc
        batch_noun_acc_topk, batch_noun_correct_topk = [], []
        for verbs in verb_pred_logits.topk(5)[1].transpose(0, 1):
            batch_noun_acc = []
            batch_noun_correct = []
            for i in range(batch_size):
                v, p2, t = verbs[i], pred_noun[i], targets[i]
                if v == t['verbs']:
                    roles = t['roles']
                    num_roles = len(roles)
                    role_pred = p2[:num_roles]
                    role_targ = t['labels'][:num_roles]
                    role_targ = role_targ.long()
                    acc_res = accuracy_swig(role_pred, role_targ)
                    batch_noun_acc += acc_res[1]
                    batch_noun_correct += acc_res[0]
                else:
                    batch_noun_acc += [torch.tensor(0., device=device)]
                    batch_noun_correct += [torch.tensor([0, 0, 0, 0, 0, 0], device=device)]
            batch_noun_acc_topk.append(torch.stack(batch_noun_acc))
            batch_noun_correct_topk.append(torch.stack(batch_noun_correct))
        noun_acc_topk = torch.stack(batch_noun_acc_topk)
        noun_correct_topk = torch.stack(batch_noun_correct_topk)  # topk x batch x max roles

        # bbox prediction
        assert 'pred_bbox' in outputs
        assert 'pred_bbox_conf' in outputs
        pred_bbox = outputs['pred_bbox']
        pred_bbox_conf = outputs['pred_bbox_conf'].squeeze(2)
        batch_bbox_acc, batch_bbox_acc_top1, batch_bbox_acc_top5 = [], [], []
        batch_bbox_loss, batch_giou_loss, batch_bbox_conf_loss = [], [], []
        for i in range(batch_size):
            pb, pbc, t = pred_bbox[i], pred_bbox_conf[i], targets[i]
            mw, mh, target_bboxes = t['max_width'], t['max_height'], t['boxes']
            cloned_pb, cloned_target_bboxes = pb.clone(), target_bboxes.clone()
            num_roles = len(t['roles'])
            bbox_exist = target_bboxes[:, 0] != -1
            num_bbox = bbox_exist.sum().item()

            # bbox conf loss
            loss_bbox_conf = F.binary_cross_entropy_with_logits(pbc[:num_roles],
                                                                bbox_exist[:num_roles].float(), reduction='mean')
            batch_bbox_conf_loss.append(loss_bbox_conf)

            # bbox reg loss and giou loss
            if num_bbox > 0:
                loss_bbox = F.l1_loss(pb[bbox_exist], target_bboxes[bbox_exist], reduction='none')
                loss_giou = 1 - torch.diag(
                    box_ops.generalized_box_iou(box_ops.swig_box_cxcywh_to_xyxy(pb[bbox_exist], mw, mh, device=device),
                                                box_ops.swig_box_cxcywh_to_xyxy(target_bboxes[bbox_exist], mw, mh,
                                                                                device=device, gt=True)))
                batch_bbox_loss.append(loss_bbox.sum() / num_bbox)
                batch_giou_loss.append(loss_giou.sum() / num_bbox)

            # top1 correct noun & top5 correct nouns
            noun_correct_top1 = noun_correct_topk[0]
            noun_correct_top5 = noun_correct_topk.sum(dim=0)

            # convert coordinates
            pb_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_pb, mw, mh, device=device)
            gt_bbox_xyxy = box_ops.swig_box_cxcywh_to_xyxy(cloned_target_bboxes, mw, mh, device=device, gt=True)

            # accuracies
            if not eval:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                     noun_correct[i], bbox_exist, t, self.SWiG_json_train,
                                                     self.idx_to_role)
                batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                          noun_correct_top1[i], bbox_exist, t, self.SWiG_json_train,
                                                          self.idx_to_role)
                batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                          noun_correct_top5[i], bbox_exist, t, self.SWiG_json_train,
                                                          self.idx_to_role)
            else:
                batch_bbox_acc += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                     noun_correct[i], bbox_exist, t, self.SWiG_json_eval,
                                                     self.idx_to_role, eval)
                batch_bbox_acc_top1 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                          noun_correct_top1[i], bbox_exist, t, self.SWiG_json_eval,
                                                          self.idx_to_role, eval)
                batch_bbox_acc_top5 += accuracy_swig_bbox(pb_xyxy.clone(), pbc, gt_bbox_xyxy.clone(), num_roles,
                                                          noun_correct_top5[i], bbox_exist, t, self.SWiG_json_eval,
                                                          self.idx_to_role, eval)

        if len(batch_bbox_loss) > 0:
            bbox_loss = torch.stack(batch_bbox_loss).mean()
            giou_loss = torch.stack(batch_giou_loss).mean()
        else:
            bbox_loss = torch.tensor(0., device=device)
            giou_loss = torch.tensor(0., device=device)

        bbox_conf_loss = torch.stack(batch_bbox_conf_loss).mean()
        bbox_acc = torch.stack(batch_bbox_acc)
        bbox_acc_top1 = torch.stack(batch_bbox_acc_top1)
        bbox_acc_top5 = torch.stack(batch_bbox_acc_top5)

        out = {}
        # losses
        out['loss_vce'] = verb_loss
        out['loss_nce_1'] = noun_1_loss
        # out['loss_nce_2'] = noun_2_loss
        out['loss_nce_3'] = noun_3_loss
        out['loss_bbox'] = bbox_loss
        out['loss_giou'] = giou_loss
        out['loss_bbox_conf'] = bbox_conf_loss

        # All metrics should be calculated per verb and averaged across verbs.
        ## In the dev and test split of SWiG dataset, there are 50 images for each verb (same number of images per verb).
        ### Our implementation is correct to calculate metrics for the dev and test split of SWiG dataset.
        ### We calculate metrics in this way for simple implementation in distributed data parallel setting.

        # accuracies (for verb and noun)
        out['verb_acc_top1'] = verb_acc_topk[0]
        out['verb_acc_top5'] = verb_acc_topk[1]
        out['noun_acc_top1'] = noun_acc_topk[0].mean()
        out['noun_acc_all_top1'] = (noun_acc_topk[0] == 100).float().mean() * 100
        out['noun_acc_top5'] = noun_acc_topk.sum(dim=0).mean()
        out['noun_acc_all_top5'] = (noun_acc_topk.sum(dim=0) == 100).float().mean() * 100
        out['noun_acc_gt'] = noun_acc.mean()
        out['noun_acc_all_gt'] = (noun_acc == 100).float().mean() * 100
        out['mean_acc'] = torch.stack([v for k, v in out.items() if 'noun_acc' in k or 'verb_acc' in k]).mean()
        # accuracies (for bbox)
        out['bbox_acc_gt'] = bbox_acc.mean()
        out['bbox_acc_all_gt'] = (bbox_acc == 100).float().mean() * 100
        out['bbox_acc_top1'] = bbox_acc_top1.mean()
        out['bbox_acc_all_top1'] = (bbox_acc_top1 == 100).float().mean() * 100
        out['bbox_acc_top5'] = bbox_acc_top5.mean()
        out['bbox_acc_all_top5'] = (bbox_acc_top5 == 100).float().mean() * 100

        return out


def build(args):
    #backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = OvGSR(#backbone,
                     transformer,
                     num_noun_classes=args.num_noun_classes,
                     vidx_ridx=args.vidx_ridx)

    criterion = None

    if not args.inference:
        weight_dict = {'loss_nce_1': args.noun_1_loss_coef, 'loss_nce_2': args.noun_2_loss_coef,
                       'loss_nce_3': args.noun_3_loss_coef, 'loss_vce': args.verb_loss_coef,
                       'loss_bbox':args.bbox_loss_coef, 'loss_giou':args.giou_loss_coef,
                       'loss_bbox_conf':args.bbox_conf_loss_coef}

        if not args.test:
            criterion = SWiGCriterion(weight_dict=weight_dict,
                                      SWiG_json_train=args.SWiG_json_train,
                                      SWiG_json_eval=args.SWiG_json_dev,
                                      idx_to_role=args.idx_to_role)
        else:
            criterion = SWiGCriterion(weight_dict=weight_dict,
                                      SWiG_json_train=args.SWiG_json_train,
                                      SWiG_json_eval=args.SWiG_json_test,
                                      idx_to_role=args.idx_to_role)

    return model, criterion
