# -----------------------------------------------------------------------------------------------------
# CoFormer Official Code
# Copyright (c) Junhyeong Cho. All Rights Reserved 
# Licensed under the Apache License 2.0 [see LICENSE for details]
# -----------------------------------------------------------------------------------------------------
# Modified from SWiG (https://github.com/allenai/swig)
# Copyright (c) Allen Institute for Artificial Intelligence. All Rights Reserved 
# Licensed under the MIT License [see https://github.com/allenai/swig/blob/master/LICENSE for details]
# -----------------------------------------------------------------------------------------------------


from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import skimage.io
import skimage.transform
import skimage.color
import skimage
import json
import cv2
import util
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler


torch.multiprocessing.set_sharing_strategy('file_system')


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, img_folder, ann_file, class_list, verb_path, role_path, verb_info, is_training, transform=None):
        """
        Parameters:
            - ann_file : CSV file with annotations
            - class_list : CSV file with class list
        """
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.class_list = class_list
        self.verb_path = verb_path
        self.role_path = role_path
        self.verb_info = verb_info
        self.transform = transform
        self.is_training = is_training
        self.color_change = transforms.Compose([transforms.ColorJitter(hue=0.1, saturation=0.1, brightness=0.1), transforms.RandomGrayscale(p=0.3)])

        with open(self.class_list, 'r') as file:
            self.classes, self.idx_to_class = self.load_classes(csv.reader(file, delimiter=','))

        with open(self.ann_file) as file:
            self.SWiG_json = json.load(file)

        self.image_data = self._read_annotations(self.SWiG_json, verb_info, self.classes)
        self.image_names = list(self.image_data.keys())

        with open(self.verb_path, 'r') as f:
            self.verb_to_idx, self.idx_to_verb = self.load_verb(f)
        with open(self.role_path, 'r') as f:
            self.role_to_idx, self.idx_to_role = self.load_role(f)

        self.image_to_image_idx = {}
        i = 0
        for image_name in self.image_names:
            self.image_to_image_idx[image_name] = i
            i += 1

        # verb_role
        self.verb_role = {verb: value['order'] for verb, value in verb_info.items()}

        # for each verb, the indices of roles in the frame.
        self.vidx_ridx = [[self.role_to_idx[role] for role in self.verb_role[verb]] for verb in self.idx_to_verb]


    def load_classes(self, csv_reader):
        result = {}
        idx_to_result = []
        for line, row in enumerate(csv_reader):
            line += 1
            class_name, class_id = row
            class_id = int(class_id)
            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            idx_to_result.append(class_name.split('_')[0])

        return result, idx_to_result


    def load_verb(self, file):
        verb_to_idx = {}
        idx_to_verb = []


        #unseen_index_list126 = [8, 9, 10, 11, 14, 15, 21, 22, 28, 34, 40, 43, 47, 50, 52, 53, 58, 60, 63, 71, 72, 75, 84, 85, 89, 96, 107, 108, 110, 116, 117, 118, 126, 128, 131, 137, 146, 149, 153, 155, 156, 163, 165, 167, 180, 181, 184, 187, 188, 193, 195, 198, 204, 206, 211, 217, 226, 229, 238, 246, 251, 255, 258, 259, 261, 262, 272, 273, 276, 277, 278, 284, 285, 287, 288, 289, 290, 295, 297, 302, 305, 318, 322, 338, 345, 349, 352, 358, 360, 362, 366, 371, 373, 376, 378, 379, 383, 389, 392, 393, 400, 409, 417, 419, 426, 429, 432, 441, 444, 449, 450, 454, 455, 456, 463, 466, 471, 474, 476, 482, 483, 486, 491, 495, 499, 503]
        unseen_index_list51 = []# [9, 11, 45, 51, 55, 58, 67, 77, 84, 96, 100, 104, 124, 133, 142, 143, 149, 155, 156, 198, 205, 209, 214, 227, 230, 238, 246, 249, 261, 270, 292, 294, 296, 305, 323, 338, 361, 370, 386, 389, 401, 414, 418, 430, 451, 458, 461, 467, 478, 483, 491]
        tokenized_labels = []
        k = 0
        for line in file:
            verb = line.split('\n')[0]
            idx_to_verb.append(verb)
            verb_to_idx[verb] = k
            # if verb in unseen_verb51:
            #     unseen_index_list51.append(k)
            # print(unseen_index_list51)
            k += 1
        return verb_to_idx, idx_to_verb


    def load_role(self, file):
        role_to_idx = {}
        idx_to_role = []
        k = 0
        for line in file:
            role = line.split('\n')[0]
            idx_to_role.append(role)
            role_to_idx[role] = k
            k += 1
        return role_to_idx, idx_to_role


    def __len__(self):
        return len(self.image_names)


    def __getitem__(self, idx):
        img, q_im, q_former, captions = self.load_image(idx) #, q_former
        annot = self.load_annotations(idx)
        # verb = self.image_names[idx].split('/')[2]
        verb = self.image_names[idx].split('/')[-1]
        verb = verb.split('_')[0]
        verb_idx = self.verb_to_idx[verb]
        verb_role = self.verb_info[verb]['order']
        verb_role_idx = [self.role_to_idx[role] for role in verb_role]
        sample = {'img': img, 'annot': annot, 'img_name': self.image_names[idx], 'verb_idx': verb_idx, 'verb_role_idx': verb_role_idx}
        if self.transform:
            sample = self.transform(sample)
        sample['q_feat'] = q_im
        sample['q_form'] = q_former
        sample['captions'] = captions
        return sample


    def load_image(self, image_index):
        im = Image.open(self.image_names[image_index])



        # torch.save(image_outputs.vision_outputs.last_hidden_state, os.path.join(dir_instruct_vsion, self.image_names[image_index].split('/')[-1].split('.')[0]+'.pt'))
        # torch.save(image_outputs.qformer_outputs.last_hidden_state, os.path.join(dir_instruct_vsion_qform, self.image_names[image_index].split('/')[-1].split('.')[0] + '.pt'))
        clip_336_image_outputs = torch.load('/root/autodl-tmp/data/image_512_clip_L_336/' + self.image_names[image_index].split('/')[-1].split('.')[0] + '.pt', weights_only=True)  # 1, 257, 1408
        teacher = torch.load('/root/autodl-tmp/data/Blip2_features/' + self.image_names[image_index].split('/')[-1].split('.')[0] + '.pt', weights_only=True)  # 1, 41, 768

        clip_text_outputs = torch.load(
            '/root/autodl-tmp/text_clip_lage_16/' + self.image_names[image_index].split('/')[-1].split('.')[
                0] + '.pt', weights_only=True)
        # image_outputs = np.load('/mnt/data/SWIG/images_512_np/' + self.image_names[image_index].split('/')[-1].split('.')[0] + '.npy') #1, 257, 1408
        # qformer_outputs = np.load('/mnt/data/SWIG/images_512_qform_np/' + self.image_names[image_index].split('/')[-1].split('.')[0] + '.npy') #1, 32, 768
        im = im.convert('RGB')

        if self.is_training:
            im = np.array(self.color_change(im))
        else:
            im = np.array(im)

        return im.astype(np.float32) / 255.0, clip_336_image_outputs, teacher, clip_text_outputs#.detach()

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.ones((0, 7)) * -1

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            annotation = np.ones((1, 7)) * -1  # allow for 3 annotations
            annotation[0, 0] = a['x1']
            annotation[0, 1] = a['y1']
            annotation[0, 2] = a['x2']
            annotation[0, 3] = a['y2']
            annotation[0, 4] = self.name_to_label(a['class1'])
            annotation[0, 5] = self.name_to_label(a['class2'])
            annotation[0, 6] = self.name_to_label(a['class3'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations


    def _read_annotations(self, json, verb_orders, classes):
        result = {}

        for image in json:
            total_anns = 0
            verb = json[image]['verb']

            order = verb_orders[verb]['order']
            img_file = f"{self.img_folder}/" + image
            result[img_file] = []
            # result[img_file].append({'verb_pos_cap': verb_pos_cap, 'verb_neg_cap': verb_neg_cap, 'class_pos_cap': class_pos_cap, 'class_neg_cap': class_neg_cap})
            for role in order:
                total_anns += 1
                [x1, y1, x2, y2] = json[image]['bb'][role]
                class1 = json[image]['frames'][0][role]
                class2 = json[image]['frames'][1][role]
                class3 = json[image]['frames'][2][role]
                if class1 == '':
                    class1 = 'blank'
                if class2 == '':
                    class2 = 'blank'
                if class3 == '':
                    class3 = 'blank'
                if class1 not in classes:
                    class1 = 'oov'
                if class2 not in classes:
                    class2 = 'oov'
                if class3 not in classes:
                    class3 = 'oov'
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})

            while total_anns < 6:
                total_anns += 1
                [x1, y1, x2, y2] = [-1, -1, -1, -1]
                class1 = 'Pad'
                class2 = 'Pad'
                class3 = 'Pad'
                result[img_file].append(
                    {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class1': class1, 'class2': class2, 'class3': class3})

        return result


    def name_to_label(self, name):
        return self.classes[name]

    def num_nouns(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    shift_0 = [s['shift_0'] for s in data]
    shift_1 = [s['shift_1'] for s in data]
    scales = [s['scale'] for s in data]
    img_names = [s['img_name'] for s in data]
    verb_indices = [s['verb_idx'] for s in data]
    verb_indices = torch.tensor(verb_indices)
    verb_role_indices = [s['verb_role_idx'] for s in data]
    verb_role_indices = [torch.tensor(vri) for vri in verb_role_indices]
    q_feats = [s['q_feat'] for s in data]
    q_forms = [s['q_form'] for s in data]
    captions = [s['captions'] for s in data]
    
    heights = [int(s.shape[0]) for s in imgs]
    widths = [int(s.shape[1]) for s in imgs]

    batch_size = len(imgs)
    max_height = 336
    max_width = 336

    padded_imgs = torch.zeros(batch_size, max_height, max_width, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, shift_0[i]:shift_0[i]+img.shape[0], shift_1[i]:shift_1[i]+img.shape[1], :] = img
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 7)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 7)) * -1

    widths = torch.tensor(widths).float()
    heights = torch.tensor(heights).float()
    shift_0 = torch.tensor(shift_0).float()
    shift_1 = torch.tensor(shift_1).float()
    scales = torch.tensor(scales).float()
    mw = torch.tensor(max_width).float()
    mh = torch.tensor(max_height).float()
    # q_feats = torch.tensor(q_feats)
    # q_forms = torch.tensor(q_forms)

    return (util.misc.nested_tensor_from_tensor_list(padded_imgs),
            [{'verbs': vi,
              'roles': vri,
              'boxes': util.box_ops.swig_box_xyxy_to_cxcywh(annot[:, :4], mw, mh, gt=True),
              'boxes_raw': annot[:, :4],
              'labels': annot[:, -3:],
              'width': w,
              'height': h,
              'shift_0': s0,
              'shift_1': s1,
              'scale': sc,
              'max_width': mw,
              'max_height': mh,
              'img_name': im,
              'q_feat': q_feats,
              'q_form': q_forms,
              'captions': captions
              }
              for vi, vri, annot, w, h, s0, s1, sc, im, q_feats, q_forms, captions in zip(verb_indices, verb_role_indices, annot_padded, widths, heights, shift_0, shift_1, scales, img_names, q_feats, q_forms, captions)]) #, q_forms


class Resizer(object):
    def __init__(self, is_for_training):
        self.is_for_training = is_for_training


    def __call__(self, sample, min_side=246, max_side=336): #246, 336, #164, 224
        image, annots, image_name = sample['img'], sample['annot'], sample['img_name']

        rows_orig, cols_orig, cns_orig = image.shape
        smallest_side = min(rows_orig, cols_orig)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows_orig, cols_orig)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        if self.is_for_training:
            scale_factor = random.choice([1, 0.75, 0.5])
            scale = scale*scale_factor

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows_orig * scale)), int(round((cols_orig * scale)))))
        rows, cols, cns = image.shape

        new_image = np.zeros((rows, cols, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        shift_1 = int((336 - cols) * 0.5)
        shift_0 = int((336 - rows) * 0.5)

        annots[:, :4][annots[:, :4] != -1] *= scale

        annots[:, 0][annots[:, 0] != -1] = annots[:, 0][annots[:, 0] != -1] + shift_1
        annots[:, 1][annots[:, 1] != -1] = annots[:, 1][annots[:, 1] != -1] + shift_0
        annots[:, 2][annots[:, 2] != -1] = annots[:, 2][annots[:, 2] != -1] + shift_1
        annots[:, 3][annots[:, 3] != -1] = annots[:, 3][annots[:, 3] != -1] + shift_0

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'img_name': image_name, 'verb_idx': sample['verb_idx'],
        'verb_role_idx': sample['verb_role_idx'], 'shift_1': shift_1, 'shift_0': shift_0}


class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        image, annots, img_name = sample['img'], sample['annot'], sample['img_name']
        if np.random.rand() < flip_x:
            image = image[:, ::-1, :]
            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            annots[:, 0][annots[:, 0] != -1] = cols - x2[annots[:, 0] != -1]
            annots[:, 2][annots[:, 2] != -1] = cols - x1[annots[:, 2] != -1]

        sample = {'img': image, 'annot': annots, 'img_name': img_name, 'verb_idx': sample['verb_idx'], 'verb_role_idx': sample['verb_role_idx']}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'img_name': sample['img_name'], 'verb_idx': sample['verb_idx'], 'verb_role_idx': sample['verb_role_idx']}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def build(image_set, args):
    root = Path(args.swig_path)
    img_folder = root / "images_512"

    PATHS = {
        "train": root / "SWiG_jsons" / "train_L10_with_obj_cap_refined.json", #train_L10_with_obj_cap_refined.json
        #"val_unseen": root / "SWiG_jsons" / "dev_L10_unseen_with_obj_cap_refined.json", #test_L10_unseen_with_obj_cap_refined
        "val": root / "SWiG_jsons" / "dev_L10_unseen_with_obj_cap_refined.json", #test_L10_unseen_with_obj_cap_refined
        "test": root / "SWiG_jsons" / "test_L10_unseen_with_obj_cap_refined.json", #test_ind.json, test_L20_rare_with_obj_cap_refined.json
        #"test_unseen": root / "SWiG_jsons" / "test_L10_unseen_with_obj_cap_refined.json",  #dev_L10_unseen_with_obj_cap_refined, dev_L20_rare_with_obj_cap_refined
    }
    ann_file = PATHS[image_set]

    classes_file = Path(args.swig_path) / "SWiG_jsons" / "train_classes.csv"
    verb_path = Path(args.swig_path) / "SWiG_jsons" / "verb_indices.txt"
    role_path = Path(args.swig_path) / "SWiG_jsons" / "role_indices.txt"

    with open(f'{args.swig_path}/SWiG_jsons/imsitu_space.json') as f:
        all = json.load(f)
        verb_orders = all['verbs']

    is_training = (image_set == 'train')

    TRANSFORMS = {
        "train": transforms.Compose([Normalizer(), Augmenter(), Resizer(False)]), #
        "val": transforms.Compose([Normalizer(), Resizer(False)]),
        "test": transforms.Compose([Normalizer(), Resizer(False)]),
        "val_unseen": transforms.Compose([Normalizer(), Resizer(False)]),
        "test_unseen": transforms.Compose([Normalizer(), Resizer(False)]),
    }
    tfs = TRANSFORMS[image_set]
    
    dataset = CSVDataset(img_folder=str(img_folder),
                         ann_file=ann_file,
                         class_list=classes_file,
                         verb_path=verb_path,
                         role_path=role_path,
                         verb_info=verb_orders,
                         is_training=is_training,
                         transform=tfs)

    if is_training:
        args.SWiG_json_train = dataset.SWiG_json
    else:
        if not args.test:
            args.SWiG_json_dev = dataset.SWiG_json
        else:
            args.SWiG_json_test = dataset.SWiG_json

    args.idx_to_verb = dataset.idx_to_verb
    args.idx_to_role = dataset.idx_to_role
    args.idx_to_class = dataset.idx_to_class
    args.vidx_ridx = dataset.vidx_ridx

    return dataset
