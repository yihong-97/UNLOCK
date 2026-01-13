# --------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# --------------------------------------------------------------------------------
import os.path
import random

import numpy as np
from ..builder import PIPELINES
from blendpassscripts.helpers.labels import id2label, labels
from tools.panoptic_deeplab.utils import rgb2id
from mmdet.core import BitmapMasks
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import glob
from PIL import Image

def isValidBox(box):
    isValid = False
    x1, y1, x2, y2 = box
    if x1 < x2 and y1 < y2:
        isValid = True
    return isValid

def get_bbox_coord(mask):
    # bbox computation for a segment
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(width) - 1
    y2 = y1 + int(height) - 1
    bbox = [x1, y1, x2, y2]
    return bbox

@PIPELINES.register_module()
class GenBlendPASSAmodalPseudoLabelsForMaskFormer(object):

    def __init__(self, sigma, mode, num_classes=19, gen_instance_classids_from_zero=True):
        self.ignore_label = 255
        self.label_divisor = 1000
        self.thing_list = [11, 12, 13, 14, 15, 16, 17]
        self.thing_list_mapids = {11:0, 12:1, 13:2, 14:3, 15:4, 16:5, 17:6}
        self.ignore_stuff_in_offset = True
        self.small_instance_area = 4096 # not using currently
        self.small_instance_weight = 3  # not using currently
        self.ignore_crowd_in_semantic = True
        self.ignore_crowd_in_instance = True
        self.sigma = sigma
        self.mode = mode
        self.num_classes = num_classes
        self.gen_instance_classids_from_zero = gen_instance_classids_from_zero
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def _map_instance_class_ids(self, catid):
        if self.gen_instance_classids_from_zero:
            return self.thing_list_mapids[catid]
        else:
            return catid

    def __call__(self, results):
        semantic = results['gt_semantic']
        instances = results['gt_instance']
        ainstances = results['gt_ainstance']

        height, width = semantic.shape[0], semantic.shape[1]

        panoptic_only_thing_classes = np.zeros(semantic.shape)
        max_inst_per_class = np.zeros(len(self.thing_list))
        class_id_tracker = {}
        for cid in self.thing_list:
            class_id_tracker[cid] = 1
        gt_masks = []
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = np.empty([0, 4], dtype=np.float32)
        for cat_id, mask in instances.items():
            if not (mask==1).sum() == 0:
                cat_class = cat_id // self.label_divisor
                if cat_class in self.thing_list:
                    box = get_bbox_coord(mask)
                    if isValidBox(box):
                        gt_bboxes.append(box)
                        gt_masks.append(mask.astype(np.uint8))
                        gt_labels.append(self._map_instance_class_ids(cat_class))
                        panoptic_only_thing_classes[mask == 1] = cat_class * self.label_divisor + class_id_tracker[cat_class]
                        class_id_tracker[cat_class] += 1
        gt_amasks = []
        gt_alabels = []
        gt_abboxes = []
        gt_abboxes_ignore = np.empty([0, 4], dtype=np.float32)
        for acat_id, amask in ainstances.items():
            if not (amask==1).sum() == 0:
                acat_class = acat_id // self.label_divisor
                if acat_class in self.thing_list:
                    abox = get_bbox_coord(amask)
                    if isValidBox(abox):
                        gt_abboxes.append(abox)
                        gt_amasks.append(amask.astype(np.uint8))
                        gt_alabels.append(self._map_instance_class_ids(acat_class))

        for cid in list(class_id_tracker.keys()):
            max_inst_per_class[self._map_instance_class_ids(cid)] = class_id_tracker[cid]
        gt_masks = BitmapMasks(gt_masks, height, width)
        gt_amasks = BitmapMasks(gt_amasks, height, width)
        results['gt_masks'] = gt_masks
        results['gt_amasks'] = gt_amasks
        results['gt_semantic_seg'] = semantic.astype('long')
        results['gt_panoptic_only_thing_classes'] = panoptic_only_thing_classes.astype('long')
        results['gt_labels'] = np.asarray(gt_labels).astype('long')
        results['gt_alabels'] = np.asarray(gt_alabels).astype('long')
        results['max_inst_per_class'] = max_inst_per_class.astype('long')
        results['gt_bboxes'] = np.asarray(gt_bboxes).astype(np.float32)
        results['gt_abboxes'] = np.asarray(gt_abboxes).astype(np.float32)
        results['gt_bboxes_ignore'] = gt_bboxes_ignore
        results['gt_abboxes_ignore'] = gt_abboxes_ignore
        # adding the fields
        results['bbox_fields'] = ['gt_bboxes_ignore', 'gt_bboxes']
        results['abbox_fields'] = ['gt_bboxes_ignore', 'gt_abboxes']
        results['mask_fields'] = ['gt_masks']
        results['amask_fields'] = ['gt_amasks']
        results['seg_fields'] = ['gt_semantic_seg']
        results['pan_fields'] = ['gt_panoptic_only_thing_classes']
        results['maxinst_fields'] = ['max_inst_per_class']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ignore_label={self.ignore_label}, ' \
                    f'(thing_list={self.thing_list}, ' \
                    f'(ignore_stuff_in_offset={self.ignore_stuff_in_offset}, ' \
                    f'(small_instance_area={self.small_instance_area}, ' \
                    f'(small_instance_weight={self.small_instance_weight}, ' \
                    f'(sigma={self.sigma}, ' \
                    f'(g={self.g}, '
        return repr_str

@PIPELINES.register_module()
class GenBlendPASSAmodalMixedLabelsForMaskFormer(object):

    def __init__(self, sigma, mode, hardlabel_dir=None, num_hard_label=5, num_classes=19, gen_instance_classids_from_zero=True):
        self.ignore_label = 255
        self.label_divisor = 1000
        self.thing_list = [11, 12, 13, 14, 15, 16, 17]
        self.thing_list_mapids = {11:0, 12:1, 13:2, 14:3, 15:4, 16:5, 17:6}
        self.ignore_stuff_in_offset = True
        self.small_instance_area = 4096 # not using currently
        self.small_instance_weight = 3  # not using currently
        self.ignore_crowd_in_semantic = True
        self.ignore_crowd_in_instance = True
        self.sigma = sigma
        self.mode = mode
        self.num_classes = num_classes
        self.gen_instance_classids_from_zero = gen_instance_classids_from_zero
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        if hardlabel_dir is not None:
            self.hardlables_list = glob.glob(os.path.join(hardlabel_dir, '*', '*.png'))
            self.hardlables_list.sort()
            self.num_hard_label = num_hard_label

    def _map_instance_class_ids(self, catid):
        if self.gen_instance_classids_from_zero:
            return self.thing_list_mapids[catid]
        else:
            return catid

    def __call__(self, results):
        semantic = results['gt_semantic']
        instances = results['gt_instance']
        ainstances = results['gt_ainstance']

        ## mixed ADCL pool
        hard_label_randoms = random.sample(self.hardlables_list, self.num_hard_label*10)
        crop_y1, crop_y2, crop_x1, crop_x2 = results['crop_bbox']
        croped_hard_label_list = []
        croped_hard_class_list = []
        croped_hard_img_list = []
        hard_label_crop_num = 1  # hard label count
        hard_label_sem = np.zeros_like(semantic)  # semantic labels
        hard_label_sum = np.zeros_like(semantic)  # hard label mask and count
        for hard_label_name in hard_label_randoms:
            if 'cer' in hard_label_name:
                continue
            hard_label = np.array(Image.open(hard_label_name), dtype=np.uint8)

            hard_label_crop = hard_label[crop_y1:crop_y2, crop_x1:crop_x2]

            if np.sum(hard_label_crop) != 0:
                if np.count_nonzero(hard_label_crop) * 8 > hard_label_crop.size:
                    continue
                ins_num, ins_cnt = np.unique(hard_label_sum[hard_label_crop == 0], return_counts=True)
                if len(ins_num)!=hard_label_crop_num or min(ins_cnt)<25:
                    continue
                hard_label_class = hard_label_name.split('/')[-1].split('_')[1][3:]
                hard_label_class = self.thing_list[int(hard_label_class)]
                hard_img = np.array(Image.open(os.path.join(results['img_prefix'], hard_label_name.split('/')[-2],
                                                            hard_label_name.split('/')[-1].split('_')[0] + '.jpg')),
                                    dtype=np.uint8)
                hard_img_crop = hard_img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
                croped_hard_label_list.append(hard_label_crop)
                croped_hard_class_list.append(hard_label_class)
                croped_hard_img_list.append(hard_img_crop[:,:,::-1])
                hard_label_sem[hard_label_crop != 0] = hard_label_class
                hard_label_sum[hard_label_crop != 0] = hard_label_crop_num
                hard_label_crop_num += 1
            if hard_label_crop_num > self.num_hard_label:
                break

        ## mixed
        semantic[hard_label_sem != 0] = hard_label_sem[hard_label_sem != 0]
        for h_img, h_label in zip(croped_hard_img_list, croped_hard_label_list):
            results['img'][h_label == 1] = h_img[h_label==1]
            results['img'][h_label == 255] = 0

        height, width = semantic.shape[0], semantic.shape[1]

        panoptic_only_thing_classes = np.zeros(semantic.shape)
        max_inst_per_class = np.zeros(len(self.thing_list))
        class_id_tracker = {}
        for cid in self.thing_list:
            class_id_tracker[cid] = 1
        gt_masks = []
        gt_labels = []
        gt_bboxes = []
        gt_bboxes_ignore = np.empty([0, 4], dtype=np.float32)
        for cat_id, mask in instances.items():
            mask[hard_label_sem != 0] = 0
            if not (mask==1).sum() == 0:
                cat_class = cat_id // self.label_divisor
                if cat_class in self.thing_list:
                    box = get_bbox_coord(mask)
                    if isValidBox(box):
                        gt_bboxes.append(box)
                        gt_masks.append(mask.astype(np.uint8))
                        gt_labels.append(self._map_instance_class_ids(cat_class))
                        panoptic_only_thing_classes[mask == 1] = cat_class * self.label_divisor + class_id_tracker[cat_class]
                        class_id_tracker[cat_class] += 1

        gt_amasks = []
        gt_alabels = []
        gt_abboxes = []
        gt_abboxes_ignore = np.empty([0, 4], dtype=np.float32)
        for acat_id, amask in ainstances.items():
            if not (amask[hard_label_sem == 0] ==1).sum() == 0:
                acat_class = acat_id // self.label_divisor
                if acat_class in self.thing_list:
                    abox = get_bbox_coord(amask)
                    if isValidBox(abox):
                        gt_abboxes.append(abox)
                        gt_amasks.append(amask.astype(np.uint8))
                        gt_alabels.append(self._map_instance_class_ids(acat_class))

        hard_ins_mask = np.zeros_like(semantic)
        for h_cls, h_mask in zip(reversed(croped_hard_class_list), reversed(croped_hard_label_list)):
            h_mask[h_mask!=0] = 1
            if not (h_mask != 0).sum() == 0:
                if h_cls in self.thing_list:
                    h_box = get_bbox_coord(h_mask)
                    if isValidBox(h_box):

                        gt_abboxes.append(h_box)
                        gt_amasks.append(h_mask.astype(np.uint8))
                        gt_alabels.append(self._map_instance_class_ids(h_cls))

                        h_mask[hard_ins_mask != 0] = 0
                        gt_bboxes.append(h_box)
                        gt_masks.append(h_mask.astype(np.uint8))
                        gt_labels.append(self._map_instance_class_ids(h_cls))
                        panoptic_only_thing_classes[h_mask == 1] = h_cls * self.label_divisor + class_id_tracker[h_cls]
                        class_id_tracker[h_cls] += 1
                        hard_ins_mask[h_mask != 0] = 1
        for cid in list(class_id_tracker.keys()):
            max_inst_per_class[self._map_instance_class_ids(cid)] = class_id_tracker[cid]
        gt_masks = BitmapMasks(gt_masks, height, width)
        gt_amasks = BitmapMasks(gt_amasks, height, width)
        results['gt_masks'] = gt_masks
        results['gt_amasks'] = gt_amasks
        results['gt_semantic_seg'] = semantic.astype('long')
        results['gt_panoptic_only_thing_classes'] = panoptic_only_thing_classes.astype('long')
        results['gt_labels'] = np.asarray(gt_labels).astype('long')
        results['gt_alabels'] = np.asarray(gt_alabels).astype('long')
        results['max_inst_per_class'] = max_inst_per_class.astype('long')
        results['gt_bboxes'] = np.asarray(gt_bboxes).astype(np.float32)
        results['gt_abboxes'] = np.asarray(gt_abboxes).astype(np.float32)
        results['gt_bboxes_ignore'] = gt_bboxes_ignore
        results['gt_abboxes_ignore'] = gt_abboxes_ignore
        # adding the fields
        results['bbox_fields'] = ['gt_bboxes_ignore', 'gt_bboxes']
        results['abbox_fields'] = ['gt_bboxes_ignore', 'gt_abboxes']
        results['mask_fields'] = ['gt_masks']
        results['amask_fields'] = ['gt_amasks']
        results['seg_fields'] = ['gt_semantic_seg']
        results['pan_fields'] = ['gt_panoptic_only_thing_classes']
        results['maxinst_fields'] = ['max_inst_per_class']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(ignore_label={self.ignore_label}, ' \
                    f'(thing_list={self.thing_list}, ' \
                    f'(ignore_stuff_in_offset={self.ignore_stuff_in_offset}, ' \
                    f'(small_instance_area={self.small_instance_area}, ' \
                    f'(small_instance_weight={self.small_instance_weight}, ' \
                    f'(sigma={self.sigma}, ' \
                    f'(g={self.g}, '
        return repr_str
