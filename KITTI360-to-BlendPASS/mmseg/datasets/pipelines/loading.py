# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp
import mmcv
import numpy as np
from ..builder import PIPELINES
import cv2
from PIL import Image
import pycocotools.mask as mask_utils
from PIL import ImageDraw
from collections import namedtuple
from amodalscripts.helpers.labels import id2label, labels
import glob
@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        # self.diffusion_check_cnt = 0

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0

        num_channels = 1 if len(img.shape) < 3 else img.shape[2]

        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadPanopticAnnotations(object):

    # --------------------------------------------------------------------------------
    # Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
    # Licensed under the Apache License, Version 2.0
    # --------------------------------------------------------------------------------

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        gt_panoptic_seg = Image.open(filename)
        gt_panoptic_seg = np.asarray(gt_panoptic_seg, dtype=np.float32)
        results['gt_panoptic_seg'] = gt_panoptic_seg
        results['seg_fields'].append('gt_panoptic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAmodalAnnotations(object):

    # --------------------------------------------------------------------------------
    # Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
    # Licensed under the Apache License, Version 2.0
    # --------------------------------------------------------------------------------

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        gt_panoptic_seg = Image.open(filename)
        gt_panoptic_seg = np.asarray(gt_panoptic_seg, dtype=np.float32)
        gt_amodal_segs = {}
        for amodal_id, gt_amodal_ann in results["ann_info"]['amodals_info'].items():
            amodal_seg = gt_amodal_ann['segmentation']
            ## coco
            rles = mask_utils.frPyObjects(amodal_seg, gt_panoptic_seg.shape[0], gt_panoptic_seg.shape[1])
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            # ## city
            # Point = namedtuple('Point', ['x', 'y'])
            # polygons = []
            # for part in amodal_seg:
            #     poly = np.array(part).reshape((int(len(part) / 2), 2))
            #     polygons.append([Point(p[0], p[1]) for p in list(poly)])
            # amodalImg = Image.new("I", (gt_panoptic_seg.shape[1], gt_panoptic_seg.shape[0]), 0)
            # drawer = ImageDraw.Draw(amodalImg)
            # for polygon in polygons:
            #     drawer.polygon(polygon, fill=1)
            # amask = np.asarray(amodalImg, dtype=np.float32)

            # ## debug vis
            # out = np.concatenate((mask, amask, mask-amask), axis=0)
            # mask_name = './' +str(np.random.randint(100)) + filename.split('/')[-1]
            # cv2.imwrite(mask_name, out*255)

            gt_amodal_segs[amodal_id] = mask
        results['gt_panoptic_seg'] = gt_panoptic_seg
        results['gt_amodal_segs'] = gt_amodal_segs
        results['seg_fields'].append('gt_panoptic_seg')
        results['amodal_fields']= [('gt_amodal_segs')]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAmodalPseudoLabels_thread(object):

    # --------------------------------------------------------------------------------
    # Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
    # Licensed under the Apache License, Version 2.0
    # --------------------------------------------------------------------------------

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.thing_list = [11, 12, 13, 14, 15, 16, 17]
        self.label_divisor = 1000

    def __call__(self, results):
        gt_semantic_filename = osp.join(results['ann_info']['semlabel_name'], results['img_info']['filename'].replace('.jpg', '.npy'))
        gt_instance_filename = osp.join(results['ann_info']['inslabel_name'], results['img_info']['filename'].replace('.jpg', '.npy'))
        gt_ainstance_filename = osp.join(results['ann_info']['ainslabel_name'], results['img_info']['filename'].replace('.jpg', '.npy'))
        if 'lab' in gt_semantic_filename:
            gt_semantic = np.load(gt_semantic_filename, allow_pickle=True)#.tolist()
        else:
            gt_semantic = np.load(gt_semantic_filename, allow_pickle=True)#.tolist()
            gt_semantic = np.argmax(gt_semantic, axis=0)
        instance = {}
        ins_thread = 0.5
        ains_thread = 0.6
        if 'lab' in gt_instance_filename:
            gt_instance = np.load(gt_instance_filename, allow_pickle=True)#.tolist()
            ins_Ids = np.unique(gt_instance)
            for ins_id in ins_Ids:
                if ins_id >= 1000:
                    if id2label[ins_id // 1000].hasInstances:
                        ins_mask = gt_instance == ins_id
                        instance[ins_id] = ins_mask.astype(np.uint8)
        else:
            gt_instance = np.load(gt_instance_filename, allow_pickle=True).tolist()
            class_id_tracker = {}
            for cid in self.thing_list:
                class_id_tracker[cid] = 1
            for ins in gt_instance:
                if ins['score'] > ins_thread:
                    instance[ins['pred_class'] * self.label_divisor + class_id_tracker[ins['pred_class']]] = ins['pred_mask']
                    class_id_tracker[ins['pred_class']] += 1
        ainstance = {}
        if 'lab' in gt_ainstance_filename:
            gt_ainstance = np.load(gt_ainstance_filename, allow_pickle=True)  # .tolist()
            ains_Ids = np.unique(gt_ainstance)
            for ains_id in ains_Ids:
                if ains_id >= 1000:
                    if id2label[ains_id // 1000].hasInstances:
                        ains_mask = gt_ainstance == ains_id
                        ains_mask = np.sum(ains_mask, axis=0)
                        ainstance[ains_id] = ains_mask.astype(np.uint8)
        else:
            gt_ainstance = np.load(gt_ainstance_filename, allow_pickle=True).tolist()
            aclass_id_tracker = {}
            for acid in self.thing_list:
                aclass_id_tracker[acid] = 1
            for ains in gt_ainstance:
                if ains['score'] > ains_thread:
                    ainstance[ains['pred_class'] * self.label_divisor + aclass_id_tracker[ains['pred_class']]] = ains['pred_mask']
                    aclass_id_tracker[ains['pred_class']] += 1
        results['gt_semantic'] = gt_semantic
        results['gt_instance'] = instance
        results['gt_ainstance'] = ainstance
        results['seg_fields'].append('gt_semantic')
        results['amodal_fields'] = []
        results['amodal_fields'].append('gt_instance')
        results['amodal_fields'].append('gt_ainstance')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAmodalHardLabels_thread(object):

    # --------------------------------------------------------------------------------
    # Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
    # Licensed under the Apache License, Version 2.0
    # --------------------------------------------------------------------------------

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.thing_list = [11, 12, 13, 14, 15, 16, 17]
        self.label_divisor = 1000

    def __call__(self, results):
        gt_semantic_filename = osp.join(results['ann_info']['semlabel_name'], results['img_info']['filename'].replace('.jpg', '.png'))
        gt_semantic = np.array(Image.open(gt_semantic_filename), dtype=np.float32)
        instance = {}
        for i in self.thing_list:
            gt_instance_filename_list_class = glob.glob(osp.join(results['ann_info']['inslabel_name'],
                                                           results['img_info']['filename'].replace('.jpg', '_cls{}_*.png'.format(i-11))))
            if len(gt_instance_filename_list_class):
                class_id_tracker = 0
                for gt_instance_filename in gt_instance_filename_list_class:
                    # ins_mask = (np.array(Image.open(gt_instance_filename), dtype=np.float32) == 1)
                    # instance[i * self.label_divisor + class_id_tracker] = ins_mask.astype(np.uint8)
                    # ins_mask = np.array(Image.open(gt_instance_filename), dtype=np.uint8)
                    instance[i * self.label_divisor + class_id_tracker] = np.array(Image.open(gt_instance_filename),
                                                                                    dtype=np.uint8)
                    class_id_tracker += 1

        ainstance = {}
        for i in self.thing_list:
            gt_ainstance_filename_list_class = glob.glob(osp.join(results['ann_info']['ainslabel_name'],
                                                                 results['img_info']['filename'].replace('.jpg',
                                                                                                         '_cls{}_*.png'.format(
                                                                                                             i - 11))))
            if len(gt_ainstance_filename_list_class):
                class_id_tracker = 0
                for gt_ainstance_filename in gt_ainstance_filename_list_class:
                    # ains_mask = (np.array(Image.open(gt_ainstance_filename), dtype=np.float32) == 1)
                    # ains_mask = np.array(Image.open(gt_ainstance_filename), dtype=np.uint8)
                    ainstance[i * self.label_divisor + class_id_tracker] = np.array(Image.open(gt_ainstance_filename), dtype=np.uint8)
                    class_id_tracker += 1

        results['gt_semantic'] = gt_semantic
        results['gt_instance'] = instance
        results['gt_ainstance'] = ainstance
        results['seg_fields'].append('gt_semantic')
        results['amodal_fields'] = []
        results['amodal_fields'].append('gt_instance')
        results['amodal_fields'].append('gt_ainstance')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class LoadAmodalSoftLabels(object):

    # --------------------------------------------------------------------------------
    # Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
    # Licensed under the Apache License, Version 2.0
    # --------------------------------------------------------------------------------

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.thing_list = [11, 12, 13, 14, 15, 16, 17]
        self.label_divisor = 1000

    def __call__(self, results):
        gt_semantic_filename = osp.join(results['ann_info']['semlabel_name'], results['img_info']['filename'].replace('.jpg', '.png'))
        gt_semantic = np.array(Image.open(gt_semantic_filename), dtype=np.float32)
        instance = {}
        if osp.exists(osp.join(results['ann_info']['inslabel_name'],
                                                           results['img_info']['filename'].replace('.jpg', '_uncertain.png'))):
            ins_uncer = np.array(Image.open(osp.join(results['ann_info']['inslabel_name'],
                                                               results['img_info']['filename'].replace('.jpg', '_uncertain.png'))))
        for i in self.thing_list:
            gt_instance_filename_list_class = glob.glob(osp.join(results['ann_info']['inslabel_name'],
                                                           results['img_info']['filename'].replace('.jpg', '_cls{}_*.png'.format(i-11))))
            if len(gt_instance_filename_list_class):
                class_id_tracker = 0
                for gt_instance_filename in gt_instance_filename_list_class:
                    ins_mask = np.array(Image.open(gt_instance_filename), dtype=np.uint8)
                    instance[i * self.label_divisor + class_id_tracker] = ins_mask
                    instance[i * self.label_divisor + class_id_tracker][(ins_uncer == 255) & (ins_mask != 1)] = 255
                    class_id_tracker += 1

        ainstance = {}
        if osp.exists(osp.join(results['ann_info']['ainslabel_name'],
                               results['img_info']['filename'].replace('.jpg', '_uncertain.png'))):
            ains_uncer = np.array(Image.open(osp.join(results['ann_info']['ainslabel_name'],
                                                     results['img_info']['filename'].replace('.jpg', '_uncertain.png'))))
        for i in self.thing_list:
            gt_ainstance_filename_list_class = glob.glob(osp.join(results['ann_info']['ainslabel_name'],
                                                                 results['img_info']['filename'].replace('.jpg',
                                                                                                         '_cls{}_*.png'.format(
                                                                                                             i - 11))))
            if len(gt_ainstance_filename_list_class):
                class_id_tracker = 0
                for gt_ainstance_filename in gt_ainstance_filename_list_class:
                    ains_mask = np.array(Image.open(gt_ainstance_filename), dtype=np.uint8)
                    ainstance[i * self.label_divisor + class_id_tracker] = ains_mask
                    ainstance[i * self.label_divisor + class_id_tracker][(ains_uncer == 255) & (ains_mask!= 1)] = 255
                    class_id_tracker += 1

        results['gt_semantic'] = gt_semantic
        results['gt_instance'] = instance
        results['gt_ainstance'] = ainstance
        results['seg_fields'].append('gt_semantic')
        results['amodal_fields'] = []
        results['amodal_fields'].append('gt_instance')
        results['amodal_fields'].append('gt_ainstance')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class LoadDepthAnnotations(object):
    def __init__(self):
        self.max_depth_val = 65536.0

    def __call__(self, results):
        if not results['depth_prefix'] == '':
            file = osp.join(results['depth_prefix'], results['img_info']['filename'])
            gt_depth_map = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
            # gt_depth_map = cv2.resize(gt_depth_map, tuple(labels_size), interpolation=cv2.INTER_NEAREST)
            gt_depth_map = self.max_depth_val / (gt_depth_map + 1)  # inverse depth
        else:
            # for cityscapes, create a dummy depth map
            gt_depth_map = np.zeros((1024, 2048), dtype=float)
        results['gt_depth_map'] = gt_depth_map
        results['seg_fields'].append('gt_depth_map')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_depth_val={self.max_depth_val},'
        return repr_str

