# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

debug = False
_base_ = [
		"../_base_/default_runtime_mmdet_mr.py",
		"../_base_/models/maskrcnn_sepaspp_amodalpassb5.py",
		"../_base_/datasets/target-only_kitti360aps_to_blendpass_maskrcnn_oass.py",
		"../_base_/schedules/adamw.py",
		"../_base_/schedules/poly10warm.py"
]

n_gpus = 1
gpu_mtotal = 23000
total_train_time = "21:00:00"
n_cpus = 16
mem_per_cpu = 16000
machine = "local"
resume_from = None
load_from = None
only_eval = False
only_train = False
activate_auto_scale_lr = False
auto_scale_lr = dict(enable=False, base_batch_size=16)
print_layer_wise_lr = False
file_sys = "Slurm"
launcher = None
generate_only_visuals_without_eval = False
seed = 0

data_root = '../datasets/BlendPASS'
pse_lab_root = './unmaskformer_experiments/exp-00006/Sourceonly_UNLOCK/oass_eval'


img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (400, 400)
num_classes = 18
blendpass_train_mix_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='LoadAmodalSoftLabels'),
                            dict(type='Resize', img_scale=(2048, 400)),
                            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='GenBlendPASSAmodalMixedLabelsForMaskFormer',
                                 sigma=8,
                                 mode='train',
                                hardlabel_dir=pse_lab_root + '/ADCL/amodal_instance',
                                num_hard_label=8,
                                 num_classes=num_classes,
                                 gen_instance_classids_from_zero=True,
                                 ),
                            dict(type='Normalize', **img_norm_cfg),
                            dict(type='DefaultFormatBundleMmdet'), dict(type='Collect',
                                                                        keys=['img', 'gt_bboxes', 'gt_abboxes',
                                                                              'gt_labels', 'gt_alabels', 'gt_masks', 'gt_amasks',
                                                                              'gt_semantic_seg',
                                                                              'gt_panoptic_only_thing_classes',
                                                                              'max_inst_per_class']), ]

blendpass_train_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='Resize', img_scale=(2048, 400)),
                            dict(type='RandomCrop', crop_size=crop_size),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='Normalize', **img_norm_cfg),
                            dict(type='DefaultFormatBundleMmdet'),
                            dict(type='Collect', keys=['img']),
                        ]


sfda = dict(
    type='SFDA',
    sourcemodel_path = "./pretrained_edaps/source_trained_model.pth",
    alpha=0.999,
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    debug_img_interval=100,
    print_grad_magnitude=False,
    share_src_backward=False,
    disable_mix_masks=False,
    pseudo_weight_ignore_top=11,
    pseudo_weight_ignore_bottom=88,
    pseudo_threshold=0.968,
)


data = dict(
	    samples_per_gpu=4, # batchsize (2 source  + 2 target images)
        workers_per_gpu=4,
        train=dict(type='SFDADataset',
                    source=dict(
                        type='BlendPASS',
                        data_root=data_root,
                        img_dir='leftImg8bit/train',
                        pse_lab_dir=pse_lab_root + '/OPLL',
                        pipeline=blendpass_train_mix_pipeline),
                    target=dict(
                        type='BlendPASS',
                        data_root=data_root,
                        img_dir='leftImg8bit/train',
                        pipeline=blendpass_train_pipeline,
                    ),),
        val=dict(
                pano_dir="Panoptic/val_trainId_K2B",
                data_root=data_root
        )
)
optimizer_config = None

optimizer = dict(
        lr=1e-07,
        paramwise_cfg=dict(
            custom_keys=dict(
                neck=dict(lr_mult=10.0), head=dict(lr_mult=0.1), decode_head=dict(lr_mult=10000.0), pos_block=dict(decay_mult=0.0), norm=dict(decay_mult=0.0)
            )
        )
)

evaluation = dict(
    interval=200,
    metric=["mIoU", "mPQ", "mAP", "mAAP", "mAPQ"],
    eval_type="maskrcnn_oass",
    dataset_name="blendpass",
    gt_dir=data_root + "/Semantic/val_K2B",
    gt_dir_insta=data_root + "/Instance/val_K2B",
    gt_dir_panop=data_root + "/Panoptic/",
    num_samples_debug=12,
    post_proccess_params=dict(
        num_classes=18,
        ignore_label=255,
        mapillary_dataloading_style="OURS",
        label_divisor=1000,
        train_id_to_eval_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        thing_list=[11, 12, 13, 14, 15, 16, 17],
        mask_score_threshold=0.55,
        amask_score_threshold=0.5,
        debug=False,
        dump_semantic_pred_as_numpy_array=False,
        load_semantic_pred_as_numpy_array=False,
        semantic_pred_numpy_array_location=None,
        use_semantic_decoder_for_instance_labeling=False,
        use_semantic_decoder_for_panoptic_labeling=False,
        nms_th=None,
        intersec_th=None,
        upsnet_mask_pruning=False,
        generate_thing_cls_panoptic_from_instance_pred=False
    ),
    visuals_pan_eval=False,
    visuals_all_eval=False,
    evalScale="2048x400",
    panop_eval_folder="Targetonly_train/panoptic_eval",
    panop_eval_temp_folder="Targetonly_train/panoptic_eval",
    debug=False,
    out_dir="Targetonly_train/panoptic_eval/visuals"
)

runner = dict(
    type="IterBasedRunner",
    max_iters=10000
)

checkpoint_config = dict(
    by_epoch=False,
    interval=10000,
    max_keep_ckpts=1
)

log_config = dict(
    interval=50,
)


name = "UNLOCK"
exp = 4
exp_root = "unmaskformer_experiments"
exp_sub = "exp-00004"
name_dataset = "kitti360aps2blendpass",

name_architecture = "maskrcnn_amodalpassb5",
name_encoder = "amodalpassb5",
name_decoder = "maskrcnn",
name_uda = "target-only",
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x4_40k",
work_dir = "Targetonly_train",
git_rev = ""

