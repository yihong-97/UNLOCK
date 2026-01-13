# ---------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

debug = False
_base_ = [
		"../_base_/default_runtime_mmdet_mr.py",
		"../_base_/models/maskrcnn_sepaspp_amodalpassb5.py",
		"../_base_/datasets/source-only_kitti360aps_to_blendpass_maskrcnn_oass.py",
		"../_base_/schedules/adamw.py",
		"../_base_/schedules/poly10warm.py"
]

n_gpus = 1
gpu_mtotal = 23000
total_train_time =  "21:00:00"
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
dump_predictions_to_disk = True  ## Genearate all predictions
evaluate_from_saved_png_predictions = False
seed = 0
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
blendpass_test_pipeline = [
                            dict(type='LoadImageFromFile'),
                            dict(type='MultiScaleFlipAug',
                                 img_scale=(2048, 400),
                                 flip=False,
                                 transforms=[dict(type='Resize', keep_ratio=True),
                                             dict(type='RandomFlip'),
                                             dict(type='Normalize', **img_norm_cfg),
                                             dict(type='ImageToTensor', keys=['img']),
                                             dict(type='Collect', keys=['img']),
                                             ])
                        ]

model = dict(type='MaskRCNNPanoptic_PL_ALL')

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(),
    val=dict(
            data_root="../../../../datasets/BlendPASS",
            img_dir='leftImg8bit/train',
            depth_dir='',
            pano_dir='',
            ann_file='',
            pipeline=blendpass_test_pipeline)
)



runner = dict(
    type="IterBasedRunner",
    max_iters= 20000
)

checkpoint_config = dict(
    by_epoch=False,
    interval=2000,
    max_keep_ckpts=10
)

log_config = dict(
    interval=50,
)
name = "Sourceonly_UNLOCK"
exp = 6
exp_root = "unmaskformer_experiments"
exp_sub = "exp-00006"
name_dataset = "kitti360aps2blendpass"
name_architecture = "maskrcnn_amodalpassb5"
name_encoder = "amodalpassb5"
name_decoder = "maskrcnn"
name_uda = "source-only"
name_opt = "adamw_6e-05_pmTrue_poly10warm_1x1_40k"
checkpoint_path = None,
checkpoint_file_path = "./pretrained_edaps/source_trained_model.pth"
# checkpoint_file_path = "/home/yihong/data2/cyh2/project/2405_SFDA_Amodal/UnmaskFormer_SFDA/pretrained_edaps/source_trained_model.pth"
work_dir = "Sourceonly_test_best"
git_rev = ""


evaluation = dict(
    interval=2000,
    metric=["mIoU", "mPQ", "mAP", "mAAP", "mAPQ"],
    eval_type="maskrcnn_oass_pl",
    dataset_name="blendpass",
    gt_dir= None,
    gt_dir_insta= None,
    gt_dir_panop= None,
    num_samples_debug=12,
    post_proccess_params=dict(
        num_classes=18,
        ignore_label=255,
        mapillary_dataloading_style="OURS",
        label_divisor=1000,
        train_id_to_eval_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        thing_list=[11, 12, 13, 14, 15, 16, 17],
        mask_score_threshold=0.95,
        amask_score_threshold=0.95,
        debug=False,
        dump_semantic_pred_as_numpy_array=False,
        load_semantic_pred_as_numpy_array=False,
        semantic_pred_numpy_array_location=None,
        dump_oass_pred_labels_as_numpy_array=True,
        delete_oass_all_save_numpy_results=True,
        use_semantic_decoder_for_instance_labeling=False,
        use_semantic_decoder_for_panoptic_labeling=False,
        nms_th=None,
        intersec_th=None,
        upsnet_mask_pruning=False,
        generate_thing_cls_panoptic_from_instance_pred=False
    ),
    visuals_pan_eval=True,
    visuals_all_eval=False,
    evalScale="2048x400",
    panop_eval_folder= name + "/panoptic_eval",
    panop_eval_temp_folder= name + "/panoptic_eval",
    debug=False,
    out_dir= name + "/panoptic_eval/visuals"
)