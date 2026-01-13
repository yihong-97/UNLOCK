# learning policy
# lr_config = dict(policy='poly')
# lr_config = dict(policy='poly', warmup='linear',  warmup_iters=10, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)
