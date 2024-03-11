_base_ = [
    '../_base_/models/upernet_swin.py', 
    '../_base_/datasets/bsds.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
model = dict(
    backbone=dict(
        embed_dim=128,
        depths= [2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=1,
        loss_decode=[dict(name='RankLoss',type='RankLoss'), dict(name='SortLoss', type='SortLoss')]
        #loss_decode=dict(type='RankLoss')
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(280, 280))
#test_cfg = dict(mode='whole')

find_unused_parameters = True
#data = dict(samples_per_gpu=1)
#bs = 4
#data = dict(
#    samples_per_gpu=2,
#    workers_per_gpu=1,
#    train=dict(
#        split='image-train_1.txt'))
#data=dict(samples_per_gpu=2)
