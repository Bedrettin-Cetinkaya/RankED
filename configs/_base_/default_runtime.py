# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
#checkpoint_config = dict(interval=100, by_epoch=False)
#evaluation = dict(interval=100)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
#test_cfg = dict(mode='slide', crop_size=(320, 320), stride=(280, 280))
