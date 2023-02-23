_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/voc0712.py',
    '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=20)),
    backbone=dict(
        type='FeedbackResNet',
        feedback_type='add',
        frozen_stages=-1, #-1 means none
        init_cfg=None
    ) )
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=8)  # actual epoch = 4 * 3 = 12
# Save best model
#evaluation = dict(save_best='auto')
# Load pretrained model
load_from = 'checkpoints/faster_rcnn_r50fb_fpn_1x_voc0712_20221208.pth'
