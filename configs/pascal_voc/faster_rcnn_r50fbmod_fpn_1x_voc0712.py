_base_ = [
    './faster_rcnn_r50fbadd_fpn_1x_voc0712.py'
]
model = dict(
    backbone=dict(
        feedback_type='mod'
    ) )