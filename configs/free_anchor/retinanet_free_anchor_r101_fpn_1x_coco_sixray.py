_base_ = './retinanet_free_anchor_r50_fpn_1x_coco_sixray.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))