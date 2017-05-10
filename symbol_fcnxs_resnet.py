import mxnet as mx
# import proposal
# import proposal_target
# from rcnn.config import config
from symbol_fcnxs_vgg16 import offset

eps = 2e-5
use_global_stats = False #True

# workspace = 512
# res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
# units = res_deps['101']
# filter_list = [256, 512, 1024, 2048]

workspace = 512
units = [2, 2, 2, 2]
filter_list = [64, 128, 256, 512]


def residual_unit(data, num_filter, stride, dim_match, name):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv2, shortcut], name=name + '_plus')
    return sum


def get_resnet_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0 = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0 = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True, name='stage1_unit%s' % i)

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='stage2_unit%s' % i)
    res3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='stage3_unit%s' % i)
    res4 = unit

    # res5
    unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
    for i in range(2, units[3] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)

    # bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
    # relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')

    return res3, res4, unit

# 
# def get_resnet_train(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
#     data = mx.symbol.Variable(name="data")
#     im_info = mx.symbol.Variable(name="im_info")
#     gt_boxes = mx.symbol.Variable(name="gt_boxes")
#     rpn_label = mx.symbol.Variable(name='label')
#     rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
#     rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')
# 
#     # shared convolutional layers
#     conv_feat = get_resnet_conv(data)
# 
#     # RPN layers
#     rpn_conv = mx.symbol.Convolution(
#         data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
#     rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
#     rpn_cls_score = mx.symbol.Convolution(
#         data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
#     rpn_bbox_pred = mx.symbol.Convolution(
#         data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
# 
#     # prepare rpn data
#     rpn_cls_score_reshape = mx.symbol.Reshape(
#         data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
# 
#     # classification
#     rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
#                                            normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
#     # bounding box regression
#     rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
#     rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)
# 
#     # ROI proposal
#     rpn_cls_act = mx.symbol.SoftmaxActivation(
#         data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
#     rpn_cls_act_reshape = mx.symbol.Reshape(
#         data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
#     if config.TRAIN.CXX_PROPOSAL:
#         rois = mx.symbol.Proposal(
#             cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
#             feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
#             rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
#             threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
#     else:
#         rois = mx.symbol.Custom(
#             cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
#             op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
#             scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
#             rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
#             threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)
# 
#     # ROI proposal target
#     gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
#     group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
#                              num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
#                              batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
#     rois = group[0]
#     label = group[1]
#     bbox_target = group[2]
#     bbox_weight = group[3]
# 
#     # Fast R-CNN
#     roi_pool = mx.symbol.ROIPooling(
#         name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
# 
#     # res5
#     unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
#     for i in range(2, units[3] + 1):
#         unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
#     bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
#     relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
#     pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
# 
#     # classification
#     cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)
#     cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
#     # bounding box regression
#     bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)
#     bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
#     bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
# 
#     # reshape output
#     label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
#     cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
#     bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')
# 
#     group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
#     return group
# 
# 
# def get_resnet_test(num_classes=config.NUM_CLASSES, num_anchors=config.NUM_ANCHORS):
#     data = mx.symbol.Variable(name="data")
#     im_info = mx.symbol.Variable(name="im_info")
# 
#     # shared convolutional layers
#     conv_feat = get_resnet_conv(data)
# 
#     # RPN
#     rpn_conv = mx.symbol.Convolution(
#         data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
#     rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
#     rpn_cls_score = mx.symbol.Convolution(
#         data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
#     rpn_bbox_pred = mx.symbol.Convolution(
#         data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")
# 
#     # ROI Proposal
#     rpn_cls_score_reshape = mx.symbol.Reshape(
#         data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
#     rpn_cls_prob = mx.symbol.SoftmaxActivation(
#         data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
#     rpn_cls_prob_reshape = mx.symbol.Reshape(
#         data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
#     if config.TEST.CXX_PROPOSAL:
#         rois = mx.symbol.Proposal(
#             cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
#             feature_stride=config.RPN_FEAT_STRIDE, scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
#             rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
#             threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
#     else:
#         rois = mx.symbol.Custom(
#             cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
#             op_type='proposal', feat_stride=config.RPN_FEAT_STRIDE,
#             scales=tuple(config.ANCHOR_SCALES), ratios=tuple(config.ANCHOR_RATIOS),
#             rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
#             threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
# 
#     # Fast R-CNN
#     roi_pool = mx.symbol.ROIPooling(
#         name='roi_pool5', data=conv_feat, rois=rois, pooled_size=(14, 14), spatial_scale=1.0 / config.RCNN_FEAT_STRIDE)
# 
#     # res5
#     unit = residual_unit(data=roi_pool, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='stage4_unit1')
#     for i in range(2, units[3] + 1):
#         unit = residual_unit(data=unit, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='stage4_unit%s' % i)
#     bn1 = mx.sym.BatchNorm(data=unit, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn1')
#     relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
#     pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
# 
#     # classification
#     cls_score = mx.symbol.FullyConnected(name='cls_score', data=pool1, num_hidden=num_classes)
#     cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
#     # bounding box regression
#     bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=pool1, num_hidden=num_classes * 4)
# 
#     # reshape output
#     cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
#     bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')
# 
#     # group output
#     group = mx.symbol.Group([rois, cls_prob, bbox_pred])
#     return group
# 

def resnet_score(_input, numclass, workspace_default=1024):
    # # group 5
    # conv5_1 = mx.symbol.Convolution(data=input, kernel=(3, 3), pad=(1, 1), num_filter=512,
    #             workspace=workspace_default, name="conv5_1")
    # relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    # conv5_2 = mx.symbol.Convolution(data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512,
    #             workspace=workspace_default, name="conv5_2")
    # relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    # conv5_3 = mx.symbol.Convolution(data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512,
    #             workspace=workspace_default, name="conv5_3")
    # relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    # pool5 = mx.symbol.Pooling(data=relu5_3, pool_type="max", kernel=(2, 2), stride=(2,2), name="pool5")
    # # group 6
    # fc6 = mx.symbol.Convolution(data=pool5, kernel=(7, 7), num_filter=4096,
    #             workspace=workspace_default, name="fc6")
    # relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # # group 7
    # fc7 = mx.symbol.Convolution(data=drop6, kernel=(1, 1), num_filter=4096,
    #             workspace=workspace_default, name="fc7")
    # relu7 = mx.symbol.Activation(data=_input, act_type="relu", name="relu7")
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # group 8
    score = mx.symbol.Convolution(data=_input, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score")
    return score

def fcnxs_score(_input, crop, offset, kernel=(64,64), stride=(32,32), numclass=21, workspace_default=1024):
    # score out
    bigscore = mx.symbol.Deconvolution(data=_input, kernel=kernel, stride=stride, adj=(stride[0]-1, stride[1]-1),
               num_filter=numclass, workspace=workspace_default, name="bigscore")
    upscore = mx.symbol.Crop(*[bigscore, crop], offset=offset, name="upscore")
    # upscore = mx.symbol.Crop(*[input, crop], offset=offset, name="upscore")
    softmax = mx.symbol.SoftmaxOutput(data=upscore, multi_output=True, use_ignore=True, ignore_label=255, name="softmax")
    return softmax

def get_fcn32s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    # pool3 = resnet_pool3(data, workspace_default)
    # pool4 = resnet_pool4(pool3, workspace_default)
    res3, res4, conv_feat = get_resnet_conv(data)
    score = resnet_score(conv_feat, numclass, workspace_default)
    softmax = fcnxs_score(score, data, offset()["fcn32s_upscore"], (64,64), (32,32), numclass, workspace_default)
    return softmax

def get_fcn16s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    # pool3 = vgg16_pool3(data, workspace_default)
    # pool4 = vgg16_pool4(pool3, workspace_default)
    # score = vgg16_score(pool4, numclass, workspace_default)
    res3, res4, conv_feat = get_resnet_conv(data)
    score = resnet_score(conv_feat, numclass, workspace_default)
    # score 2X
    score2 = mx.symbol.Deconvolution(data=score, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_filter=numclass,
                                     adj=(1, 1),
                                     workspace=workspace_default, name="score2")  # 2X
    score_pool4 = mx.symbol.Convolution(data=res4, kernel=(1, 1), stride=(1, 1), pad=(1, 1), num_filter=numclass,
                 workspace=workspace_default, name="score_pool4")
    # score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
    score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=(0,0), name="score_pool4c")
    # score_fused = score2 + score_pool4c
    score_fused = mx.sym.ElementWiseSum(*[score2, score_pool4c], name='score_fused')
    softmax = fcnxs_score(score_fused, data, offset()["fcn16s_upscore"], (32, 32), (16, 16), numclass, workspace_default)
    return softmax

def get_fcn8s_symbol(numclass=21, workspace_default=1024):
    data = mx.symbol.Variable(name="data")
    # pool3 = vgg16_pool3(data, workspace_default)
    # pool4 = vgg16_pool4(pool3, workspace_default)
    # score = vgg16_score(pool4, numclass, workspace_default)
    res3, res4, conv_feat = get_resnet_conv(data)
    score = resnet_score(conv_feat, numclass, workspace_default)
    # score 2X
    score2 = mx.symbol.Deconvolution(data=score, kernel=(4, 4), stride=(2, 2),num_filter=numclass,
                adj=(1, 1), workspace=workspace_default, name="score2")  # 2X
    score_pool4 = mx.symbol.Convolution(data=res4, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score_pool4")
    score_pool4c = mx.symbol.Crop(*[score_pool4, score2], offset=offset()["score_pool4c"], name="score_pool4c")
    score_fused = score2 + score_pool4c
    # score 4X
    score4 = mx.symbol.Deconvolution(data=score_fused, kernel=(4, 4), stride=(2, 2),num_filter=numclass,
                adj=(1, 1), workspace=workspace_default, name="score4") # 4X
    score_pool3 = mx.symbol.Convolution(data=res3, kernel=(1, 1), num_filter=numclass,
                workspace=workspace_default, name="score_pool3")
    score_pool3c = mx.symbol.Crop(*[score_pool3, score4], offset=offset()["score_pool3c"], name="score_pool3c")
    score_final = score4 + score_pool3c
    softmax = fcnxs_score(score_final, data, offset()["fcn8s_upscore"], (16, 16), (8, 8), numclass, workspace_default)
    return softmax

