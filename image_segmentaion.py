#!/usr/bin/env python

# pylint: skip-file
import numpy as np
import mxnet as mx
from PIL import Image
import time
from pprint import pprint

# def getpallete(num_cls):
#     # this function is to get the colormap for visualizing the segmentation mask
#     n = num_cls
#     pallete = [0]*(n*3)
#     for j in xrange(0,n):
#         lab = j
#         pallete[j*3+0] = 0
#         pallete[j*3+1] = 0
#         pallete[j*3+2] = 0
#         i = 0
#         while (lab > 0):
#             pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
#             pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
#             pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
#             i = i + 1
#             lab >>= 3
#     return pallete
# pallete = getpallete(256)
# print(np.array(pallete).reshape((256,3))[:21,:])

import cv2
from utils import getpallete
palette = np.array(getpallete(256)).reshape((256,3))
# palette = np.loadtxt("VOC2012/palette.txt")
# palette = np.array(Image.open("000129_.png").getpalette()).reshape((256,3))

color2index = {tuple(p):idx for idx,p in enumerate(palette)} # (255, 255, 255) : 0,
index2color = {idx:p.tolist() for idx,p in enumerate(palette)} # (255, 255, 255) : 0,

# print index2color

# img = "./person_bicycle.jpg"
# img = "./000129.jpg"
# img = "./003546.jpg"
# img = "./000065.jpg"
img = "./001311.jpg"
segfile = img.replace("jpg", "png")
model_prefix = "models/FCN8s_ResNet_VOC2012"
epoch = 7
ctx = mx.gpu(0)

def get_data(img_path):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    img = np.array(Image.open(img_path), dtype=np.float32)
    print img[:2,:2,:]
    # img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB).astype(np.float32)
    # print img[:2,:2,:]
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    fcnxs, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(model_prefix, epoch)
    fcnxs_args = {key: val.as_in_context(ctx) for key, val in fcnxs_args.items()}
    fcnxs_auxs = {key: val.as_in_context(ctx) for key, val in fcnxs_auxs.items()}
    # pprint(fcnxs_args)
    # pprint(fcnxs_auxs)
    fcnxs_args["data"] = mx.nd.array(get_data(img), ctx)
    data_shape = fcnxs_args["data"].shape
    label_shape = (1, data_shape[2]*data_shape[3])
    fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    exector = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_auxs, group2ctx={'gpu(0)':mx.gpu(0),})
    tic = time.time()
    exector.forward(is_train=False)
    output = exector.outputs[0].asnumpy()
    print output.shape
    out_img = np.squeeze(output.argmax(axis=1).astype(np.uint8))
    print out_img
    print 'elapsed: %.0f ms' % (1000.*(time.time()-tic),)

    h, w = out_img.shape
    out_img = map(lambda x:index2color[x[0]],out_img.reshape((h*w,1)).tolist())
    out_img = np.array(out_img).reshape((h,w,3)).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(segfile,out_img)
    
    # out_img = Image.fromarray(out_img)
    # print out_img
    # out_img.putpalette(palette)
    # out_img.save(segfile)

    # print index2color
    # h, w = out_img.shape
    # print out_img.reshape((h*w,)).tolist()
    # out_img = map(lambda x:index2color[x],out_img.reshape((h*w,)).tolist())
    # out_img = np.array(out_img).reshape((h,w,3))


if __name__ == "__main__":
    main()

