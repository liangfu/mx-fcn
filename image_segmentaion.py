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

# imgname = "./person_bicycle.jpg"
# imgname = "./000129.jpg"
# imgname = "./000143.jpg"
# imgname = "./003546.jpg"
imgname = "./000034_resized.jpg"
# imgname = "./000065.jpg"
# imgname = "./000058.jpg"
# imgname = "./001311.jpg"
# imgname = "./2011_003240.jpg"
model_prefix = "models/FCN32s_ResNet_Cityscapes"
epoch = 5
ctx = mx.gpu(0)

def get_data(img_path):
    """get the (1, 3, h, w) np.array data for the img_path"""
    mean = np.array([123.68, 116.779, 103.939])  # (R,G,B)
    # img = np.array(Image.open(img_path), dtype=np.float32)
    # print img[:2,:2,:]
    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    # print img[:2,:2,:]

    # img = cv2.resize(img, (640, 480)).astype(np.float32)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    
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

    # for imgidx in range(1,21):
    #     imgname = "/home/liangfu/workspace/mx-rcnn/data/JPEGImages/%06d.jpg" % (imgidx,)

    segfile = imgname.replace("jpg", "png")
    fcnxs_args["data"] = mx.nd.array(get_data(imgname), ctx)
    data_shape = fcnxs_args["data"].shape
    label_shape = (1, data_shape[2]*data_shape[3])
    fcnxs_args["softmax_label"] = mx.nd.empty(label_shape, ctx)
    executor = fcnxs.bind(ctx, fcnxs_args ,args_grad=None, grad_req="null", aux_states=fcnxs_auxs, group2ctx={'gpu(0)':mx.gpu(0),})

    def stat_helper(name, array):
        """wrapper for executor callback"""
        import ctypes
        from mxnet.ndarray import NDArray
        from mxnet.base import NDArrayHandle, py_str
        array = ctypes.cast(array, NDArrayHandle)
        array = NDArray(array, writable=False)
        array.wait_to_read()
        elapsed = float(time.time()-stat_helper.start_time)*1000.
        if elapsed>1.:
            print (name, array.shape, ('%.1fms' % (elapsed,)))
        # stat_helper.start_time=time.time()
    stat_helper.start_time=float(time.time())
    executor.set_monitor_callback(stat_helper)

    tic = time.time()
    executor.forward(is_train=False)
    output = executor.outputs[0].asnumpy()
    out_img = np.squeeze(output.argmax(axis=1).astype(np.uint8))
    print('result wrote to %s' % (segfile,)),
    print('elapsed: %.0f ms' % (1000.*(time.time()-tic),))
    h, w = out_img.shape
    out_img = map(lambda x:index2color[x[0]],out_img.reshape((h*w,1)).tolist())
    out_img = np.array(out_img).reshape((h,w,3)).astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    # out_img = cv2.resize(out_img, (800, 600), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(segfile,out_img)


if __name__ == "__main__":
    main()

