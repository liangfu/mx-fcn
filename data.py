# pylint: skip-file
""" file iterator for pasval voc 2012

>>> from data import FileIter
>>> iter = FileIter(root_dir="Cityscapes",flist_name="train.lst",rgb_mean=(123.68, 116.779, 103.939))
>>> print iter.provide_data
[('data', (32, 3, 512, 1024))]
>>> print iter.provide_label
[('softmax_label', (32, 1, 512, 1024))]
"""


import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image
import cv2
# import random
from mxnet.image import ImageIter
import time
from utils import getpallete
palette = np.array(getpallete(256)).reshape((256,3))
color2index = {tuple(p):idx for idx,p in enumerate(palette)} # (255, 255, 255) : 0,

class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    """
    def __init__(self, root_dir, flist_name,
                 batch_size = 2, 
                 rgb_mean = (117, 117, 117),
                 cut_off_size = None,
                 data_name = "data",
                 label_name = "softmax_label"):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self._batch_size = batch_size
        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name

        self.num_data = len(open(self.flist_name, 'r').readlines())
        fp = open(self.flist_name, 'r')
        self.cached_table = [tuple(line.strip('\n').split('\t')) for line in fp.readlines()]
        fp.close()

        self._current = 0
        self._index = np.arange(self.num_data)
        np.random.seed(55)
        np.random.shuffle(self._index)

        # random.shuffle(self.cached_table)
        # self.f = None
        # self.cursor = -1
        tic = time.time()
        self.data, self.label = self._read()
        # print 'elapsed: %.0fms' % ((time.time()-tic)*1000.,)
        # print self.data[self.data_name].shape,self.label[self.label_name].shape
        # self._display()

    def _display(self):
        """Display currently loaded image data and label data."""
        import cv2
        lut_r = np.array(palette).astype(np.uint8).reshape((256,3))[:,0]
        lut_g = np.array(palette).astype(np.uint8).reshape((256,3))[:,1]
        lut_b = np.array(palette).astype(np.uint8).reshape((256,3))[:,2]
        for i in range(self.data[self.data_name].shape[0]):
            # data  = {self.data_name:self.data[0][1], self.label_name:self.label[0][1]}
            # sys.stderr.write(str(i)+"\n")
            data  = {self.data_name:self.data[self.data_name][i,:,:,:],
                     self.label_name:np.expand_dims(self.label[self.label_name][i,:,:],axis=0)}
            label_img = data[self.label_name].astype(np.uint8)
            label_img = np.swapaxes(label_img, 1, 2)
            label_img = np.swapaxes(label_img, 0, 2).astype(np.uint8)
            label_img_r = cv2.LUT(label_img,lut_r)
            label_img_g = cv2.LUT(label_img,lut_g)
            label_img_b = cv2.LUT(label_img,lut_b)
            label_img = cv2.merge((label_img_r,label_img_g,label_img_b))
            img = np.squeeze(data[self.data_name])
            img = (img + np.array([123.68, 116.779, 103.939]).reshape((3,1,1))).astype(np.uint8)
            img = np.swapaxes(img, 1, 2)
            img = np.swapaxes(img, 0, 2).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            displayimg = np.hstack((img,label_img))
            cv2.imshow('out_img',displayimg); [exit(0) if (cv2.waitKey(0)&0xff)==27 else None]

    def _fetch_next(self):
        pass
            
    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        # load first frame and get data shape
        _, data_img_name, label_img_name = self.cached_table[self._index[self._current]]
        data_img, label_img = self._read_img(data_img_name, label_img_name)
        self._current += 1

        # initialize data and label shape, and put loaded data
        data_shape = data_img.shape
        label_shape = label_img.shape
        data = {self.data_name:np.zeros(shape=(self._batch_size,data_shape[1],data_shape[2],data_shape[3])),}
        label = {self.label_name:np.zeros(shape=(self._batch_size,label_shape[1],label_shape[2]))}
        # sys.stderr.write(str((data[self.data_name].shape,data_img.shape))+"\n")
        # sys.stderr.write(str((label[self.label_name].shape,label_img.shape))+"\n")
        data[self.data_name][0,:,:,:]= np.expand_dims(data_img, axis=0)
        label[self.label_name][0,:,:]= label_img # np.expand_dims(label_img, axis=0)

        # generate batch data
        for i in range(1,self._batch_size):
            # sys.stderr.write(str(label_img.shape)+"\n")
            _, data_img_name, label_img_name = self.cached_table[self._index[self._current]]
            data_img, label_img = self._read_img(data_img_name, label_img_name)
            data[self.data_name][i,:,:,:]= np.expand_dims(data_img, axis=0)
            label[self.label_name][i,:,:]= label_img # np.expand_dims(label_img, axis=0)
            self._current += 1

        # return list(data.items()), list(label.items())
        return data,label

    def _read_img(self, img_name, label_name):
        # img = Image.open(os.path.join(self.root_dir, img_name))
        img = cv2.imread(os.path.join(self.root_dir, img_name),1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # label = cv2.imread(os.path.join(self.root_dir, label_name),1) # color
        label = cv2.imread(os.path.join(self.root_dir, label_name),0) # grayscale

        # h, w, ch = label.shape
        # label = map(lambda x:color2index[tuple(x)],label.reshape((h*w,ch)).tolist())
        # label = np.array(label).reshape((h,w))
        label = label.astype(np.float32)
        # print label.shape
        
        # assert img.size == label.size
        img = np.array(img, dtype=np.float32)  # (h, w, c)
        # label = np.array(label)  # (h, w)
                    
        reshaped_mean = self.mean.reshape(1, 1, 3)
        img = img - reshaped_mean

        # print((np.mean(img),np.std(img)))
        
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)
        img = np.expand_dims(img, axis=0)  # (1, c, h, w)
        label = np.array(label)  # (h, w)
        label = np.expand_dims(label, axis=0)  # (1, h, w)
        return (img, label)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        # return [(k, tuple([self._batch_size] + list(v.shape[1:]))) for k, v in self.data]
        return [(k, v.shape) for k, v in zip(self.data.keys(),self.data.values())]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        # return [(k, tuple([self._batch_size] + list(v.shape[1:]))) for k, v in self.label]
        return [(k, v.shape) for k, v in zip(self.label.keys(),self.label.values())]

    def get_batch_size(self):
        return self._batch_size

    def reset(self):
        self._current = 0
        np.random.shuffle(self._index)
        # self.cursor = -1
        # random.shuffle(self.cached_table)
        # self.f.close()
        # self.f = open(self.flist_name, 'r')

    def iter_next(self):
        return self._current+self._batch_size < self.num_data

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            # uncomment the following lines to visualize `data` and `label`
            # cv2.imshow('softmax_label',np.squeeze(self.label[0][1],axis=(0,)).astype(np.uint8))
            # cv2.imshow('data',np.squeeze(self.data[0][1],axis=(0,)).astype(np.uint8))
            # if (cv2.waitKey()&0xff)==27: exit(0)
            return {self.data_name  :  self.data[self.data_name],
                    self.label_name :  self.label[self.label_name]}
        else:
            raise StopIteration


if __name__ == "__main__":
    import doctest
    doctest.testmod()


