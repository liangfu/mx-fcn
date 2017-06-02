#!/usr/bin/env python

import xml.etree.ElementTree as et
import os,sys
from pprint import pprint
from utils import getpallete
import cv2
import numpy as np
import Image

w, h = 1280, 960
pallete = getpallete(256)
classes = ['__background__','sky','building','store','tree',
           'road','sidewalk','car','bicycle','pedestrian']
priority=range(len(classes))
class2id=dict(zip(classes,priority))
id2class=dict(zip(priority,classes))

def lm2mask(xmlname,segfile):
    mask = np.zeros((h, w)).astype(np.uint8)
    tree = et.parse(xmlname)
    root = tree.getroot()
    elems = root.findall("object")
    objects = {}
    for elem in elems:
        name = filter(lambda e:e.tag=='name',elem)[0].text.split()[0]
        deleted = int(filter(lambda e:e.tag=='deleted',elem)[0].text.split()[0])
        if deleted: continue
        polygon = filter(lambda e:e.tag=='polygon',elem)[0]
        points = polygon.findall("pt")
        pts = map(lambda p:[int(p[0].text.split()[0]),int(p[1].text.split()[0])],points)
        if objects.get(name) is None:
            objects[name]=[]
        objects[name].append(np.array(pts))
    # pprint(objects)
    for classname in classes:
        idval = class2id[classname]
        
        if idval<5:
            idval=0
        else:
            idval=idval-4
            
        if objects.get(classname) is not None:
            # cv2.drawContours(mask, objects[classname], -1, idval,cv2.FILLED)
            for obj in objects[classname]:
                cv2.fillPoly(mask, [obj], idval)
    mask = cv2.resize(mask, (640,480), interpolation=cv2.INTER_NEAREST)
    out_img = Image.fromarray(mask)
    out_img.putpalette(pallete)
    out_img.save(segfile)
    cv2.imwrite(segfile[:-4]+'_index.png',mask)

if __name__=="__main__":
    lm2mask(sys.argv[1],sys.argv[2])


