#!/usr/bin/env bash

range=`seq -f %06.0f 1 200`

echo '' > task.txt

for ii in $range
do
    echo "echo $ii" >> task.txt
    echo "python lm2mask.py Annotations/$ii.xml SegmentationClass/$ii.png" >> task.txt
done

# for ii in $range
# do
#     echo "echo $ii" >> task.txt
#     echo "convert -interpolate nearest -filter point -resize 640x480 SegmentationClass/$ii.png SegmentationClass/$ii.png" >> task.txt
# done

parallel -j2 < task.txt


