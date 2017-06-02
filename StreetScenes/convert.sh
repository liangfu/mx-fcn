#!/usr/bin/env bash

echo '' > task.txt

# range=`seq -f %06.0f 1 200`
# for ii in $range
# do
#     echo "echo $ii" >> task.txt
#     echo "python lm2mask.py Annotations/$ii.xml SegmentationClass/$ii.png" >> task.txt
# done

echo "echo '===================='" >> task.txt

range=`seq -f %06.0f 3001 3020`
for ii in $range
do
    echo "echo $ii" >> task.txt
    echo "python lm2mask.py Annotations/$ii.xml SegmentationClass/$ii.png" >> task.txt
done

parallel -j2 < task.txt


