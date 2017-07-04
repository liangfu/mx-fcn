# dataset=VOC2012
# dataset=StreetScenes
dataset=Cityscapes

## train fcn-32s model
# python -u fcn_xs.py --model=fcn32s --prefix=models/VGG16_ILSVRC_18_layers --epoch=0 --lr 1e-4 --init-type=vgg16
# python -u fcn_xs.py --model=fcn32s --prefix=models/ResNet_ILSVRC_18_layers --epoch=0 --lr 1e-4 --init-type=resnet
# python -u fcn_xs.py --model=fcn32s --prefix=models/FCN32s_ResNet_$dataset --epoch=32 --lr 1e-4 --init-type=resnet --retrain

## train fcn-16s model
# python -u fcn_xs.py --model=fcn16s --prefix=models/FCN32s_VGG16 --epoch=31 --lr 1e-5 --init-type=fcnxs
python -u fcn_xs.py --model=fcn16s --prefix=models/FCN32s_ResNet_$dataset --epoch=7 --lr 1e-5 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn16s --prefix=models/FCN16s_ResNet_$dataset --epoch=2 --lr 1e-5 --init-type=fcnxs --retrain

## train fcn-8s model
# python -u fcn_xs.py --model=fcn8s --prefix=models/FCN16s_VGG16 --epoch=27 --lr 1e-6 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn8s --prefix=models/FCN16s_ResNet_$dataset --epoch=2 --lr 1e-6 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn8s --prefix=models/FCN8s_ResNet_$dataset --epoch=10 --lr 1e-6 --init-type=fcnxs --retrain

