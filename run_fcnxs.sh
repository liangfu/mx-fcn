## train fcn-32s model
# python -u fcn_xs.py --model=fcn32s --prefix=models/VGG16_ILSVRC_18_layers --epoch=0 --init-type=vgg16
# python -u fcn_xs.py --model=fcn32s --prefix=models/ResNet_ILSVRC_18_layers --epoch=0 --init-type=resnet
python -u fcn_xs.py --model=fcn32s --prefix=models/FCN32s_ResNet --epoch=26 --init-type=resnet --retrain

## train fcn-16s model
# python -u fcn_xs.py --model=fcn16s --prefix=models/FCN32s_VGG16 --epoch=31 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn16s --prefix=models/FCN32s_ResNet --epoch=42 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn16s --prefix=models/FCN16s_ResNet --epoch=42 --init-type=fcnxs --retrain

## train fcn-8s model
# python -u fcn_xs.py --model=fcn8s --prefix=models/FCN16s_VGG16 --epoch=27 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn8s --prefix=models/FCN16s_ResNet --epoch=19 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn8s --prefix=models/FCN8s_ResNet --epoch=12 --init-type=fcnxs --retrain

