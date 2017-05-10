## train fcn-32s model
# python -u fcn_xs.py --model=fcn32s --prefix=VGG16_ILSVRC_18_layers --epoch=0 --init-type=vgg16
python -u fcn_xs.py --model=fcn32s --prefix=ResNet_ILSVRC_18_layers --epoch=0 --init-type=resnet

## train fcn-16s model
# python -u fcn_xs.py --model=fcn16s --prefix=FCN32s_VGG16 --epoch=31 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn16s --prefix=FCN32s_ResNet --epoch=42 --init-type=fcnxs

## train fcn-8s model
# python -u fcn_xs.py --model=fcn8s --prefix=FCN16s_VGG16 --epoch=27 --init-type=fcnxs
# python -u fcn_xs.py --model=fcn8s --prefix=FCN16s_ResNet --epoch=27 --init-type=fcnxs

