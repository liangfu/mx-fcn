# pylint: skip-file
import sys, os
import argparse
import mxnet as mx
import numpy as np
import logging
import symbol_fcnxs_resnet
import init_fcnxs
from data import FileIter
from solver import Solver
from pprint import pprint
import time

# os.environ['MXNET_ENGINE_TYPE']='NaiveEngine' # enable native code debugging

logger = logging.getLogger()
fh = logging.FileHandler(os.path.join('log',time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

ctx = mx.gpu(0)
numclass = 30
root_dir = 'Cityscapes'
# numclass = 21
# root_dir = 'VOC2012'
# numclass = 6
# root_dir = 'StreetScenes'

def main():
    fcnxs = symbol_fcnxs_resnet.get_fcn32s_symbol(numclass=numclass, workspace_default=1024)
    # pprint(fcnxs.list_arguments())
    fcnxs_model_prefix = os.path.join("models","FCN32s_ResNet_"+root_dir)
    if args.model == "fcn16s":
        fcnxs = symbol_fcnxs_resnet.get_fcn16s_symbol(numclass=numclass, workspace_default=1024)
        fcnxs_model_prefix = os.path.join("models","FCN16s_ResNet_"+root_dir)
    elif args.model == "fcn8s":
        fcnxs = symbol_fcnxs_resnet.get_fcn8s_symbol(numclass=numclass, workspace_default=1024)
        fcnxs_model_prefix = os.path.join("models","FCN8s_ResNet_"+root_dir)
    arg_names = fcnxs.list_arguments()
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(args.prefix, args.epoch)
    # mx.model.save_checkpoint(args.prefix,args.epoch,_,fcnxs_args,fcnxs_auxs)
    # exit(0) # update symbol and parameter file version ...
    if not args.retrain:
        if args.init_type == "vgg16":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
        elif args.init_type == "resnet":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_resnet(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
        elif args.init_type == "fcnxs":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_fcnxs(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    train_dataiter = FileIter(
        root_dir             = root_dir,
        flist_name           = "train.lst",
        # cut_off_size         = 400,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    val_dataiter = FileIter(
        root_dir             = root_dir,
        flist_name           = "val.lst",
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    fcnxs_args = {key: val.as_in_context(ctx) for key, val in fcnxs_args.items()}
    fcnxs_auxs = {key: val.as_in_context(ctx) for key, val in fcnxs_auxs.items()}
    # pprint(fcnxs_args)
    # pprint(fcnxs_auxs)

    # network visualization
    # dot = mx.viz.plot_network(fcnxs, shape={'data':(1,3,224,224)})
    # dot.view()

    model = Solver(
        ctx                 = ctx,
        symbol              = fcnxs,
        begin_epoch         = args.epoch if args.retrain else 0,
        num_epoch           = 30, # 50 epoch
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_auxs,
        learning_rate       = args.lr, # 1e-5
        momentum            = 0.9,  # 0.99
        wd                  = 0.0005) # 0.0005
    model.fit(
        train_data          = train_dataiter,
        eval_data           = val_dataiter,
        batch_end_callback  = mx.callback.Speedometer(1, 50),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--model', default='fcn16s', # fcnxs
        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    parser.add_argument('--prefix', default='ResNet_ILSVRC_18_layers',
        help='The prefix(include path) of resnet model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=0,
        help='The epoch number of vgg16 model.')
    parser.add_argument('--lr', type=float, default=1e-5,
        help='The learning rate of current training.')
    parser.add_argument('--init-type', default="resnet",
          help='the init type of fcn-xs model, e.g. resnet, vgg16, fcnxs')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')
    args = parser.parse_args()
    logging.info(args)
    main()
