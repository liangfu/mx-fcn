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

os.environ['MXNET_ENGINE_TYPE']='NaiveEngine' # enable native code debugging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ctx = mx.gpu(0)

def main():
    fcnxs = symbol_fcnxs_resnet.get_fcn32s_symbol(numclass=21, workspace_default=1024)
    pprint(fcnxs.list_arguments())
    fcnxs_model_prefix = "FCN32s_ResNet"
    if args.model == "fcn16s":
        fcnxs = symbol_fcnxs.get_fcn16s_symbol(numclass=21, workspace_default=1024)
        fcnxs_model_prefix = "FCN16s_ResNet"
    elif args.model == "fcn8s":
        fcnxs = symbol_fcnxs.get_fcn8s_symbol(numclass=21, workspace_default=1024)
        fcnxs_model_prefix = "FCN8s_ResNet"
    arg_names = fcnxs.list_arguments()
    _, fcnxs_args, fcnxs_auxs = mx.model.load_checkpoint(args.prefix, args.epoch)
    if not args.retrain:
        if args.init_type == "vgg16":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_vgg16(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
        elif args.init_type == "resnet":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_resnet(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
        elif args.init_type == "fcnxs":
            fcnxs_args, fcnxs_auxs = init_fcnxs.init_from_fcnxs(ctx, fcnxs, fcnxs_args, fcnxs_auxs)
    train_dataiter = FileIter(
        root_dir             = "./VOC2012",
        flist_name           = "train.lst",
        # cut_off_size         = 400,
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    val_dataiter = FileIter(
        root_dir             = "./VOC2012",
        flist_name           = "val.lst",
        rgb_mean             = (123.68, 116.779, 103.939),
        )
    fcnxs_args = {key: val.as_in_context(ctx) for key, val in fcnxs_args.items()}
    fcnxs_auxs = {key: val.as_in_context(ctx) for key, val in fcnxs_auxs.items()}
    pprint(fcnxs_args)
    pprint(fcnxs_auxs)

    model = Solver(
        ctx                 = ctx,
        symbol              = fcnxs,
        begin_epoch         = 0,
        num_epoch           = 50,
        arg_params          = fcnxs_args,
        aux_params          = fcnxs_auxs,
        learning_rate       = 1e-2,
        momentum            = 0.9,
        wd                  = 0.0005)
    model.fit(
        train_data          = train_dataiter,
        eval_data           = val_dataiter,
        batch_end_callback  = mx.callback.Speedometer(1, 10),
        epoch_end_callback  = mx.callback.do_checkpoint(fcnxs_model_prefix))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert vgg16 model to vgg16fc model.')
    parser.add_argument('--model', default='fcnxs',
        help='The type of fcn-xs model, e.g. fcnxs, fcn16s, fcn8s.')
    # parser.add_argument('--prefix', default='VGG_FC_ILSVRC_16_layers',
    #     help='The prefix(include path) of vgg16 model with mxnet format.')
    parser.add_argument('--prefix', default='ResNet_ILSVRC_18_layers',
        help='The prefix(include path) of resnet model with mxnet format.')
    parser.add_argument('--epoch', type=int, default=0,
        help='The epoch number of vgg16 model.')
    # parser.add_argument('--init-type', default="vgg16",
    #     help='the init type of fcn-xs model, e.g. vgg16, fcnxs')
    parser.add_argument('--init-type', default="resnet",
          help='the init type of fcn-xs model, e.g. resnet, vgg16, fcnxs')
    parser.add_argument('--retrain', action='store_true', default=False,
        help='true means continue training.')
    args = parser.parse_args()
    logging.info(args)
    main()
