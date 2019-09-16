import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, gpu, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter

from gluoncv.data.transforms import video
from gluoncv.data import ucf101, kinetics400
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained model for action recognition.')
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'kinetics400'],
                        help='which dataset to use.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/ucf101/rawframes',
                        help='training (and validation) pictures to use.')
    parser.add_argument('--val-data-dir', type=str, default='~/.mxnet/datasets/ucf101/rawframes',
                        help='validation pictures to use.')
    parser.add_argument('--train-list', type=str, default='~/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_rgb_split1.txt',
                        help='the list of training data')
    parser.add_argument('--val-list', type=str, default='~/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_val_rgb_split1.txt',
                        help='the list of validation data')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model for distillation training')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train.log',
                        help='name of training log file')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--eval', action='store_true',
                        help='directly evaluate the model.')
    parser.add_argument('--num-segments', type=int, default=1,
                        help='number of segments to evenly split the video.')
    parser.add_argument('--use-tsn', action='store_true',
                        help='whether to use temporal segment networks.')
    parser.add_argument('--new-height', type=int, default=256,
                        help='new height of the resize image. default is 256')
    parser.add_argument('--new-width', type=int, default=340,
                        help='new width of the resize image. default is 340')
    parser.add_argument('--new-length', type=int, default=1,
                        help='new length of video sequence. default is 1')
    parser.add_argument('--new-step', type=int, default=1,
                        help='new step to skip video sequence. default is 1')
    parser.add_argument('--num-classes', type=int, default=101,
                        help='number of classes.')
    parser.add_argument('--ten-crop', action='store_true',
                        help='whether to use ten crop evaluation.')
    parser.add_argument('--use-amp', action='store_true',
                        help='whether to use automatic mixed precision.')
    parser.add_argument('--prefetch-ratio', type=float, default=2.0,
                        help='set number of workers to prefetch data batch, default is 2 in MXNet.')
    parser.add_argument('--input-5d', action='store_true',
                        help='the input is 4d or 5d tensor. 5d is for 3D CNN models.')
    opt = parser.parse_args()
    return opt

def batch_fn(batch, ctx):
    data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

def main():
    opt = parse_args()
    print(opt)

    # set env
    num_gpus = opt.num_gpus
    batch_size = opt.batch_size
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers
    print('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))

    # get model
    classes = opt.num_classes
    model_name = opt.model
    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained)
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.resume_params is not '' and not opt.use_pretrained:
        net.load_parameters(opt.resume_params, ctx=context)
        print('Pre-trained model %s is successfully loaded.' % (opt.resume_params))
    else:
        print('Pre-trained model is successfully loaded from the model zoo.')

    # get data
    if opt.ten_crop:
        transform_test = transforms.Compose([
            video.VideoTenCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform_test = transforms.Compose([
            video.VideoCenterCrop(opt.input_size),
            video.VideoToTensor(),
            video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if opt.dataset == 'ucf101':
        val_dataset = ucf101.classification.UCF101(setting=opt.val_list, root=opt.data_dir, train=False,
                                               new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                                               target_width=opt.input_size, target_height=opt.input_size,
                                               test_mode=True, num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'kinetics400':
        val_dataset = kinetics400.classification.Kinetics400(setting=opt.val_list, root=opt.data_dir, train=False,
                                               new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                               target_width=opt.input_size, target_height=opt.input_size,
                                               test_mode=True, num_segments=opt.num_segments, transform=transform_test)
    else:
        logger.info('Dataset %s is not supported yet.' % (opt.dataset))

    val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch=int(opt.prefetch_ratio * num_workers))
    print('Load %d test samples.' % len(val_dataset))

    # start evaluation
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    """Common practice during evaluation is to evenly sample 25 frames from a single video, and then perform 10-crop data augmentation.
    This leads to 250 samples per video (750 channels). If this is too large to fit into one GPU, we can split it into multiple data batches.
    `num_data_batches` has to be set to a value as long as `num_split_frames` is multiples of 3.
    For example, when `num_data_batches` is set to 10,  `num_split_frames` will be 750/10=75, which is multiples of 3.
    If you have enough GPU memory and prefer faster evaluation speed, you can set `num_data_batches` to 1.
    """
    num_data_batches = 10
    if opt.ten_crop:
        num_frames = opt.num_segments * 10
    else:
        num_frames = opt.num_segments
    num_split_frames = int(num_frames * 3 / num_data_batches)

    def test(ctx, val_data):
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in enumerate(val_data):
            outputs = []
            for seg_id in range(num_data_batches):
                bs = seg_id * num_split_frames
                be = (seg_id + 1) * num_split_frames
                if opt.input_5d:
                    new_batch = [batch[0][:,bs:be,:,:,:], batch[1]]
                else:
                    new_batch = [batch[0][:,bs:be,:,:], batch[1]]
                data, label = batch_fn(new_batch, ctx)
                for gpu_id, X in enumerate(data):
                    if opt.input_5d:
                        new_X = X.reshape((-1, 3, opt.new_length, opt.input_size, opt.input_size))
                    else:
                        new_X = X.reshape((-1, 3, opt.input_size, opt.input_size))
                    pred = net(new_X)
                    if seg_id == 0:
                        outputs.append(pred)
                    else:
                        outputs[gpu_id] = nd.concat(outputs[gpu_id], pred, dim=0)
            # Perform the mean operation on 'num_frames' samples of each video
            for gpu_id, out in enumerate(outputs):
                outputs[gpu_id] = nd.expand_dims(out.mean(axis=0), axis=0)

            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)
            mx.ndarray.waitall()

            _, cur_top1 = acc_top1.get()
            _, cur_top5 = acc_top5.get()

            if i > 0 and i % opt.log_interval == 0:
                print('%04d/%04d is done: acc-top1=%f acc-top5=%f' % (i, len(val_data), cur_top1*100, cur_top5*100))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return (top1, top5)

    start_time = time.time()
    acc_top1_val, acc_top5_val = test(context, val_data)
    end_time = time.time()

    print('Test accuracy: acc-top1=%f acc-top5=%f' % (acc_top1_val*100, acc_top5_val*100))
    print('Total evaluation time is %4.2f minutes' % ((end_time - start_time) / 60))

if __name__ == '__main__':
    main()
