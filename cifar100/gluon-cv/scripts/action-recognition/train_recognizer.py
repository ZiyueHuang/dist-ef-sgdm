import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxboard import SummaryWriter
from mxnet.contrib import amp

from gluoncv.data.transforms import video
from gluoncv.data import ucf101, kinetics400
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load
from gluoncv.data.dataloader import tsn_mp_batchify_fn
from gluoncv.data.sampler import SplitSampler

# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for action recognition.')
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
    parser.add_argument('--batch-size', type=int, default=25,
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
    parser.add_argument('--new-height', type=int, default=0,
                        help='new height of the resize image. default is 0')
    parser.add_argument('--new-width', type=int, default=0,
                        help='new width of the resize image. default is 0')
    parser.add_argument('--new-length', type=int, default=1,
                        help='new length of video sequence. default is 1')
    parser.add_argument('--new-step', type=int, default=1,
                        help='new step to skip video sequence. default is 1')
    parser.add_argument('--clip-grad', type=int, default=0,
                        help='clip gradient to a certain threshold. Set the value to be larger than zero to enable gradient clipping.')
    parser.add_argument('--partial-bn', action='store_true',
                        help='whether to freeze bn layers except the first layer.')
    parser.add_argument('--num-classes', type=int, default=101,
                        help='number of classes.')
    parser.add_argument('--scale-ratios', type=str, default='1.0, 0.875, 0.75, 0.66',
                        help='Scale ratios used in multi-scale cropping data augmentation technique.')
    parser.add_argument('--use-amp', action='store_true',
                        help='whether to use automatic mixed precision.')
    parser.add_argument('--prefetch-ratio', type=float, default=2.0,
                        help='set number of workers to prefetch data batch, default is 2 in MXNet.')
    parser.add_argument('--kvstore', type=str, default=None,
                        help='KVStore type. Supports local, device, dist_sync_device, dist_async_device')
    opt = parser.parse_args()
    return opt

def get_data_loader(opt, batch_size, num_workers, logger, kvstore=None):
    data_dir = opt.data_dir
    val_data_dir = opt.val_data_dir
    scale_ratios = [float(i) for i in opt.scale_ratios.split(',')]
    input_size = opt.input_size

    def batch_fn(batch, ctx):
        if opt.num_segments > 1:
            data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False, multiplier=opt.num_segments)
        else:
            data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        return data, label

    transform_train = transforms.Compose([
        video.VideoMultiScaleCrop(size=(input_size, input_size), scale_ratios=scale_ratios),
        video.VideoRandomHorizontalFlip(),
        video.VideoToTensor(),
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        video.VideoCenterCrop(size=input_size),
        video.VideoToTensor(),
        video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if opt.dataset == 'kinetics400':
        train_dataset = kinetics400.classification.Kinetics400(setting=opt.train_list, root=data_dir, train=True,
                                                     new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                                     target_width=input_size, target_height=input_size,
                                                     num_segments=opt.num_segments, transform=transform_train)
        val_dataset = kinetics400.classification.Kinetics400(setting=opt.val_list, root=val_data_dir, train=False,
                                                   new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length, new_step=opt.new_step,
                                                   target_width=input_size, target_height=input_size,
                                                   num_segments=opt.num_segments, transform=transform_test)
    elif opt.dataset == 'ucf101':
        train_dataset = ucf101.classification.UCF101(setting=opt.train_list, root=data_dir, train=True,
                                                     new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                                                     target_width=input_size, target_height=input_size,
                                                     num_segments=opt.num_segments, transform=transform_train)
        val_dataset = ucf101.classification.UCF101(setting=opt.val_list, root=data_dir, train=False,
                                                   new_width=opt.new_width, new_height=opt.new_height, new_length=opt.new_length,
                                                   target_width=input_size, target_height=input_size,
                                                   num_segments=opt.num_segments, transform=transform_test)
    else:
        logger.info('Dataset %s is not supported yet.' % (opt.dataset))

    logger.info('Load %d training samples and %d validation samples.' % (len(train_dataset), len(val_dataset)))

    if opt.num_segments > 1:
        train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, batchify_fn=tsn_mp_batchify_fn, last_batch='rollover')
        val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, batchify_fn=tsn_mp_batchify_fn, last_batch='keep')
    else:
        if kvstore is not None:
            train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, prefetch=int(opt.prefetch_ratio * num_workers),
                                           sampler=SplitSampler(len(train_dataset), num_parts=kvstore.num_workers, part_index=kvstore.rank), last_batch='rollover')
            val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, prefetch=int(opt.prefetch_ratio * num_workers),
                                           sampler=SplitSampler(len(val_dataset), num_parts=kvstore.num_workers, part_index=kvstore.rank), last_batch='keep')
        else:
            train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, prefetch=int(opt.prefetch_ratio * num_workers), last_batch='rollover')
            val_data = gluon.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, prefetch=int(opt.prefetch_ratio * num_workers), last_batch='keep')

    return train_data, val_data, batch_fn

def main():
    opt = parse_args()

    makedirs(opt.save_dir)

    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    sw = SummaryWriter(logdir=opt.save_dir, flush_secs=5)

    if opt.kvstore is not None:
        kv = mx.kvstore.create(opt.kvstore)
        logger.info('Distributed training with %d workers and current rank is %d' % (kv.num_workers, kv.rank))
    if opt.use_amp:
        amp.init()

    batch_size = opt.batch_size
    classes = opt.num_classes

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    logger.info('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]

    optimizer = 'sgd'
    if opt.clip_grad > 0:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum, 'clip_gradient': opt.clip_grad}
    else:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

    if opt.dtype != 'float32':
        optimizer_params['multi_precision'] = True

    model_name = opt.model
    net = get_model(name=model_name, nclass=classes, pretrained=opt.use_pretrained,
                    tsn=opt.use_tsn, num_segments=opt.num_segments, partial_bn=opt.partial_bn)
    net.cast(opt.dtype)
    net.collect_params().reset_ctx(context)
    logger.info(net)

    if opt.resume_params is not '':
        net.load_parameters(opt.resume_params, ctx=context)

    if opt.kvstore is not None:
        train_data, val_data, batch_fn = get_data_loader(opt, batch_size, num_workers, logger, kv)
    else:
        train_data, val_data, batch_fn = get_data_loader(opt, batch_size, num_workers, logger)

    train_metric = mx.metric.Accuracy()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    def test(ctx, val_data, kvstore=None):
        acc_top1.reset()
        acc_top5.reset()
        num_test_iter = len(val_data)
        for i, batch in enumerate(val_data):
            data, label = batch_fn(batch, ctx)
            outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

            if opt.log_interval and not (i+1) % opt.log_interval:
                logger.info('Batch [%04d]/[%04d]: evaluated' % (i, num_test_iter))

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()

        if kvstore is not None:
            top1_nd = nd.zeros(1)
            top5_nd = nd.zeros(1)
            kvstore.push(111111, nd.array(np.array([top1])))
            kvstore.pull(111111, out=top1_nd)
            kvstore.push(555555, nd.array(np.array([top5])))
            kvstore.pull(555555, out=top5_nd)
            top1 = top1_nd.asnumpy() / kvstore.num_workers
            top5 = top5_nd.asnumpy() / kvstore.num_workers

        return (top1, top5)

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if opt.partial_bn:
            train_patterns = None
            if 'inceptionv3' in opt.model:
                train_patterns = '.*weight|.*bias|inception30_batchnorm0_gamma|inception30_batchnorm0_beta|inception30_batchnorm0_running_mean|inception30_batchnorm0_running_var'
            else:
                logger.info('Current model does not support partial batch normalization.')

            if opt.kvstore is not None:
                trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params, kvstore=kv, update_on_kvstore=False)
            else:
                trainer = gluon.Trainer(net.collect_params(train_patterns), optimizer, optimizer_params, update_on_kvstore=False)
        else:
            if opt.kvstore is not None:
                trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, kvstore=kv, update_on_kvstore=False)
            else:
                trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params, update_on_kvstore=False)

        if opt.resume_states is not '':
            trainer.load_states(opt.resume_states)

        if opt.use_amp:
            amp.init_trainer(trainer)

        L = gluon.loss.SoftmaxCrossEntropyLoss()

        best_val_score = 0
        lr_decay_count = 0

        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_metric.reset()
            btic = time.time()
            num_train_iter = len(train_data)

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate * lr_decay)
                lr_decay_count += 1

            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                with ag.record():
                    outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                    loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]

                    if opt.use_amp:
                        with amp.scale_loss(loss, trainer) as scaled_loss:
                            ag.backward(scaled_loss)
                    else:
                        ag.backward(loss)

                if opt.kvstore is not None:
                    trainer.step(batch_size * kv.num_workers)
                else:
                    trainer.step(batch_size)

                train_metric.update(label, outputs)

                if opt.log_interval and not (i+1) % opt.log_interval:
                    train_metric_name, train_metric_score = train_metric.get()
                    logger.info('Epoch[%03d] Batch [%04d]/[%04d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                                epoch, i, num_train_iter, batch_size*opt.log_interval/(time.time()-btic),
                                train_metric_name, train_metric_score*100, trainer.learning_rate))
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i /(time.time() - tic))
            mx.ndarray.waitall()

            if opt.kvstore is not None and epoch == opt.resume_epoch:
                kv.init(111111, nd.zeros(1))
                kv.init(555555, nd.zeros(1))

            if opt.kvstore is not None:
                acc_top1_val, acc_top5_val = test(ctx, val_data, kv)
            else:
                acc_top1_val, acc_top5_val = test(ctx, val_data)

            logger.info('[Epoch %03d] training: %s=%f' % (epoch, train_metric_name, train_metric_score*100))
            logger.info('[Epoch %03d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time()-tic))
            logger.info('[Epoch %03d] validation: acc-top1=%f acc-top5=%f' % (epoch, acc_top1_val*100, acc_top5_val*100))

            sw.add_scalar(tag='train_acc', value=train_metric_score*100, global_step=epoch)
            sw.add_scalar(tag='valid_acc', value=acc_top1_val*100, global_step=epoch)

            if acc_top1_val > best_val_score:
                best_val_score = acc_top1_val
                if opt.use_tsn:
                    net.basenet.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))
                else:
                    net.save_parameters('%s/%.4f-%s-%s-%03d-best.params'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))
                trainer.save_states('%s/%.4f-%s-%s-%03d-best.states'%(opt.save_dir, best_val_score, opt.dataset, model_name, epoch))

            if opt.save_frequency and opt.save_dir and (epoch + 1) % opt.save_frequency == 0:
                if opt.use_tsn:
                    net.basenet.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, epoch))
                else:
                    net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, epoch))
                trainer.save_states('%s/%-%s-%03d.states'%(opt.save_dir, opt.dataset, model_name, epoch))

        # save the last model
        if opt.use_tsn:
            net.basenet.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))
        else:
            net.save_parameters('%s/%s-%s-%03d.params'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))
        trainer.save_states('%s/%s-%s-%03d.states'%(opt.save_dir, opt.dataset, model_name, opt.num_epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    train(context)
    sw.close()


if __name__ == '__main__':
    main()
