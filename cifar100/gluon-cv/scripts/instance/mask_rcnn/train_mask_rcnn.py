"""Train Mask RCNN end to end."""
import argparse
import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import MaskRCNNDefaultTrainTransform, \
    MaskRCNNDefaultValTransform
from gluoncv.utils.metrics.coco_instance import COCOInstanceMetric
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric, MaskAccMetric, MaskFGAccMetric
from gluoncv.utils.parallel import Parallelizable, Parallel

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN network end to end.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Now support coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs '
                                        'are powerful.')
    parser.add_argument('--batch-size', type=int, default=8, help='Training mini-batch size.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./mask_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.00125 for coco single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 17,23 for coco.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 8000 for coco.')
    parser.add_argument('--lr-warmup-factor', type=float, default=1. / 3.,
                        help='warmup factor of base lr.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 1e-4 for coco')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None,
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be fixed,'
                             ' and no normalization layer will be used. '
                             'Currently supports \'bn\', and None, default is None')

    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the entire model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--horovod', action='store_true',
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                             '--gpus is ignored when using --horovod.')
    parser.add_argument('--executor-threads', type=int, default=1,
                        help='Number of threads for executor for scheduling ops. '
                             'More threads may incur higher GPU memory footprint, '
                             'but may speed up throughput. Note that when horovod is used, '
                             'it is set to 1.')
    parser.add_argument('--kv-store', type=str, default='nccl',
                        help='KV store options. local, device, nccl, dist_sync, dist_device_sync, '
                             'dist_async are available.')

    args = parser.parse_args()
    if args.horovod:
        if hvd is None:
            raise SystemExit("Horovod not found, please check if you installed it correctly.")
        hvd.init()
    args.epochs = int(args.epochs) if args.epochs else 26
    args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
    args.lr = float(args.lr) if args.lr else 0.01
    args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
    args.wd = float(args.wd) if args.wd else 1e-4
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        train_dataset = gdata.COCOInstance(splits='instances_train2017')
        val_dataset = gdata.COCOInstance(splits='instances_val2017', skip_empty=False)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_shards, args):
    """Get dataloader."""
    train_bfn = batchify.MaskRCNNTrainBatchify(net, num_shards)
    train_sampler = \
        gcv.nn.sampler.SplitSortedBucketSampler(train_dataset.get_im_aspect_ratio(),
                                                batch_size,
                                                num_parts=hvd.size() if args.horovod else 1,
                                                part_index=hvd.rank() if args.horovod else 0,
                                                shuffle=True)
    train_loader = mx.gluon.data.DataLoader(train_dataset.transform(
        train_transform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=args.use_fpn)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=args.num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size)), num_shards, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not args.disable_hybridization:
        net.hybridize(static_alloc=args.static_alloc)
    for ib, batch in enumerate(val_data):
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        det_masks = []
        det_infos = []
        for x, im_info in zip(*batch):
            # get prediction results
            ids, scores, bboxes, masks = net(x)
            det_bboxes.append(clipper(bboxes, x))
            det_ids.append(ids)
            det_scores.append(scores)
            det_masks.append(masks)
            det_infos.append(im_info)
        # update metric
        for det_bbox, det_id, det_score, det_mask, det_info in zip(det_bboxes, det_ids, det_scores,
                                                                   det_masks, det_infos):
            for i in range(det_info.shape[0]):
                # numpy everything
                det_bbox = det_bbox[i].asnumpy()
                det_id = det_id[i].asnumpy()
                det_score = det_score[i].asnumpy()
                det_mask = det_mask[i].asnumpy()
                det_info = det_info[i].asnumpy()
                # filter by conf threshold
                im_height, im_width, im_scale = det_info
                valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                det_id = det_id[valid]
                det_score = det_score[valid]
                det_bbox = det_bbox[valid] / im_scale
                det_mask = det_mask[valid]
                # fill full mask
                im_height, im_width = int(round(im_height / im_scale)), int(
                    round(im_width / im_scale))
                full_masks = []
                for bbox, mask in zip(det_bbox, det_mask):
                    full_masks.append(gdata.transforms.mask.fill(mask, bbox, (im_width, im_height)))
                full_masks = np.array(full_masks)
                eval_metric.update(det_bbox, det_id, det_score, full_masks)
    return eval_metric.get()


def get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    return lr_warmup_factor * (1 - alpha) + alpha


class ForwardBackwardTask(Parallelizable):
    def __init__(self, net, optimizer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss,
                 rcnn_mask_loss):
        super(ForwardBackwardTask, self).__init__()
        self.net = net
        self._optimizer = optimizer
        self.rpn_cls_loss = rpn_cls_loss
        self.rpn_box_loss = rpn_box_loss
        self.rcnn_cls_loss = rcnn_cls_loss
        self.rcnn_box_loss = rcnn_box_loss
        self.rcnn_mask_loss = rcnn_mask_loss

    def forward_backward(self, x):
        data, label, gt_mask, rpn_cls_targets, rpn_box_targets, rpn_box_masks = x
        with autograd.record():
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            cls_pred, box_pred, mask_pred, roi, samples, matches, rpn_score, rpn_box, anchors = net(
                data, gt_box)
            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = self.rpn_cls_loss(rpn_score, rpn_cls_targets,
                                          rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = self.rpn_box_loss(rpn_box, rpn_box_targets,
                                          rpn_box_masks) * rpn_box.size / num_rpn_pos
            # rpn overall loss, use sum rather than average
            rpn_loss = rpn_loss1 + rpn_loss2
            # generate targets for rcnn
            cls_targets, box_targets, box_masks = self.net.target_generator(roi, samples,
                                                                            matches, gt_label,
                                                                            gt_box)
            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = self.rcnn_cls_loss(cls_pred, cls_targets,
                                            cls_targets.expand_dims(-1) >= 0) * cls_targets.size / \
                         num_rcnn_pos
            rcnn_loss2 = self.rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / \
                         num_rcnn_pos
            rcnn_loss = rcnn_loss1 + rcnn_loss2

            # generate targets for mask
            mask_targets, mask_masks = self.net.mask_target(roi, gt_mask, matches, cls_targets)
            # loss of mask
            mask_loss = self.rcnn_mask_loss(mask_pred, mask_targets, mask_masks) * \
                        mask_targets.size / mask_masks.sum()

            # overall losses
            total_loss = rpn_loss.sum() + rcnn_loss.sum() + mask_loss.sum()

            rpn_loss1_metric = rpn_loss1.mean()
            rpn_loss2_metric = rpn_loss2.mean()
            rcnn_loss1_metric = rcnn_loss1.sum()
            rcnn_loss2_metric = rcnn_loss2.sum()
            mask_loss_metric = mask_loss.sum()
            rpn_acc_metric = [[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]]
            rpn_l1_loss_metric = [[rpn_box_targets, rpn_box_masks], [rpn_box]]
            rcnn_acc_metric = [[cls_targets], [cls_pred]]
            rcnn_l1_loss_metric = [[box_targets, box_masks], [box_pred]]
            rcnn_mask_metric = [[mask_targets, mask_masks], [mask_pred]]
            rcnn_fgmask_metric = [[mask_targets, mask_masks], [mask_pred]]

            if args.amp:
                with amp.scale_loss(total_loss, self._optimizer) as scaled_losses:
                    autograd.backward(scaled_losses)
            else:
                total_loss.backward()

        return rpn_loss1_metric, rpn_loss2_metric, rcnn_loss1_metric, rcnn_loss2_metric, \
               mask_loss_metric, rpn_acc_metric, rpn_l1_loss_metric, rcnn_acc_metric, \
               rcnn_l1_loss_metric, rcnn_mask_metric, rcnn_fgmask_metric


def train(net, train_data, val_data, eval_metric, batch_size, ctx, args):
    """Training pipeline"""
    kv = mx.kvstore.create(args.kv_store)
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    for k, v in net.collect_params('.*bias').items():
        v.wd_mult = 0.0
    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            'sgd',
            optimizer_params
        )
    else:
        trainer = gluon.Trainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            'sgd',
            optimizer_params,
            update_on_kvstore=(False if args.amp else None), kvstore=kv)

    if args.amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=1 / 9.)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss()  # == smoothl1
    rcnn_mask_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'),
               mx.metric.Loss('RCNN_Mask')]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    rcnn_mask_metric = MaskAccMetric()
    rcnn_fgmask_metric = MaskFGAccMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric,
                rcnn_acc_metric, rcnn_bbox_metric,
                rcnn_mask_metric, rcnn_fgmask_metric]

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        rcnn_task = ForwardBackwardTask(net, trainer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss,
                                        rcnn_box_loss, rcnn_mask_loss)
        executor = Parallel(args.executor_threads, rcnn_task) if not args.horovod else None
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup, args.lr_warmup_factor)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            # losses = []
            if executor is not None:
                for data in zip(*batch):
                    executor.put(data)
            for j in range(len(ctx)):
                if executor is not None:
                    result = executor.get()
                else:
                    result = rcnn_task.forward_backward(list(zip(*batch))[0])
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])
            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])
            trainer.step(batch_size)
            # update metrics
            if (not args.horovod or hvd.rank() == 0) and args.log_interval \
                    and not (i + 1) % args.log_interval:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * args.batch_size / (time.time() - btic), msg))
                btic = time.time()
        # validate and save params
        if (not args.horovod) or hvd.rank() == 0:
            msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
            logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                epoch, (time.time() - tic), msg))
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, logger, best_map, current_map, epoch, args.save_interval,
                        args.save_prefix)


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    if args.amp:
        amp.init()

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == 'bn':
            kwargs['num_devices'] = len(ctx)
    num_gpus = hvd.size() if args.horovod else len(ctx)
    net_name = '_'.join(('mask_rcnn', *module_list, args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True,
                    per_device_batch_size=args.batch_size // num_gpus, **kwargs)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    batch_size = args.batch_size // num_gpus if args.horovod else args.batch_size
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, MaskRCNNDefaultTrainTransform, MaskRCNNDefaultValTransform,
        batch_size, len(ctx), args)

    # training
    train(net, train_data, val_data, eval_metric, batch_size, ctx, args)
