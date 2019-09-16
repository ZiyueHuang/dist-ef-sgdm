import argparse, time, logging

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.data import transforms as gcv_transforms

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=8,
                    help='number of workers on gpu_id')
parser.add_argument('--gpu-id', type=int, default=0,
                    help='GPU ID')
parser.add_argument('--optimizer', type=str, default='DistEFSGD',
                    help='optimizer to use. default is DistEFSGD')
parser.add_argument('--kv', type=str, default='local',
                    help='kvstore to use. default is local')
parser.add_argument('--model', type=str, default='cifar_resnet20_v2',
                    help='model to use. options are resnet and wrn. default is cifar_resnet20_v2.')
parser.add_argument('--num-epochs', type=int, default=200,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0005,
                    help='weight decay rate. default is 0.0005.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                    help='epoches at which learning rate decays. default is 100,150.')
parser.add_argument('--log', type=str, default='train_cifar', help='log name')
opt = parser.parse_args()


batch_size = opt.batch_size
classes = 100

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

lr_decay = opt.lr_decay
lr_decay_updates = [int(int(i)*50000.0/batch_size) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

model_name = opt.model

if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate}
else:
    kwargs = {'classes': classes}

net = get_model(model_name, **kwargs)

optimizer = opt.optimizer


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)s %(levelname)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='./log/%s.log' % opt.log,
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logging.info(opt)

# from https://github.com/weiaicunzai/pytorch-cifar/blob/master/conf/global_settings.py
CIFAR100_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]



transform_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])


class CustomIter(mx.io.DataIter):
    def __init__(self, data_loader, data_names, data_shapes,
                 label_names, label_shapes):
        self._provide_data = list(zip(data_names, data_shapes))
        self._provide_label = list(zip(label_names, label_shapes))
        self.data_loader = data_loader
        self.cur_batch = 0

    def __iter__(self):
        for batch in self.data_loader:
            self.cur_batch += 1
            yield mx.io.DataBatch([batch[0]], [batch[1].astype(np.float32)])

    def reset(self):
        self.cur_batch = 0

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label


def main():
    data = mx.sym.var('data')
    label = mx.sym.var('softmax_label')
    out = net(data)
    softmax = mx.sym.SoftmaxOutput(out, label, name='softmax')
    mod = mx.mod.Module(softmax, context=context)
    print('batch_size ', batch_size)
    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR100(train=True, fine_label=True).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR100(train=False, fine_label=True).transform_first(transform_test),
            batch_size=batch_size, shuffle=False)

    train_data_iter = CustomIter(train_data, ['data'], [(batch_size, 3, 32, 32)],
        ['softmax_label'], [(batch_size,)])

    val_data_iter = CustomIter(val_data, ['data'], [(batch_size, 3, 32, 32)],
        ['softmax_label'], [(batch_size,)])

    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=lr_decay_updates, factor=opt.lr_decay)

    if optimizer == 'SignSGD':
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd,
                            'lr_scheduler': lr_scheduler}
    else:
        optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd,
                            'lr_scheduler': lr_scheduler,
                            'momentum': opt.momentum}

    mod.fit(train_data_iter,
            eval_data=val_data_iter,
            eval_metric=mx.metric.create(['acc', 'ce']),
            num_epoch=opt.num_epochs,
            kvstore=opt.kv,
            batch_end_callback = mx.callback.Speedometer(batch_size, 150),
            optimizer = optimizer,
            optimizer_params = optimizer_params,
            initializer = mx.init.Xavier())

if __name__ == '__main__':
    main()

