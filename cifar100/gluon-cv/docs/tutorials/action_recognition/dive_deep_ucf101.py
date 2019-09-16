"""2. Dive Deep into Training TSN mdoels on UCF101
==================================================

This is a video action recognition tutorial using Gluon CV toolkit, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. note::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train_recognizer.py<../../../scripts/action-recognition/train_recognizer.py>`

    Example training command::

        # Finetune a pretrained VGG16 model without using temporal segment network.
        python train_recognizer.py --model vgg16_ucf101 --num-classes 101 --num-gpus 8 --lr-mode step --lr 0.001 --lr-decay 0.1 --lr-decay-epoch 30,60,80 --num-epochs 80

        # Finetune a pretrained VGG16 model using temporal segment network.
        python train_recognizer.py --model vgg16_ucf101 --num-classes 101 --num-gpus 8 --num-segments 3 --use-tsn --lr-mode step --lr 0.001 --lr-decay 0.1 --lr-decay-epoch 30,60,80 --num-epochs 80

    For more training command options, please run ``python train_recognizer.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#action_recognition>`_ for training commands of reproducing the pretrained model.


Network Structure
-----------------

First, let's import the necessary libraries into python.

"""
from __future__ import division

import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
from gluoncv.data import ucf101
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory
from gluoncv.data.dataloader import tsn_mp_batchify_fn


################################################################
#
# Video action recognition is a classification problem.
# Here we pick a simple yet well-performing structure, ``vgg16_ucf101``, for the
# tutorial. In addition, we use the the idea of temporal segments (TSN) [Wang16]_
# to wrap the backbone VGG16 network for adaptation to video domain.

# number of GPUs to use
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]

# Get the model vgg16_ucf101 with temporal segment network, with 101 output classes, without pre-trained weights
net = get_model(name='vgg16_ucf101', nclass=101, tsn=True, num_segments=3)
net.collect_params().reset_ctx(ctx)
print(net)

################################################################
# Data Augmentation and Data Loader
# ---------------------------------
#
# Data augmentation for video is different from image. For example, if you
# want to randomly crop a video sequence, you need to make sure all the video
# frames in this sequence undergo the same cropping process. We provide a
# new set of transformation functions, working with multiple images.
# Please checkout the `video.py <../../../gluoncv/data/transforms/video.py>`_ for more details.
# Most video data augmentation strategies used here are introduced in [Wang15]_.

transform_train = transforms.Compose([
    # Fix the input video frames size as 256×340 and randomly sample the cropping width and height from
    # {256,224,192,168}. After that, resize the cropped regions to 224 × 224.
    video.VideoMultiScaleCrop(size=(224, 224), scale_ratios=[1.0, 0.875, 0.75, 0.66]),
    # Randomly flip the video frames horizontally
    video.VideoRandomHorizontalFlip(),
    # Transpose the video frames from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    video.VideoToTensor(),
    # Normalize the video frames with mean and standard deviation calculated across all images
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

##################################################################
# With the transform functions, we can define data loaders for our
# training datasets.

# Batch Size for Each GPU
per_device_batch_size = 25
# Number of data loader workers
num_workers = 8
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# Set train=True for training data. Here we only use a subset of UCF101 for demonstration purpose.
# The subset has 101 training samples, one sample per class.
train_dataset = ucf101.classification.UCF101(train=True, num_segments=3, transform=transform_train)
print('Load %d training samples.' % len(train_dataset))

##################################################################
# When temporal segment option is turned on, the training samples from ``train_dataset``
# have a dimention of ``num_segments x 3 x 224 x 224``.
# Hence, we can't use default batchify function because we already have a batch dimention.
# We modify the default batchify function to ``tsn_mp_batchify_fn``, basically changing ``nd.stack`` to ``nd.concat``.

train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, batchify_fn=tsn_mp_batchify_fn)

################################################################
# Optimizer, Loss and Metric
# --------------------------

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [30, 60, np.inf]

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': 0.001, 'wd': 0.0001, 'momentum': 0.9}

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

################################################################
# In order to optimize our model, we need a loss function.
# For classification tasks, we usually use softmax cross entropy as the
# loss function.

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

################################################################
# For simplicity, we use accuracy as the metric to monitor our training
# process. Besides, we record metric values, and will print them at the
# end of training.

train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-acc'])

################################################################
# Training
# --------
#
# After all the preparations, we can finally start training!
# Following is the script.
#
# .. note::
#   In order to finish the tutorial quickly, we only train for 3 epochs on the tiny subset.
#   In your experiments, we recommend setting ``epochs=80`` for the full UCF101 dataset.

epochs = 3
lr_decay_count = 0

for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Learning rate decay
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0, multiplier=3)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        # AutoGrad
        with ag.record():
            output = [net(X) for X in data]
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        # Update metrics
        train_loss += sum([l.sum().asscalar() for l in loss])
        train_metric.update(label, output)

    name, acc = train_metric.get()

    # Update history and print metrics
    train_history.update([acc])
    print('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss, time.time()-tic))

# We can plot the metric scores with:
train_history.plot()

##############################################################################
# You can `Start Training Now`_.
#
# References
# ----------
#
# .. [Wang15] Limin Wang, Yuanjun Xiong, Zhe Wang, and Yu Qiao. \
#     "Towards Good Practices for Very Deep Two-Stream ConvNets." \
#     arXiv preprint arXiv:1507.02159 (2015).
#
# .. [Wang16] Limin Wang, Yuanjun Xiong, Zhe Wang, Yu Qiao, Dahua Lin, Xiaoou Tang and Luc Van Gool. \
#     "Temporal Segment Networks: Towards Good Practices for Deep Action Recognition." \
#     In European Conference on Computer Vision (ECCV). 2016.
