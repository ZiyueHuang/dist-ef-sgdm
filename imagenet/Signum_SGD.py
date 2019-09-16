#coding=utf-8
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import time
import os
#Signum with majority vote


class SGD_distribute(Optimizer):

    def __init__(self, params, args, log_writer, **kwargs):

        lr = args.lr
        momentum = args.momentum
        weight_decay = args.weight_decay
        compression_buffer = args.compress
        all_reduce = args.all_reduce
        local_rank = args.local_rank
        gpus_per_machine = args.gpus_per_machine

        self.err_buf = []
        self.prev_lr = 0
        self.server_err_buf = []

        self.compression_buffer = compression_buffer
        self.all_reduce = all_reduce
        self.signum = args.signum
        self.log_writer = log_writer

        self.args = args

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)

        super(SGD_distribute, self).__init__(params, defaults)

        self.MB = 1024 * 1024
        self.bucket_size = 50 * self.MB

        if self.compression_buffer:
            import compressor

            self.compressor = compressor.compressor(using_cuda = True, local_rank = local_rank)
            self.local_rank = local_rank
            self.global_rank = dist.get_rank()
            self.local_dst_in_global = self.global_rank - self.local_rank

            self.inter_node_group = []
            self.nodes = dist.get_world_size() // gpus_per_machine

            self.intra_node_group_list = []
            for index in range(self.nodes):
                # set inter_node_group
                self.inter_node_group.append(0 + index * gpus_per_machine)
                # set all intra_node_group
                intra_node_group_temp = []
                for intra_index in range(gpus_per_machine):
                    intra_node_group_temp.append(intra_index + index * gpus_per_machine)
                intra_node_group_temp = dist.new_group(intra_node_group_temp)
                self.intra_node_group_list.append(intra_node_group_temp)

                if self.local_dst_in_global == 0 + index * gpus_per_machine:
                    self.nodes_rank = index


            #self.intra_node_list = self.intra_node_group
            self.inter_node_list = self.inter_node_group
            self.inter_node_group_list = []
            for index in range(len(self.inter_node_list)):
                if index is not 0:
                    temp = dist.new_group([self.inter_node_list[0],self.inter_node_list[index]])
                    self.inter_node_group_list.append(temp)
            self.all_gpu = dist.new_group()

            self.all_inter_node_group = dist.new_group(self.inter_node_list)

            if dist.get_rank() == 0 or dist.get_rank() == 8:
                print('nodes', self.nodes)
                print('intra_node_group_list',self.intra_node_group_list)
                print('inter_node_group',self.inter_node_group_list)
                print('all_inter_node_group', self.inter_node_list)

    def __setstate__(self, state):
        super(SGD_distribute, self).__setstate__(state)


    def step(self, closure=None):

        args = self.args

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            cur_lr = group['lr']

            all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if self.compression_buffer==False:
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    # signum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']

                    buf.mul_(momentum).add_(d_p)
                    d_p.add_(momentum, buf)

                all_grads.append(d_p)

            length = 0
            for _ in _take_tensors(all_grads, self.bucket_size):
                length += 1

            dev_grads_buckets = _take_tensors(all_grads, self.bucket_size)
            for i, dev_grads in enumerate(dev_grads_buckets):
                d_p_new = _flatten_dense_tensors(dev_grads)

                if len(self.err_buf) < length:
                    self.err_buf.append(torch.zeros_like(d_p_new))
                    self.server_err_buf.append(torch.zeros_like(d_p_new))

                err_buf = self.err_buf[i]
                server_err_buf = self.server_err_buf[i]

                d_p_new.add_(self.prev_lr/cur_lr, err_buf)

                p_buf = d_p_new

                if self.all_reduce:
                    dist.all_reduce(d_p_new) #self.all_gpu, group = 0
                    if self.signum:
                        d_p_new = torch.sign(d_p_new)
                elif self.signum:
                    if self.nodes > 1:
                        if self.compression_buffer:
                            d_p_new_scale = torch.ones(1)
                            d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()
                            d_p_new, tensor_size = self.compressor.compress(d_p_new)

                            tmp = self.compressor.uncompress(d_p_new.clone(), tensor_size)
                            tmp.mul_(d_p_new_scale.item())

                            err_buf.copy_(p_buf).sub_(tmp)
                        else:
                            d_p_new = torch.sign(d_p_new)

                        if dist.get_rank() == 0:
                            d_p_new_list = []
                            d_p_new_scale_list = []
                            for index, inter_node_group in enumerate(self.inter_node_group_list):
                                d_p_temp = d_p_new.clone()
                                d_p_scale_temp = d_p_new_scale.clone()
                                dist.broadcast(d_p_scale_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                dist.broadcast(d_p_temp, self.inter_node_list[index + 1], group = inter_node_group)
                                d_p_new_list.append(d_p_temp)
                                d_p_new_scale_list.append(d_p_scale_temp)
                        else:
                            dist.broadcast(d_p_new_scale, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])
                            dist.broadcast(d_p_new, dist.get_rank(), group = self.inter_node_group_list[self.nodes_rank - 1])
                            dist.barrier(group = self.all_inter_node_group)

                        if dist.get_rank() == 0:
                            if self.compression_buffer:
                                d_p_new_list.append(d_p_new) #count itself
                                d_p_new_scale_list.append(d_p_new_scale) #count itself
                                #d_p_new = self.compressor.majority_vote(d_p_new_list)
                                d_p_new = torch.zeros(tensor_size).cuda()
                                for d_p, d_p_scale in zip(d_p_new_list, d_p_new_scale_list):
                                    tmp = self.compressor.uncompress(d_p, tensor_size)
                                    d_p_new.add_(d_p_scale.item(), tmp)
                                d_p_new /= self.nodes

                                d_p_new.add_(self.prev_lr/cur_lr, server_err_buf)

                                un_compr = d_p_new

                                d_p_new_scale = torch.ones(1)
                                d_p_new_scale[0] = d_p_new.abs().sum().cpu().item()/d_p_new.numel()

                                d_p_new, _ = self.compressor.compress(d_p_new)

                                tmp = self.compressor.uncompress(d_p_new.clone(), tensor_size)
                                tmp.mul_(d_p_new_scale.item())

                                server_err_buf.copy_(un_compr).sub_(tmp)
                            else:
                                for d_p_temp in d_p_new_list:
                                    d_p_new.add_(d_p_temp)
                                d_p_new = d_p_new / self.nodes

                            dist.barrier(group = self.all_inter_node_group)

                        dist.broadcast(d_p_new, 0, group = self.all_inter_node_group)
                        if self.compression_buffer:
                            dist.broadcast(d_p_new_scale, 0, group = self.all_inter_node_group)

                        if self.compression_buffer:
                            d_p_new = self.compressor.uncompress(d_p_new, tensor_size)
                            d_p_new.mul_(d_p_new_scale.item())
                else:
                    print('You can not run without signum or all_reduce')

                #unflatten
                dev_grads_new = _unflatten_dense_tensors(d_p_new,dev_grads)
                for grad, reduced in zip(dev_grads, dev_grads_new):
                    grad.copy_(reduced)

            for p in group['params']:
                if self.compression_buffer: #This part of code is temporary
                    if weight_decay != 0:
                        if momentum != 0:
                            param_state = self.state[p]
                            if 'wd_mom' not in param_state:
                                buf = param_state['wd_mom'] = torch.zeros_like(p.data)
                            else:
                                buf = param_state['wd_mom']

                            buf.mul_(momentum).add_(weight_decay, p.data)
                            p.grad.data.add_(momentum, buf)

                        p.grad.data.add_(weight_decay, p.data)

                p.data.add_(-group['lr'], p.grad.data)

            self.prev_lr = group['lr']

        return loss

