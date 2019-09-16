/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_KVSTORE_COMM_H_
#define MXNET_KVSTORE_COMM_H_
#include <dmlc/omp.h>
#include <string>
#include <algorithm>
#include <utility>
#include <limits>
#include <vector>
#include <tuple>
#include <thread>
#include "mxnet/ndarray.h"
#include "gradient_compression.h"
#include "../ndarray/ndarray_function.h"
#include "../operator/tensor/sparse_retain-inl.h"
#include "./kvstore_utils.h"
namespace mxnet {
namespace kvstore {
/**
 * \brief multiple device commmunication
 */
class Comm {
 public:
  Comm() {
    pinned_ctx_ = Context::CPUPinned(0);
  }
  virtual ~Comm() { }
  /**
   * \brief init key with the data shape and storage shape
   */
  virtual void Init(int key, const NDArrayStorageType stype,
                    const TShape& shape, int dtype = mshadow::kFloat32) = 0;
  /**
   * \brief returns src[0] + .. + src[src.size()-1]
   */
  virtual const NDArray& Reduce(
      int key, const std::vector<NDArray>& src, int priority) = 0;
  /**
   * \brief copy from src to dst[i] for every i
   */
  virtual void Broadcast(
      int key, const NDArray& src,
      const std::vector<NDArray*> dst, int priority) = 0;

  /**
   * \brief broadcast src to dst[i] with target row_ids for every i
   * \param key the identifier key for the stored ndarray
   * \param src the source row_sparse ndarray to broadcast
   * \param dst a list of destination row_sparse NDArray and its target row_ids to broadcast,
            where the row_ids are expected to be unique and sorted in row_id.data()
   * \param priority the priority of the operation
   */
  virtual void BroadcastRowSparse(int key, const NDArray& src,
                                  const std::vector<std::pair<NDArray*, NDArray>>& dst,
                                  const int priority) = 0;

  /**
   * \brief return a pinned contex
   */
  Context pinned_ctx() const {
    return pinned_ctx_;
  }

  /**
   * \brief Sets gradient compression parameters to be able to
   * perform reduce with compressed gradients
   */
  void SetGradientCompression(std::shared_ptr<GradientCompression> gc) {
    gc_ = gc;
  }

 protected:
  Context pinned_ctx_;

  std::shared_ptr<GradientCompression> gc_;

  float eta_;
  size_t batch_size_per_worker_;
  size_t num_workers_;

  size_t num_update_;
  size_t num_params_;

  float lr_decay_factor_;
  size_t lr_decay_update1_;
  size_t lr_decay_update2_;

  float momentum_;

  bool sign_sgd_;

  std::unordered_map<int, NDArray> tilde_delta_buf_;
  std::unordered_map<int, std::vector<NDArray> > delta_buf_;
  std::unordered_map<int, NDArray> err_server_buf_;
  std::unordered_map<int, std::vector<NDArray> > err_worker_buf_;
  std::unordered_map<int, std::vector<NDArray> > mom_buf_;
};

/**
 * \brief an implemention of Comm that first copy data to CPU memeory, and then
 * reduce there
 */
class CommCPU : public Comm {
 public:
  CommCPU() {
    nthread_reduction_ = dmlc::GetEnv("MXNET_KVSTORE_REDUCTION_NTHREADS", 4);
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
    // TODO(junwu) delete the following data member, now for benchmark only
    is_serial_push_ = dmlc::GetEnv("MXNET_KVSTORE_SERIAL_PUSH", 0);

    eta_ = dmlc::GetEnv("MXNET_LR", 0.1);
    batch_size_per_worker_ = dmlc::GetEnv("MXNET_BATCH_SIZE_PER_WORKER", 32);
    num_workers_ = dmlc::GetEnv("MXNET_NUM_WORKERS", 8);
    LOG(INFO) << "learning rate " << eta_;
    LOG(INFO) << "batch_size_per_worker_ " << batch_size_per_worker_;
    LOG(INFO) << "num_workers_ " << num_workers_;

    num_params_ = 0;
    num_update_ = 0;

    lr_decay_factor_ = dmlc::GetEnv("MXNET_LR_DECAY_FACTOR", 0.1);
    // default 100-th epoch, same with `lr_decay_updates[0]` in the training script
    lr_decay_update1_ = dmlc::GetEnv("MXNET_LR_DECAY_UPDATE1", 19531);
    // default 150-th epoch, same with `lr_decay_updates[1]` in the training script
    lr_decay_update2_ = dmlc::GetEnv("MXNET_LR_DECAY_UPDATE2", 29296);
    LOG(INFO) << "lr_decay_factor_ " << lr_decay_factor_;
    LOG(INFO) << "lr_decay_update1_ " << lr_decay_update1_;
    LOG(INFO) << "lr_decay_update2_ " << lr_decay_update2_;

    momentum_ = dmlc::GetEnv("MXNET_MOMENTUM", 0.9);
    LOG(INFO) << "momentum_ " << momentum_;

    sign_sgd_ = dmlc::GetEnv("MXNET_SIGNSGD", false);
    LOG(INFO) << "sign_sgd_ " << sign_sgd_;
  }
  virtual ~CommCPU() { }

  void Init(int key, const NDArrayStorageType stype, const TShape& shape,
            int type = mshadow::kFloat32) override {
    // Delayed allocation - the dense merged buffer might not be used at all if push()
    // only sees sparse arrays
    bool delay_alloc = true;
    merge_buf_[key].merged = NDArray(shape, pinned_ctx_, delay_alloc, type);

    NDArray &err_server = err_server_buf_[key];
    std::vector<NDArray> &err_worker_vec = err_worker_buf_[key];
    std::vector<NDArray> &delta_vec = delta_buf_[key];
    NDArray &tilde_delta = tilde_delta_buf_[key];

    err_server = NDArray(TShape{static_cast<int64_t>(shape.Size())},
                         pinned_ctx_, false, mshadow::kFloat32);
    tilde_delta = NDArray(TShape{static_cast<int64_t>(shape.Size())},
                          pinned_ctx_, false, mshadow::kFloat32);
    err_server = 0.;
    tilde_delta = 0.;

    err_worker_vec.resize(num_workers_);
    delta_vec.resize(num_workers_);

    for (NDArray &arr : err_worker_vec) {
      arr = NDArray(TShape{static_cast<int64_t>(shape.Size())},
                    pinned_ctx_, false, mshadow::kFloat32);
      arr = 0.;
    }
    for (NDArray &arr : delta_vec) {
      arr = NDArray(TShape{static_cast<int64_t>(shape.Size())},
                    pinned_ctx_, false, mshadow::kFloat32);
      arr = 0.;
    }

    std::vector<NDArray> &mom_vec = mom_buf_[key];
    mom_vec.resize(num_workers_);
    for (NDArray &arr : mom_vec) {
      arr = NDArray(TShape{static_cast<int64_t>(shape.Size())},
                    pinned_ctx_, false, mshadow::kFloat32);
      arr = 0.;
    }

    num_params_ += 1;
  }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    auto& buf = merge_buf_[key];
    const auto stype = src[0].storage_type();
    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      if (stype == kDefaultStorage) {
        return src[0];
      } else {
        // With 'local' kvstore, we could store the weight on CPU while compute
        // the gradient on GPU when the weight is extremely large.
        // To avoiding copying the weight to the same context of the gradient,
        // we always copy the gradient to merged buf.
        NDArray& merged = buf.merged_buf(stype);
        CopyFromTo(src[0], &merged, priority);
        return merged;
      }
    }

    NDArray& buf_merged = buf.merged_buf(stype);
    // normal dense reduce
    if (stype == kDefaultStorage) {
      std::vector<Engine::VarHandle> const_vars(src.size() - 1);
      std::vector<NDArray> reduce(src.size());
      CopyFromTo(src[0], &buf_merged, priority);
      reduce[0] = buf_merged;

      if (buf.copy_buf.empty()) {
        buf.copy_buf.resize(src.size()-1);
        for (size_t j = 0; j < src.size() - 1; ++j) {
          // allocate copy buffer
          buf.copy_buf[j] = NDArray(
            src[0].shape(), pinned_ctx_, false, src[0].dtype());
        }
      }
      CHECK(stype == buf.copy_buf[0].storage_type())
           << "Storage type mismatch detected. " << stype << "(src) vs. "
           << buf.copy_buf[0].storage_type() << "(buf.copy_buf)";
      for (size_t i = 1; i < src.size(); ++i) {
        CopyFromTo(src[i], &(buf.copy_buf[i-1]), priority);
        reduce[i] = buf.copy_buf[i-1];
        const_vars[i-1] = reduce[i].var();
      }


      NDArray &err_server = err_server_buf_[key];
      std::vector<NDArray> &err_worker_vec = err_worker_buf_[key];
      std::vector<NDArray> &delta_vec = delta_buf_[key];
      NDArray &tilde_delta = tilde_delta_buf_[key];

      std::vector<NDArray> &mom_vec = mom_buf_[key];

      std::vector<Engine::VarHandle> mutable_vars;

      mutable_vars.push_back(err_server.var());
      mutable_vars.push_back(reduce[0].var());
      for (size_t i = 0; i < err_worker_vec.size(); i++) {
        mutable_vars.push_back(err_worker_vec[i].var());
      }
      mutable_vars.push_back(tilde_delta.var());
      for (size_t i = 0; i < delta_vec.size(); i++) {
        mutable_vars.push_back(delta_vec[i].var());
      }
      for (size_t i = 0; i < mom_vec.size(); i++) {
        mutable_vars.push_back(mom_vec[i].var());
      }

      bool sign_sgd = sign_sgd_;

      Engine::Get()->PushAsync(
        [reduce, err_server, err_worker_vec, delta_vec, tilde_delta, mom_vec, sign_sgd, this](RunContext rctx,
            Engine::CallbackOnComplete on_complete) {
          //ReduceSumCPU(reduce);
          if (!sign_sgd) {
            dist_ef_compr_sgd(reduce, err_server, err_worker_vec, delta_vec, tilde_delta, mom_vec);
          } else {
            dist_sign_sgd(reduce, delta_vec, mom_vec);
          }
          on_complete();
        }, Context::CPU(), const_vars, mutable_vars,
        FnProperty::kCPUPrioritized, priority, "KVStoreReduce");

    } else {
      // sparse reduce
      std::vector<Engine::VarHandle> const_vars(src.size());
      std::vector<NDArray> reduce(src.size());

      if (buf.copy_buf.empty()) {
        buf.copy_buf.resize(src.size());
        for (size_t j = 0; j < src.size(); ++j) {
          buf.copy_buf[j] = NDArray(
            src[0].storage_type(), src[0].shape(), pinned_ctx_, true, src[0].dtype());
        }
      }
      CHECK(stype == buf.copy_buf[0].storage_type())
           << "Storage type mismatch detected. " << stype << "(src) vs. "
           << buf.copy_buf[0].storage_type() << "(buf.copy_buf)";
      for (size_t i = 0; i < src.size(); ++i) {
        CopyFromTo(src[i], &(buf.copy_buf[i]), priority);
        reduce[i] = buf.copy_buf[i];
        const_vars[i] = reduce[i].var();
      }
      Resource rsc = ResourceManager::Get()->Request(buf_merged.ctx(),
          ResourceRequest(ResourceRequest::kTempSpace));
      Engine::Get()->PushAsync(
        [reduce, buf_merged, rsc, this](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          NDArray out = buf_merged;
          is_serial_push_?
            ReduceSumCPUExSerial(reduce, &out)
            : mxnet::ndarray::ElementwiseSum(rctx.get_stream<cpu>(), rsc, reduce, &out);
          on_complete();
        }, Context::CPU(), const_vars, {buf_merged.var(), rsc.var},
        FnProperty::kCPUPrioritized, priority, "KVStoreReduce");
    }

    return buf_merged;
  }

  void Broadcast(int key, const NDArray& src,
                 const std::vector<NDArray*> dst, int priority) override {
    int mask = src.ctx().dev_mask();
    if (mask == Context::kCPU) {
      for (auto d : dst) CopyFromTo(src, d, priority);
    } else {
      // First copy data to pinned_ctx, then broadcast.
      // Note that kv.init initializes the data on pinned_ctx.
      // This branch indicates push() with ndarrays on gpus were called,
      // and the source is copied to gpu ctx.
      // Also indicates that buffers are already initialized during push().
      auto& buf = merge_buf_[key].merged_buf(src.storage_type());
      CopyFromTo(src, &buf, priority);
      for (auto d : dst) CopyFromTo(buf, d, priority);
    }
  }

  void BroadcastRowSparse(int key, const NDArray& src,
                          const std::vector<std::pair<NDArray*, NDArray>>& dst,
                          const int priority) override {
    using namespace mshadow;
    CHECK_EQ(src.storage_type(), kRowSparseStorage)
      << "BroadcastRowSparse expects row-sparse src NDArray";
    CHECK_EQ(src.ctx().dev_mask(), Context::kCPU)
      << "BroadcastRowSparse with src on gpu context not supported";
    for (const auto& dst_kv : dst) {
      NDArray* out = dst_kv.first;
      NDArray row_id = dst_kv.second;
      CHECK_EQ(out->storage_type(), kRowSparseStorage)
               << "BroadcastRowSparse expects row_sparse dst NDArray";
      CHECK_EQ(row_id.ctx().dev_mask(), Context::kCPU)
               << "BroadcastRowSparse with row_indices on gpu context not supported";
      // retain according to unique indices
      const bool is_same_ctx = out->ctx() == src.ctx();
      const bool is_diff_var = out->var() != src.var();
      NDArray retained_cpu = (is_same_ctx && is_diff_var) ? *out :
          NDArray(kRowSparseStorage, src.shape(), src.ctx(), true,
                  src.dtype(), src.aux_types());
      if (!is_diff_var) {
        common::LogOnce("The output of row_sparse_pull() on key " + std::to_string(key) +
                        "refers to the same NDArray as the one stored in KVStore."
                        "Performing row_sparse_pull() with such output is going to change the "
                        "data stored in KVStore. Incorrect result may be generated "
                        "next time row_sparse_pull() is called. To avoid such an issue,"
                        "consider create a new NDArray buffer to store the output.");
      }
      Engine::Get()->PushAsync(
        [=](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob& indices = row_id.data();
          NDArray temp = retained_cpu;  // get rid the of const qualifier
          op::SparseRetainOpForwardRspImpl<cpu>(rctx.get_stream<cpu>(),
                                                src, indices, kWriteTo,
                                                &temp);
          on_complete();
        }, Context::CPU(), {src.var(), row_id.var()}, {retained_cpu.var()},
        FnProperty::kNormal, priority, "KVStoreSparseRetain");
      // if retained_cpu == out, CopyFromTo will ignore the copy operation
      CopyFromTo(retained_cpu, out, priority);
    }
  }

 private:
  inline void dist_ef_compr_sgd(const std::vector<NDArray> &in_data,
                                const NDArray &err_server, const std::vector<NDArray> &err_worker_vec,
                                const std::vector<NDArray> &delta_vec, const NDArray &tilde_delta,
                                const std::vector<NDArray> &mom_vec) {
    CHECK_EQ(in_data.size(), num_workers_);
    CHECK_EQ(in_data[0].dtype(), mshadow::kFloat32);
    CHECK_EQ(err_worker_vec.size(), num_workers_);
    CHECK_EQ(delta_vec.size(), num_workers_);
    CHECK_EQ(mom_vec.size(), num_workers_);

    const size_t dim = in_data[0].shape().Size();

    // gradients
    std::vector<float*> in_dptrs(in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      in_dptrs[i] = in_data[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    std::vector<float*> mom_dptrs(in_data.size());
    for (size_t i = 0; i < mom_vec.size(); ++i) {
      mom_dptrs[i] = mom_vec[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    // result buffer to broadcast to all workers
    float *red_ptr = in_data[0].data().FlatTo2D<cpu, float>().dptr_;

    float *err_server_ptr = err_server.data().FlatTo2D<cpu, float>().dptr_;
    float *tilde_delta_ptr = tilde_delta.data().FlatTo2D<cpu, float>().dptr_;

    std::vector<float*> err_worker_ptrs(num_workers_);
    for (size_t i = 0; i < err_worker_vec.size(); ++i) {
      err_worker_ptrs[i] = err_worker_vec[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    std::vector<float*> delta_ptrs(num_workers_);
    for (size_t i = 0; i < delta_ptrs.size(); ++i) {
      delta_ptrs[i] = delta_vec[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    bool lr_decay = false;

    if (num_update_ / num_params_ == lr_decay_update1_) {
      //eta_ *= lr_decay_factor_;
      //LOG(INFO) << "In comm.h, lr decays to " << eta_;
      lr_decay = true;
    }
    if (num_update_ / num_params_ == lr_decay_update2_) {
      //eta_ *= lr_decay_factor_;
      //LOG(INFO) << "In comm.h, lr decays to " << eta_;
      lr_decay = true;
    }

    num_update_++;


    // on worker, compute delta
    for (size_t i = 0; i < num_workers_; i++) {
      float l1_norm = 0.;
      float val = 0.;
      for (size_t j = 0; j < dim; j++) {
        // During normal multi-gpu training, module will assign batch_size/num_gpus data points to each gpu,
        // optimizer has an argument `rescale_grad` which is set to batch_size and used to compute the mean.
        // Here we use batch_size/num_gpus to normalize the gradient.
        // p
        float grad = in_dptrs[i][j] / batch_size_per_worker_;
        float err = 0;
        if (lr_decay) {
          err = 10.*err_worker_ptrs[i][j];
        } else {
          err = err_worker_ptrs[i][j];
        }
        if (momentum_ > 0.0) {
          mom_dptrs[i][j] = momentum_ * mom_dptrs[i][j] + grad;
          val = momentum_ * mom_dptrs[i][j] + grad + err;
        } else {
          val = grad + err;
        }
        l1_norm += std::abs(val);
        if (val >= 0) {
          delta_ptrs[i][j] = 1;
        } else {
          delta_ptrs[i][j] = -1;
        }
      }
      for (size_t j = 0; j < dim; j++) {
        delta_ptrs[i][j] *= (l1_norm / dim);
        // recompute p
        float grad = in_dptrs[i][j] / batch_size_per_worker_;
        float err = 0;
        if (lr_decay) {
          err = 10.*err_worker_ptrs[i][j];
        } else {
          err = err_worker_ptrs[i][j];
        }
        if (momentum_ > 0.0) {
          val = momentum_ * mom_dptrs[i][j] + grad + err;
        } else {
          val = grad + err;
        }
        // update err, e = p - delta
        err_worker_ptrs[i][j] = val - delta_ptrs[i][j];
      }
    }

    // on server, compute tilde_delta
    float l1_norm = 0.;
    float val = 0.;
    for (size_t j = 0; j < dim; j++) {
      val = 0.;
      for (size_t i = 0; i < num_workers_; i++) {
        val += delta_ptrs[i][j];
      }
      // tilde_p
      if (lr_decay) {
        val = val / num_workers_ + 10.*err_server_ptr[j];
      } else {
        val = val / num_workers_ + err_server_ptr[j];
      }
      l1_norm += std::abs(val);
      if (val >= 0) {
        tilde_delta_ptr[j] = 1;
      } else {
        tilde_delta_ptr[j] = -1;
      }
    }

    for (size_t j = 0; j < dim; j++) {
      tilde_delta_ptr[j] *= (l1_norm / dim);
      // recompute tilde_p
      val = 0.;
      for (size_t i = 0; i < num_workers_; i++) {
        val += delta_ptrs[i][j];
      }
      if (lr_decay) {
        val = val / num_workers_ + 10.*err_server_ptr[j];
      } else {
        val = val / num_workers_ + err_server_ptr[j];
      }
      // update err, e = tilde_p - tilde_delta
      err_server_ptr[j] = val - tilde_delta_ptr[j];
    }

    // will broadcasdt to all workers in pull operation
    for (size_t j = 0; j < dim; j++) {
      red_ptr[j] = tilde_delta_ptr[j];
    }
  }


  inline void dist_sign_sgd(const std::vector<NDArray> &in_data,
                            const std::vector<NDArray> &delta_vec,
                            const std::vector<NDArray> &mom_vec) {
    CHECK_EQ(in_data.size(), num_workers_);
    CHECK_EQ(in_data[0].dtype(), mshadow::kFloat32);
    CHECK_EQ(mom_vec.size(), num_workers_);

    const size_t dim = in_data[0].shape().Size();

    // gradients
    std::vector<float*> in_dptrs(in_data.size());
    for (size_t i = 0; i < in_data.size(); ++i) {
      in_dptrs[i] = in_data[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    std::vector<float*> mom_dptrs(in_data.size());
    for (size_t i = 0; i < mom_vec.size(); ++i) {
      mom_dptrs[i] = mom_vec[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    std::vector<float*> delta_ptrs(num_workers_);
    for (size_t i = 0; i < delta_ptrs.size(); ++i) {
      delta_ptrs[i] = delta_vec[i].data().FlatTo2D<cpu, float>().dptr_;
    }

    // result buffer to broadcast to all workers
    float *red_ptr = in_data[0].data().FlatTo2D<cpu, float>().dptr_;

    // on workers
    for (size_t i = 0; i < num_workers_; i++) {
      for (size_t j = 0; j < dim; j++) {
        // During normal multi-gpu training, module will assign batch_size/num_gpus data points to each gpu,
        // optimizer has an argument `rescale_grad` which is set to batch_size and used to compute the mean.
        // Here we use batch_size/num_gpus to normalize the gradient.
        float grad = in_dptrs[i][j] / batch_size_per_worker_;
        if (momentum_ > 0.0) {
          mom_dptrs[i][j] = momentum_ * mom_dptrs[i][j] + (1 - momentum_) * grad;
          grad = mom_dptrs[i][j];
        }
        if (grad > 0) {
          delta_ptrs[i][j] = 1;
        } else if (grad < 0) {
          delta_ptrs[i][j] = -1;
        } else {
          delta_ptrs[i][j] = 0;
        }
      }
    }

    // on server
    for (size_t j = 0; j < dim; j++) {
      float val = 0.;
      for (size_t i = 0; i < num_workers_; i++) {
        val += delta_ptrs[i][j];
      }
      val = val / num_workers_;
      if (val > 0) {
        red_ptr[j] = 1;
      } else if (val < 0) {
        red_ptr[j] = -1;
      } else {
        red_ptr[j] = 0;
      }
    }
  }


  // reduce sum into val[0]
  inline void ReduceSumCPU(const std::vector<NDArray> &in_data) {
    MSHADOW_TYPE_SWITCH(in_data[0].dtype(), DType, {
      std::vector<DType*> dptr(in_data.size());
      for (size_t i = 0; i < in_data.size(); ++i) {
        TBlob data = in_data[i].data();
        CHECK(data.CheckContiguous());
        dptr[i] = data.FlatTo2D<cpu, DType>().dptr_;
      }
      size_t total = in_data[0].shape().Size();
      ReduceSumCPUImpl(dptr, total);
    });
  }

  // serial implementation of reduce sum for row sparse NDArray.
  inline void ReduceSumCPUExSerial(const std::vector<NDArray> &in, NDArray *out) {
    using namespace rowsparse;
    using namespace mshadow;
    auto stype = out->storage_type();
    CHECK_EQ(stype, kRowSparseStorage) << "Unexpected storage type " << stype;
    size_t total_num_rows = 0;
    size_t num_in = in.size();
    // skip the ones with empty indices and values
    std::vector<bool> skip(num_in, false);
    // the values tensor of the inputs
    MSHADOW_TYPE_SWITCH(out->dtype(), DType, {
      MSHADOW_IDX_TYPE_SWITCH(out->aux_type(kIdx), IType, {
        std::vector<Tensor<cpu, 2, DType>> in_vals(num_in);
        std::vector<Tensor<cpu, 1, IType>> in_indices(num_in);
        // offset to the values tensor of all inputs
        std::vector<size_t> offsets(num_in, 0);
        std::vector<size_t> num_rows(num_in, 0);
        for (size_t i = 0; i < num_in; i++) {
          if (!in[i].storage_initialized()) {
            skip[i] = true;
            continue;
          }
          auto size = in[i].aux_shape(kIdx).Size();
          num_rows[i] = size;
          total_num_rows += size;
          in_vals[i] = in[i].data().FlatTo2D<cpu, DType>();
          in_indices[i] = in[i].aux_data(kIdx).FlatTo1D<cpu, IType>();
        }
        std::vector<IType> indices;
        indices.reserve(total_num_rows);
        // gather indices from all inputs
        for (size_t i = 0; i < num_in; i++) {
          for (size_t j = 0; j < num_rows[i]; j++) {
            indices.emplace_back(in_indices[i][j]);
          }
        }
        CHECK_EQ(indices.size(), total_num_rows);
        // dedup indices
        std::sort(indices.begin(), indices.end());
        indices.resize(std::unique(indices.begin(), indices.end()) - indices.begin());
        // the one left are unique non-zero rows
        size_t nnr = indices.size();
        // allocate memory for output
        out->CheckAndAlloc({Shape1(nnr)});
        auto idx_data = out->aux_data(kIdx).FlatTo1D<cpu, IType>();
        auto val_data = out->data().FlatTo2D<cpu, DType>();

        for (size_t i = 0; i < nnr; i++) {
          // copy indices back
          idx_data[i] = indices[i];
          bool zeros = true;
          for (size_t j = 0; j < num_in; j++) {
            if (skip[j]) continue;
            size_t offset = offsets[j];
            if (offset < num_rows[j]) {
              if (indices[i] == in_indices[j][offset]) {
                if (zeros) {
                  Copy(val_data[i], in_vals[j][offset], nullptr);
                  zeros = false;
                } else {
                  val_data[i] += in_vals[j][offset];
                }
                offsets[j] += 1;
              }
            }
          }
        }
      });
    });
  }

  template<typename DType>
  inline static void ReduceSumCPU(
      const std::vector<DType*> &dptr, size_t offset, index_t size) {
    using namespace mshadow;  // NOLINT(*)
    Tensor<cpu, 1, DType> in_0(dptr[0] + offset, Shape1(size));
    for (size_t i = 1; i < dptr.size(); i+=4) {
      switch (dptr.size() - i) {
        case 1: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          in_0 += in_1;
          break;
        }
        case 2: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_2(dptr[i+1] + offset, Shape1(size));
          in_0 += in_1 + in_2;
          break;
        }
        case 3: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_2(dptr[i+1] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_3(dptr[i+2] + offset, Shape1(size));
          in_0 += in_1 + in_2 + in_3;
          break;
        }
        default: {
          Tensor<cpu, 1, DType> in_1(dptr[i] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_2(dptr[i+1] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_3(dptr[i+2] + offset, Shape1(size));
          Tensor<cpu, 1, DType> in_4(dptr[i+3] + offset, Shape1(size));
          in_0 += in_1 + in_2 + in_3 + in_4;
          break;
        }
      }
    }
  }

  template<typename DType>
  inline void ReduceSumCPUImpl(std::vector<DType*> dptr, size_t total) {
    const size_t step = std::min(bigarray_bound_, static_cast<size_t>(4 << 10));
    long ntask = (total + step - 1) / step; // NOLINT(*)
    if (total < bigarray_bound_ || nthread_reduction_ <= 1) {
      ReduceSumCPU(dptr, 0, total);
    } else {
      #pragma omp parallel for schedule(static) num_threads(nthread_reduction_)
      for (long j = 0; j < ntask; ++j) { // NOLINT(*)
        size_t k = static_cast<size_t>(j);
        size_t begin = std::min(k * step, total);
        size_t end = std::min((k + 1) * step, total);
        if (j == ntask - 1) CHECK_EQ(end, total);
        ReduceSumCPU(dptr, begin, static_cast<index_t>(end - begin));
      }
    }
  }

  /// \brief temporal space for pushing and pulling
  struct BufferEntry {
    /// \brief the merged value
    NDArray merged;
    /// \brief the cpu buffer for gpu data
    std::vector<NDArray> copy_buf;
    /// \brief the merged buffer for the given storage type
    inline NDArray& merged_buf(NDArrayStorageType stype) {
      if (stype == kDefaultStorage) {
        return merged;
      }
      CHECK(stype == kRowSparseStorage) << "unexpected storage type " << stype;
      // check if sparse_merged is initialized
      if (sparse_merged.is_none()) {
        CHECK(!merged.is_none());
        sparse_merged = NDArray(kRowSparseStorage, merged.shape(), merged.ctx(),
                                true, merged.dtype());
      }
      return sparse_merged;
    }

   private:
    /// \brief the sparse merged value
    NDArray sparse_merged;
  };
  std::unordered_map<int, BufferEntry> merge_buf_;
  size_t bigarray_bound_;
  int nthread_reduction_;
  bool is_serial_push_;
};

/**
 * \brief an implementation of Comm that performs reduction on device
 * directly.
 *
 * It is faster if the total device-to-device bandwidths is larger than
 * device-to-cpu, which is often true for 4 or 8 GPUs. But it uses more device
 * memory.
 */
class CommDevice : public Comm {
 public:
  CommDevice() {
    inited_ = false;
  }

  virtual ~CommDevice() { }

  void Init(int key, const NDArrayStorageType stype, const TShape& shape,
            int dtype = mshadow::kFloat32) override {
    sorted_key_attrs_.emplace_back(key, shape, dtype);
    inited_ = false;
  }

  void InitBuffersAndComm(const std::vector<NDArray>& src) {
    if (!inited_) {
      std::vector<Context> devs;
      for (const auto& a : src) {
        devs.push_back(a.ctx());
      }
      InitMergeBuffer(devs);
      if (dmlc::GetEnv("MXNET_ENABLE_GPU_P2P", 1)) {
        EnableP2P(devs);
      }
    }
  }

  const NDArray& ReduceRowSparse(int key, const std::vector<NDArray>& src,
                                 int priority) {
    auto& buf = merge_buf_[key];
    std::vector<NDArray> reduce(src.size());

    const NDArrayStorageType stype = src[0].storage_type();
    NDArray& buf_merged = buf.merged_buf(stype);
    if (buf.copy_buf.empty()) {
      // initialize buffer for copying during reduce
      buf.copy_buf.resize(src.size());
      for (size_t j = 0; j < src.size(); ++j) {
        buf.copy_buf[j] = NDArray(stype, src[0].shape(), buf_merged.ctx(), true, src[0].dtype());
      }
    }
    CHECK(src[0].storage_type() == buf.copy_buf[0].storage_type())
         << "Storage type mismatch detected. " << src[0].storage_type() << "(src) vs. "
         << buf.copy_buf[0].storage_type() << "(buf.copy_buf)";
    for (size_t i = 0; i < src.size(); ++i) {
      CopyFromTo(src[i], &(buf.copy_buf[i]), priority);
      reduce[i] = buf.copy_buf[i];
    }
    ElementwiseSum(reduce, &buf_merged, priority);
    return buf_merged;
  }

  const NDArray& Reduce(int key, const std::vector<NDArray>& src,
                        int priority) override {
    // when this reduce is called from kvstore_dist, gc is not set
    // we don't do compression twice in dist_sync_device
    if ((gc_ != nullptr) && (gc_->get_type() != CompressionType::kNone)) {
      return ReduceCompressed(key, src, priority);
    }

    // avoid extra copy for single device, but it may bring problems for
    // abnormal usage of kvstore
    if (src.size() == 1) {
      return src[0];
    }

    InitBuffersAndComm(src);
    auto& buf = merge_buf_[key];

    const NDArrayStorageType stype = src[0].storage_type();
    NDArray& buf_merged = buf.merged_buf(stype);
    // normal dense reduce
    if (stype == kDefaultStorage) {
      CopyFromTo(src[0], &buf_merged, priority);

      std::vector<NDArray> reduce(src.size());
      reduce[0] = buf_merged;

      if (buf.copy_buf.empty()) {
        // TODO(mli) this results in large device memory usage for huge ndarray,
        // such as the largest fullc in VGG. consider to do segment reduce with
        // NDArray.Slice or gpu direct memory access. for the latter, we need to
        // remove some ctx check, and also it reduces 20% perf
        buf.copy_buf.resize(src.size()-1);
        for (size_t i = 0; i < src.size()-1; ++i) {
          buf.copy_buf[i] = NDArray(
            buf_merged.shape(), buf_merged.ctx(), false, buf_merged.dtype());
        }
      }
      for (size_t i = 0; i < src.size()-1; ++i) {
        CopyFromTo(src[i+1], &(buf.copy_buf[i]), priority);
        reduce[i+1] = buf.copy_buf[i];
      }
      ElementwiseSum(reduce, &buf_merged, priority);
    } else {
      // sparse reduce
      buf_merged = ReduceRowSparse(key, src, priority);
    }
    return buf_merged;
  }

  const NDArray& ReduceCompressed(int key, const std::vector<NDArray>& src,
                                  int priority) {
    InitBuffersAndComm(src);
    auto& buf = merge_buf_[key];
    std::vector<NDArray> reduce(src.size());
    if (buf.copy_buf.empty()) {
      // one buf for each context
      buf.copy_buf.resize(src.size());
      buf.compressed_recv_buf.resize(src.size());
      buf.compressed_send_buf.resize(src.size());
      buf.residual.resize(src.size());

      for (size_t i = 0; i < src.size(); ++i) {
        buf.copy_buf[i] = NDArray(buf.merged.shape(), buf.merged.ctx(),
                                  false, buf.merged.dtype());
        buf.residual[i] = NDArray(buf.merged.shape(), src[i].ctx(),
                                  false, buf.merged.dtype());
        buf.residual[i] = 0;
        int64_t small_size = gc_->GetCompressedSize(buf.merged.shape().Size());
        buf.compressed_recv_buf[i] = NDArray(TShape{small_size}, buf.merged.ctx(),
                                        false, buf.merged.dtype());
        buf.compressed_send_buf[i] = NDArray(TShape{small_size}, src[i].ctx(),
                                        false, buf.merged.dtype());
      }
    }

    for (size_t i = 0; i < src.size(); ++i) {
      // compress before copy
      // this is done even if the data is on same context as copy_buf because
      // we don't want the training to be biased towards data on this GPU
      gc_->Quantize(src[i], &(buf.compressed_send_buf[i]), &(buf.residual[i]), priority);

      if (buf.compressed_send_buf[i].ctx() != buf.compressed_recv_buf[i].ctx()) {
        CopyFromTo(buf.compressed_send_buf[i], &(buf.compressed_recv_buf[i]), priority);
      } else {
        // avoid memory copy when they are on same context
        buf.compressed_recv_buf[i] = buf.compressed_send_buf[i];
      }

      gc_->Dequantize(buf.compressed_recv_buf[i], &(buf.copy_buf[i]), priority);
      reduce[i] = buf.copy_buf[i];
    }
    ElementwiseSum(reduce, &buf.merged);
    return buf.merged;
  }

  void Broadcast(int key, const NDArray& src,
                 const std::vector<NDArray*> dst, int priority) override {
    if (!inited_) {
      // copy to a random device first
      int dev_id = key % dst.size();
      CopyFromTo(src, dst[dev_id], priority);
      for (size_t i = 0; i < dst.size(); ++i) {
        if (i != static_cast<size_t>(dev_id)) {
          CopyFromTo(*dst[dev_id], dst[i], priority);
        }
      }
    } else {
      auto& buf_merged = merge_buf_[key].merged_buf(src.storage_type());
      CopyFromTo(src, &buf_merged, priority);
      for (auto d : dst) {
        CopyFromTo(buf_merged, d, priority);
      }
    }
  }

  void BroadcastRowSparse(int key, const NDArray& src,
                          const std::vector<std::pair<NDArray*, NDArray>>& dst,
                          const int priority) override {
    CHECK_EQ(src.storage_type(), kRowSparseStorage)
      << "BroadcastRowSparse expects row-sparse src NDArray";

    for (const auto& dst_kv : dst) {
      NDArray* out = dst_kv.first;
      NDArray row_id = dst_kv.second;
      CHECK_EQ(out->storage_type(), kRowSparseStorage)
               << "BroadcastRowSparse expects row_sparse dst NDArray";
      CHECK_EQ(row_id.ctx(), src.ctx())
              << "row_id and src are expected to be on the same context";

      // retain according to indices
      const bool is_same_ctx = out->ctx() == src.ctx();
      const bool is_diff_var = out->var() != src.var();
      NDArray retained_gpu = (is_same_ctx && is_diff_var) ? *out :
          NDArray(kRowSparseStorage, out->shape(), src.ctx(), true,
                  out->dtype(), out->aux_types());
      if (!is_diff_var) {
        common::LogOnce("The output of row_sparse_pull() on key " + std::to_string(key) +
                        "refers to the same NDArray as the one stored in KVStore."
                        "Performing row_sparse_pull() with such output is going to change the "
                        "data stored in KVStore. Incorrect result may be generated "
                        "next time row_sparse_pull() is called. To avoid such an issue,"
                        "consider create a new NDArray buffer to store the output.");
      }
      bool is_gpu = retained_gpu.ctx().dev_mask() == gpu::kDevMask;
      Engine::Get()->PushAsync([=](RunContext rctx, Engine::CallbackOnComplete on_complete) {
          const TBlob& indices = row_id.data();
          using namespace mxnet::common;
          NDArray temp = retained_gpu;
          switch (temp.ctx().dev_mask()) {
            case cpu::kDevMask: {
              SparseRetainOpForwardRspWrapper<cpu>(rctx.get_stream<cpu>(),
                  src, indices, kWriteTo, &temp);
              break;
            }
#if MXNET_USE_CUDA
            case gpu::kDevMask: {
              SparseRetainOpForwardRspWrapper<gpu>(rctx.get_stream<gpu>(),
                  src, indices, kWriteTo, &temp);
              // wait for GPU operations to complete
              rctx.get_stream<gpu>()->Wait();
              break;
            }
#endif
            default: LOG(FATAL) << MXNET_GPU_NOT_ENABLED_ERROR;
          }
          on_complete();
        }, retained_gpu.ctx(), {src.var(), row_id.var()}, {retained_gpu.var()},
      is_gpu ? FnProperty::kGPUPrioritized : FnProperty::kCPUPrioritized,
      priority, "KVStoreSparseRetain");
      CopyFromTo(retained_gpu, out, priority);
    }
  }

  using KeyAttrs = std::tuple<int, TShape, int>;
  // try to allocate buff on device evenly
  void InitMergeBuffer(const std::vector<Context>& devs) {
    std::sort(sorted_key_attrs_.begin(), sorted_key_attrs_.end(), [](
              const KeyAttrs& a, const KeyAttrs& b) {
      return std::get<1>(a).Size() > std::get<1>(b).Size();
    });

    std::unordered_map<int, std::pair<Context, size_t>> ctx_info;
    for (auto d : devs) {
      ctx_info[d.dev_id] = std::make_pair(d, 0);
    }

    for (auto& sorted_key_attr : sorted_key_attrs_) {
      const int key  = std::get<0>(sorted_key_attr);
      const TShape& shape = std::get<1>(sorted_key_attr);
      const int type = std::get<2>(sorted_key_attr);
      auto& buf = merge_buf_[key];
      Context ctx;
      size_t min_size = std::numeric_limits<size_t>::max();
      for (auto& ctx_info_kv : ctx_info) {
        size_t size = ctx_info_kv.second.second;
        if (size <= min_size) {
          ctx = ctx_info_kv.second.first;
          min_size = size;
        }
      }
      // Delayed allocation - as the dense merged buffer might not be used at all if push()
      // only sees sparse arrays
      if (buf.merged.is_none()) {
        bool delay_alloc = true;
        buf.merged = NDArray(shape, ctx, delay_alloc, type);
      }
      ctx_info[ctx.dev_id].second += shape.Size();
    }
    inited_ = true;
  }

 private:
  void EnableP2P(const std::vector<Context>& devs) {
#if MXNET_USE_CUDA
    std::vector<int> gpus;
    for (const auto& d : devs) {
      if (d.dev_mask() == gpu::kDevMask) {
        gpus.push_back(d.dev_id);
      }
    }
    int n = static_cast<int>(gpus.size());
    int enabled = 0;
    std::vector<int> p2p(n*n);

    for (int i = 0; i < n; ++i) {
      // Restores active device to what it was before EnableP2P
      mxnet::common::cuda::DeviceStore device_store(gpus[i]);
      for (int j = 0; j < n; j++) {
        int access;
        cudaDeviceCanAccessPeer(&access, gpus[i], gpus[j]);
        if (access) {
          cudaError_t e = cudaDeviceEnablePeerAccess(gpus[j], 0);
          if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
            ++enabled;
            p2p[i*n+j] = 1;
          }
        }
      }
    }
    if (enabled != n*(n-1)) {
      // print warning info if not fully enabled
      LOG(WARNING) << "only " << enabled <<  " out of "
                   << n*(n-1) << " GPU pairs are enabled direct access. "
                   << "It may affect the performance. "
                   << "You can set MXNET_ENABLE_GPU_P2P=0 to turn it off";
      std::string access(n, '.');
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          access[j] = p2p[i*n+j] ? 'v' : '.';
        }
        LOG(WARNING) << access;
      }
    }
#endif
  }

  /// \brief temporal space for pushing and pulling
  struct BufferEntry {
    /// \brief the dense merged value for reduce and broadcast operations
    NDArray merged;
    /// \brief the gpu buffer for copy during reduce operation
    std::vector<NDArray> copy_buf;
    /// \brief the residual buffer for gradient compression
    std::vector<NDArray> residual;
    /// \brief the small buffer for compressed data in sender
    std::vector<NDArray> compressed_send_buf;
    /// \brief the small buffer for compressed data in receiver
    std::vector<NDArray> compressed_recv_buf;

    /// \brief the merged buffer for the given storage type (could be either dense or row_sparse)
    inline NDArray& merged_buf(NDArrayStorageType stype) {
      if (stype == kDefaultStorage) {
        CHECK(!merged.is_none()) << "unintialized merge buffer detected";
        return merged;
      }
      CHECK(stype == kRowSparseStorage) << "unexpected storage type " << stype;
      // check if sparse_merged is initialized
      if (sparse_merged.is_none()) {
        CHECK(!merged.is_none());
        sparse_merged = NDArray(kRowSparseStorage, merged.shape(), merged.ctx(),
                                true, merged.dtype());
      }
      return sparse_merged;
    }

   private:
    /// \brief the sparse merged value for reduce and rowsparse broadcast operations
    NDArray sparse_merged;
  };
  std::unordered_map<int, BufferEntry> merge_buf_;

 public:
  bool inited_;
  std::vector<KeyAttrs> sorted_key_attrs_;
};

}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_COMM_H_
