/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_ITERATOR_HPP_
#define _MLI_ITERATOR_HPP_

#include "mli_api.h"
#include "mli_math_macros.h"
#include "mli_types.h"
#include "mli_types.hpp"

namespace snps_arc::metaware::mli {

/**
 * @brief 
 *
 *
 */
template <unsigned maxRank>
class IteratorCfg {

  public:
    /**
     * @brief constructor
     *
     */
    IteratorCfg() {
        for (uint32_t i = 0; i < maxRank; i++){
          first_increment_[i] = 1;
          increment_[i] = 1;
          first_size_[i] = 1;
          size_[i] = 1;
        }
        rank_ = maxRank;
    }
    /**
     * @brief constructor
     *
     *
     * @param first_increment [I] increment per dimension in elements for the first increment on each dimension
     * @param increment       [I] increment per dimension in elements 
     * @param first_size      [I] size in each dimension for the first tile in that dimension
     * @param size            [I] size in each dimension for the remaining tiles in that dimension
     *                            Note that the last tile could be smaller as it is clipped to the shape of the full tensor.
     * @param rank            [I] Optional rank in case the rank is smaller than the maxRank template parameter.
     */
    IteratorCfg(int32_t first_increment[],
                int32_t increment[],
                uint32_t first_size[],
                uint32_t size[],
                unsigned rank = maxRank
                ) {
        for (unsigned i = 0; i < rank; i++){
          first_increment_[i] = first_increment[i];
          increment_[i] = increment[i];
          first_size_[i] = first_size[i];
          size_[i] = size[i];
        }
        rank_ = rank;
    }
    /**
     * @brief constructor
     *
     *
     * @param increment       [I] increment per dimension in elements 
     * @param size            [I] size in each dimension for the remaining tile in that dimension
     *                            Note that the last tile could be smaller as it is clipped to the shape of the full tensor.
     * @param rank            [I] Optional rank in case the rank is smaller than the maxRank template parameter.
     */
    IteratorCfg(int32_t increment[],
                uint32_t size[],
                unsigned rank = maxRank
                ) {
        for (unsigned i = 0; i < rank; i++){
          first_increment_[i] = increment[i];
          increment_[i] = increment[i];
          first_size_[i] = size[i];
          size_[i] = size[i];
        }
        rank_ = rank;
    }

    /**
     * @brief set config from smaller rank configuration
     *
     * @param in       [I] IteratorCfg with a smaller rank that is stored inside this cfg
     */
    template <unsigned N>
    void set_config(const IteratorCfg<N> in) {
        for (uint32_t i = 0; i < N; i++){
          first_increment_[i] = in.get_first_increment(i);
          increment_[i] = in.get_increment(i);
          first_size_[i] = in.get_first_size(i);
          size_[i] = in.get_size(i);
        }
        for (uint32_t i = N; i < maxRank; i++){
          first_increment_[i] = 0;
          increment_[i] = 0;
          first_size_[i] = 0;
          size_[i] = 0;
        }
        rank_ = in.get_rank();
    }

    template <typename buf_T>
    void set_config_single_tile(const Tensor<buf_T, maxRank> tensor){
        for (uint32_t i = 0; i < maxRank; i++){
          first_increment_[i] = tensor.get_dim(i);
          increment_[i] = tensor.get_dim(i);
          first_size_[i] = tensor.get_dim(i);
          size_[i] = tensor.get_dim(i);
        }
        rank_ = maxRank;
    }

    IteratorCfg<maxRank> transpose(uint32_t new_order[]) const {
        // create a transposed Iterator, reordering the dimensions
        IteratorCfg<maxRank> iter;
        // change order of axes
        uint32_t c = 0;
        for (uint32_t axis = 0; axis < maxRank; axis++) {
          assert(new_order[axis] >= 0 && new_order[axis] < maxRank);
          // axis can only be selected once
          assert((c & (1 << new_order[axis])) == 0);
          c |= (1 << new_order[axis]);
          iter.first_increment_[axis] = first_increment_[new_order[axis]];
          iter.increment_[axis] = increment_[new_order[axis]];
          iter.first_size_[axis] = first_size_[new_order[axis]];
          iter.size_[axis] = size_[new_order[axis]];
        }
        return iter;
    }

    unsigned get_first_increment(unsigned dim) const {
      return first_increment_[dim];
    }
    unsigned get_increment(unsigned dim) const {
      return increment_[dim];
    }
    unsigned get_first_size(unsigned dim) const {
      return first_size_[dim];
    }
    unsigned get_size(unsigned dim) const {
      return size_[dim];
    }
    unsigned get_rank() const {
      return rank_;
    }

  private:
  //TODO: change to unsigned where need
    int32_t first_increment_[maxRank];
    int32_t increment_[maxRank];
    uint32_t first_size_[maxRank];
    uint32_t size_[maxRank];
  //TODO: add max[] parameter
    unsigned rank_;
};
/**
 * @brief 
 *
 *
 */

template <unsigned maxRank>
class TensorIterator {

  public:
    /**
     * @brief constructor
     *
     *
     * @param tensor [I] full tensor which is tiled by this iterator
     * @param config [I] iterator configuration
     */
    TensorIterator(Tensor<InternalBuffer, maxRank> tensor, IteratorCfg<maxRank> config){
      full_tensor_ = tensor;
      config_ = config;
      Reset();
    }

    /**
     * @brief constructor
     *
     * simple iterator with default iterator config will iterate in steps of 1
     * @param tensor [I] full tensor which is tiled by this iterator
     */
    TensorIterator(Tensor<InternalBuffer, maxRank> tensor)
    : config_(){
      full_tensor_ = tensor;
      Reset();
    }

    /**
     * @brief constructor
     *
     * empty constructor
     */
    TensorIterator()
    : config_(),
      full_tensor_(){
      Reset();
    }

    /**
     * @brief Reset the iterator to zero position
     *
     * This method will reset the internal position to zero
     */
    mli_status Reset() {
      for (uint32_t i = 0; i < maxRank; i++ ){
        pos_[i] = 0;
      }
      offset_ = 0;
      return MLI_STATUS_OK;
    }

    bool Next() {
      bool done = false;
      int rank = config_.get_rank();
      for (int r = 0; r < rank; r++) {
        if (pos_[r] == 0) {
            pos_[r] = config_.get_first_increment(r);
            offset_ += config_.get_first_increment(r) * full_tensor_.get_mem_stride(r);
        } else {
            pos_[r] += config_.get_increment(r);
            offset_ += config_.get_increment(r) * full_tensor_.get_mem_stride(r);
        }
        if (pos_[r] >= full_tensor_.get_dim(r)) {
          // end of axis, reset axis iterator to 0 and continue with next axis
          offset_ -= pos_[r] * full_tensor_.get_mem_stride(r);
          pos_[r] = 0;
          done = (r == rank - 1) ? true : false;
        } else {
          // not at end of axis, break
          break;
        }
      }
      return done;
    }

    Tensor<InternalBuffer, maxRank> GetSubTensor() {
      uint32_t copysize[maxRank];
      unsigned r = 0;
      for (r = 0; r < config_.get_rank(); r++) {
        copysize[r] = (pos_[r] == 0)? config_.get_first_size(r) : config_.get_size(r);
        copysize[r] = MIN(copysize[r], full_tensor_.get_dim(r) - pos_[r]);
      }
      for ( ; r < maxRank; r++){
        copysize[r] = 1;
      }
      return full_tensor_.slice(pos_, copysize);
    }

    TensorIterator<maxRank> transpose(uint32_t new_order[]) const {
      return TensorIterator(full_tensor_.transpose(new_order), config_.transpose(new_order));
    }

  template<typename T>
  T read(){
    return full_tensor_.template read<T>(offset_);
  }

  template<typename T>
  void write(T data){
    full_tensor_.write(offset_, data);
  }

private:
  uint32_t pos_[maxRank];
  int32_t offset_;
  Tensor<InternalBuffer, maxRank> full_tensor_;
  IteratorCfg<maxRank> config_;
};

} // namespace mli

#endif // _MLI_ITERATOR_HPP_
