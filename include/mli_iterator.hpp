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
template <unsigned iterRank>
class IteratorCfg {

  public:

    // Empty constructor
    IteratorCfg() {
      for (uint32_t i = 0; i < iterRank; ++i) {
        order_[i] = i;
        count_[i] = 1;
        size_[i] = first_size_[i] = last_size_[i] = 1;
        pos_inc_[i] = first_pos_inc_[i] = last_pos_inc_[i] = 0;
      }
    }

    // Constructor that will create config for a single tile for whole given Tensor
    template <typename buf_T>
    IteratorCfg(Tensor<buf_T, iterRank> tensor) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        order_[i] = i;
        count_[i] = 1;
        size_[i] = first_size_[i] = last_size_[i] = tensor.get_dim(i);
        pos_inc_[i] = first_pos_inc_[i] = last_pos_inc_[i] = 0;
      }
    }

    // Old-style constructor for compatibilit, to be removed
    IteratorCfg(int32_t increment[],
                uint32_t size[]
                ) {
      for (unsigned i = 0; i < iterRank; ++i) {
        order_[i] = i;
        count_[i] = size[i]*increment[i] != 0 ? CEIL_DIV(size[i], increment[i]) : 1;
        size_[i] = first_size_[i] = last_size_[i] = size[i];
        pos_inc_[i] = first_pos_inc_[i] = increment[i];
        last_pos_inc_[i] = increment[i] * (1 - count_[i]);
      }
    }

    // Constructor with explicit parameters
    IteratorCfg(int32_t order[],           // iteration order
                int32_t count[],           // iterations count
                int32_t first_increment[], // first increment values
                int32_t increment[],       // middle increment values
                int32_t last_increment[],  // last increment values
                int32_t first_size[],      // first size values
                int32_t size[],            // middle size values
                int32_t last_size[]        // last size values
                ) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        order_[i] = order[i];
        count_[i] = count[i];
        first_pos_inc_[i] = first_increment[i];
        pos_inc_[i] = increment[i];
        last_pos_inc_[i] = last_increment[i];
        first_size_[i] = first_size[i];
        size_[i] = size[i];
        last_size_[i] = last_size[i];
      }
    }

    // Constructor that will compute the number of tiles in each dimension, it will also compute the increment values and sizes.
    template <typename buf_T>
    IteratorCfg(const Tensor<buf_T, iterRank>& tensor, // full tensor to be iterated
                uint32_t tilesize[],                   // size of the tile
                int32_t order[]) {                     // iteration order
      for (uint32_t i = 0; i < iterRank; ++i) {
        int32_t dim = order[i];
        order_[i] = dim;
        count_[i] = CEIL_DIV(tensor.get_dim(dim), tilesize[dim]);
        first_size_[i] = size_[i] = tilesize[dim];
        last_size_[i] = (tensor.get_dim(dim) - 1) % tilesize[dim] + 1;
        first_pos_inc_[i] = pos_inc_[i] = tilesize[dim];
        last_pos_inc_[i] = tilesize[dim] * (1 - count_[i]);
      }
    }

    // Constructor that updates previously constructed iterator taking into account a kernel apperture and paddings
    template <typename buf_T>
    IteratorCfg(const IteratorCfg& icfg,          // origin config
                Tensor<buf_T, iterRank> tensor,   // full tensor
                uint32_t effective_kernel_size[], // used to calculate the overlaps between the tiles
                uint32_t stride[],                // used to calculate the overlaps between the tiles
                uint32_t pre_padding[]            // number of virtual pixels added before each dimension
               ) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        int32_t dim = icfg.order_[i];
        order_[i] = dim;
        count_[i] = icfg.count_[i];
        pos_inc_[i] = icfg.pos_inc_[i] * stride[dim];
        first_pos_inc_[i] = pos_inc_[i] - pre_padding[dim];
        last_pos_inc_[i] = count_[i] > 1 ? pos_inc_[i] * (1 - count_[i]) + pre_padding[dim] : 0;
        size_[i] = (icfg.size_[i] - 1) * stride[dim] + effective_kernel_size[dim];
        first_size_[i] = size_[i] - pre_padding[dim];
        last_size_[i] = tensor.get_dim(dim) + last_pos_inc_[i];
      }
    }

    // Constructor that updates previously constructed iterator for work on buffer for only one or several tiles
    IteratorCfg(const IteratorCfg& icfg, // origin config
                int32_t shrinksz         // 1 or more
               ) {
      assert(shrinksz > 0);
      for (uint32_t i = 0; i < iterRank; ++i) {
        order_[i] = icfg.order_[i];
        count_[i] = icfg.count_[i];
        if (shrinksz > 1) { // cyclic buffer for double- (or more) buffering
          if (i == 0) {
            pos_inc_[0] = icfg.pos_inc_[0];
            first_pos_inc_[0] = icfg.first_pos_inc_[0];
            last_pos_inc_[0] = icfg.last_size_[0];
          } else {
            pos_inc_[i] = first_pos_inc_[i] = last_pos_inc_[i] = 0;
          }
        } else { // buffer for single tile
          pos_inc_[i] = first_pos_inc_[i] = last_pos_inc_[i] = 0;
        }
        size_[i] = icfg.size_[i];
        first_size_[i] = icfg.first_size_[i];
        last_size_[i] = icfg.last_size_[i];
      }
    }

    // This method will remove the overlap between adjacent tiles in the first iteration dimension to avoid duplicate data transfers
    // It will update the (first)size and (first)inc parameters
    void RemoveOverlapAdjacentTiles() {
      if (count_[0] > 1) {
        int32_t overlap = size_[0] - pos_inc_[0];
        first_pos_inc_[0] = first_size_[0];
        size_[0] = pos_inc_[0];
        last_size_[0] -= overlap;
        last_pos_inc_[0] -= overlap;
      }
    }

    template <unsigned N>
    void set_config(const IteratorCfg<N>& in) {
        assert(N <= iterRank);
        uint32_t i = 0;
        for (; i < N; ++i) {
          order_[i] = in.order_[i];
          count_[i] = in.count_[i];
          size_[i] = in.size_[i];
          first_size_[i] = in.first_size_[i];
          last_size_[i] = in.last_size_[i];
          pos_inc_[i] = in.pos_inc_[i];
          first_pos_inc_[i] = in.first_pos_inc_[i];
          last_pos_inc_[i] = in.last_pos_inc_[i];
        }
        for (; i < iterRank; ++i) {
          order_[i] = -1;
          count_[i] = 0;
          size_[i] = first_size_[i] = last_size_[i] = 0;
          pos_inc_[i] = first_pos_inc_[i] = last_pos_inc_[i] = 0;
        }
    }

    IteratorCfg<iterRank> transposeorder(uint32_t new_order[]) const {
        // create a transposed Iterator, reordering the dimensions
        IteratorCfg<iterRank> iter;
        // change order of axes
        uint32_t c = 0;
        for (uint32_t axis = 0; axis < iterRank; axis++) {
          assert(new_order[axis] >= 0 && new_order[axis] < iterRank);
          // axis can only be selected once
          assert((c & (1 << new_order[axis])) == 0);
          c |= (1 << new_order[axis]);
          iter.order_[axis] = order_[axis] < 0 ? -1 : new_order[order_[axis]];
          iter.count_[axis] = count_[axis];
          iter.size_[axis] = size_[axis];
          iter.first_size_[axis] = first_size_[axis];
          iter.last_size_[axis] = last_size_[axis];
          iter.pos_inc_[axis] = pos_inc_[axis];
          iter.first_pos_inc_[axis] = first_pos_inc_[axis];
          iter.last_pos_inc_[axis] = last_pos_inc_[axis];
        }
        return iter;
    }

    int32_t get_order(unsigned dim) const {
      return order_[dim];
    }
    int32_t get_count(unsigned dim) const {
      return count_[dim];
    }
    int32_t get_first_inc(unsigned dim) const {
      return first_pos_inc_[dim];
    }
    int32_t get_inc(unsigned dim) const {
      return pos_inc_[dim];
    }
    int32_t get_last_inc(unsigned dim) const {
      return last_pos_inc_[dim];
    }
    uint32_t get_first_size(unsigned dim) const {
      return first_size_[dim];
    }
    uint32_t get_size(unsigned dim) const {
      return size_[dim];
    }
    uint32_t get_last_size(unsigned dim) const {
      return last_size_[dim];
    }

  private:
    int32_t order_[iterRank];
    int32_t count_[iterRank];
    int32_t pos_inc_[iterRank];
    int32_t first_pos_inc_[iterRank];
    int32_t last_pos_inc_[iterRank];
    uint32_t size_[iterRank];
    uint32_t first_size_[iterRank];
    uint32_t last_size_[iterRank];
};

template <typename buf_T, unsigned tensorRank, unsigned iterRank>
class TensorIterator {

  public:

    // Constructor that will configure the iterator for a single tile
    TensorIterator(Tensor<buf_T, tensorRank> tensor) : config_(tensor) {
      full_tensor_ = tensor;
      Reset();
    }

    // Constructor that will configure the iterator for a predefined config
    TensorIterator(Tensor<buf_T, tensorRank> tensor, IteratorCfg<iterRank> config) {
      full_tensor_ = tensor;
      config_ = config;
      Reset();
    }

    TensorIterator(const TensorIterator<NoBuffer, tensorRank, iterRank>& tensor_iterator) {
      Tensor<NoBuffer, tensorRank> tensor = tensor_iterator.get_tensor();
      full_tensor_ = Tensor<buf_T, tensorRank>(buf_T(), tensor);
      config_ = tensor_iterator.get_config();
      offset_ = tensor_iterator.get_offset();
      tensor_iterator.get_pos(pos_);
      tensor_iterator.get_tile_idx(tile_idx_);
    }

    /**
     * @brief constructor
     *
     * empty constructor
     */
    TensorIterator() : full_tensor_(), config_() {
      Reset();
    }

    mli_status Reset() {
      for (uint32_t i = 0; i < iterRank; ++i ) {
        pos_[i] = tile_idx_[i] = 0;
      }
      offset_ = 0;
      return MLI_STATUS_OK;
    }

    // This method will compute the minimal buffersize required to fit worst case num_tiles
    // It will return the number size in bytes needed for that buffer
    // it will also update the mem_strides and increment values
    uint32_t ShrinkBuffer(uint32_t num_tiles);

    // prints the current position, size and ptr
    void Print();

    // increments to the next tile
    bool Next() {
      bool done = false;
      for (unsigned r = 0; r < iterRank; ++r) {
        if (tile_idx_[r] == config_.get_count(r) - 1) { // Last iteration
          pos_[r] += config_.get_last_inc(r);
          offset_ += config_.get_last_inc(r)*full_tensor_.get_mem_stride(config_.get_order(r));
          tile_idx_[r] = 0;
          if (r == iterRank - 1) done = true;
        } else {
          if (tile_idx_[r] == 0) { // First iteration
            pos_[r] += config_.get_first_inc(r);
            offset_ += config_.get_first_inc(r)*full_tensor_.get_mem_stride(config_.get_order(r));
          } else { // Middle iteration
            pos_[r] += config_.get_inc(r);
            offset_ += config_.get_inc(r)*full_tensor_.get_mem_stride(config_.get_order(r));
          }
          // not at end of axis, break
          ++tile_idx_[r];
          break;
        }
      }
      return done;
    }

    // Return increments to the next tile (to increment outside the iterator)
    bool Next(int32_t* pos_inc) {
      bool done = false;
      for (unsigned r = 0; r < iterRank; ++r) {
        int32_t dim = config_.get_order(r);
        if (dim >= 0) {
          if (tile_idx_[r] == config_.get_count(r) - 1) { // Last iteration
            pos_inc[dim] = config_.get_last_inc(r);
            tile_idx_[r] = 0;
            if (r == iterRank - 1) done = true;
          } else {
            if (tile_idx_[r] == 0) { // First iteration
              pos_inc[dim] = config_.get_first_inc(r);
            } else { // Middle iteration
              pos_inc[dim] = config_.get_inc(r);
            }
            // not at end of axis, break
            ++tile_idx_[r];
            break;
          }
        }
      }
      return done;
    }

    // returns a tensor of the current tile
    Tensor<buf_T, tensorRank> GetSubTensor() {
      uint32_t pos[tensorRank];
      uint32_t copysize[tensorRank];
      unsigned r = 0;
      for (r = 0; r < iterRank; ++r) {
        int32_t dim = config_.get_order(r);
        if (dim >= 0) {
            pos[dim] = (uint32_t)pos_[r];
            if (tile_idx_[r] == config_.get_count(r) - 1) { // Last iteration
              copysize[dim] = config_.get_last_size(r);
            } else if (tile_idx_[r] == 0) { // First iteration
              copysize[dim] = config_.get_first_size(r);
            } else { // Middle iteration
              copysize[dim] = config_.get_size(r);
            }
        }
      }
      return full_tensor_.slice(pos, copysize);
    }

    template<typename T>
    T read(){
//      uint32_t pos[tensorRank];
//      for (unsigned r = 0; r < iterRank; ++r) {
//        pos[config_.get_order(r)] = (uint32_t)pos_[r];
//      }
//      return full_tensor_.template read<T>(full_tensor_.get_offset(pos));
      return full_tensor_.template read<T>(offset_);
    }

    template<typename T>
    void write(T data){
//      uint32_t pos[tensorRank];
//      for (unsigned r = 0; r < iterRank; ++r) {
//        pos[config_.get_order(r)] = (uint32_t)pos_[r];
//      }
//      full_tensor_.write(full_tensor_.get_offset(pos), data);
      full_tensor_.write(offset_, data);
    }

    uint32_t get_dim(uint32_t dim_idx) const {
      return full_tensor_.get_dim(dim_idx);
    }

    void get_full_shape(uint32_t shape[]) const {
      full_tensor_.get_dims(shape);
    }

    void get_mem_strides(int32_t mem_stride[]) const {
      full_tensor_.get_mem_strides(mem_stride);
    }

    uint32_t get_mem_stride(uint32_t dim_idx) const {
      return full_tensor_.get_mem_stride(dim_idx);
    }

    void set_buf(const buf_T& buf) {
      full_tensor_.set_buf(buf);
    }

    buf_T get_buf() {
      return full_tensor_.get_buf();
    }

    void set_config(const IteratorCfg<iterRank>& config) {
      config_ = config;
    }

    const IteratorCfg<iterRank>& get_config() const {
      return config_;
    }

    int32_t get_offset() const {
      return offset_;
    }

    void get_pos(int32_t pos[iterRank]) const {
      for (unsigned i = 0; i < iterRank; i++) {
        pos[i] = pos_[i];
      }
    }

    void get_tile_idx(int32_t tile_idx[iterRank]) const {
      for (unsigned i = 0; i < iterRank; i++) {
        tile_idx[i] = tile_idx_[i];
      }
    }

    Tensor<buf_T, tensorRank> get_tensor() const {
      return full_tensor_;
    }

  private:
    Tensor<buf_T, tensorRank> full_tensor_;
    IteratorCfg<iterRank> config_;
    int32_t offset_;
    int32_t pos_[iterRank];
    int32_t tile_idx_[iterRank];
};

} // namespace mli

#endif // _MLI_ITERATOR_HPP_
