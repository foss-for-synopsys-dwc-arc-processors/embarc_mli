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
#include "mli_service_functions.hpp"

namespace snps_arc::metaware::mli {

typedef enum : uint8_t {
  SZ0_SZ0_SZ0 = 0, // also used for SZ0 and SZ0_SZ0 cases
  SZ0_SZ0_SZ1 = 1, // also used for SZ0_SZ1 case
  SZ0_SZ1_SZ0 = 2,
  SZ0_SZ1_SZ1 = 3,
  SZ0_SZ1_SZ2 = 4
} TileLayoutCode;

/**
 * @brief 
 *
 *
 */
template <uint32_t iterRank>
class IteratorCfg {

 public:
  /**
   * @brief constructor
   *
   */
    IteratorCfg(const int32_t iteration_order[] = kDefaultIterOrder) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = iteration_order[i];
        m_count[i] = 1;
        m_size[i] = m_first_size[i] = m_last_size[i] = 1;
        m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        m_diff_code[i] = SZ0_SZ0_SZ0;
      }
      m_buf_tiles_num = 0;
    }
    
    // Constructor that will create config for a single tile for whole given Tensor (generic tensorRank, iterRank)
    template <typename buf_T, uint32_t tensorRank = iterRank>
    IteratorCfg(Tensor<buf_T, tensorRank> tensor) {
      static_assert(tensorRank <= iterRank);
      for (uint32_t i = 0; i < tensorRank; ++i) {
        m_order[i] = i;
        m_count[i] = 1;
        m_size[i] = m_first_size[i] = m_last_size[i] = tensor.get_dim(i);
        m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        m_diff_code[i] = SZ0_SZ0_SZ0;
      }
      for (uint32_t i = tensorRank; i < iterRank; ++i) {
        m_order[i] = i;
        m_count[i] = kSkipIterDim;
        m_size[i] = m_first_size[i] = m_last_size[i] = 0;
        m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        m_diff_code[i] = SZ0_SZ0_SZ0;
      }
      m_buf_tiles_num = 0;
    }

    // Constructor that will create config for a single tile for whole given Tensor (special case tensorRank = iterRank)
    template <typename buf_T>
    IteratorCfg(Tensor<buf_T, iterRank> tensor) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = i;
        m_count[i] = 1;
        m_size[i] = m_first_size[i] = m_last_size[i] = tensor.get_dim(i);
        m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        m_diff_code[i] = SZ0_SZ0_SZ0;
      }
      m_buf_tiles_num = 0;
    }

    // Old-style constructor for compatibility, to be removed
    IteratorCfg(int32_t increment[],
                uint32_t size[]
                ) {
      DEPRECATED_METHOD
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = i;
        m_count[i] = size[i]*increment[i] != 0 ? CEIL_DIV(size[i], increment[i]) : 1;
        m_size[i] = m_first_size[i] = m_last_size[i] = size[i];
        m_pos_inc[i] = m_first_pos_inc[i] = increment[i];
        m_last_pos_inc[i] = increment[i] * (1 - m_count[i]);
        m_diff_code[i] = SZ0_SZ0_SZ0;
      }
      m_buf_tiles_num = 0;
    }


    /**
     * @brief Construct a new Iterator Cfg object
     *
     * @param order           [I] The iteration order 
     * @param count           [I] Number of tiles in each dimention
     * @param first_increment [I] Increment per dimension in elements for the first increment on each dimension
     * @param increment       [I] Increment per dimension in elements 
     * @param last_incerement [I] Increment per dimension in elements for the last tile and also can be used to reset the dimention
     * @param first_size      [I] Size in each dimension for the first tile in that dimension
     * @param size            [I] Size in each dimension for the remaining tiles in that dimension
     * @param last_size       [I] Size in each dimension for the last tile in that dimension
     * @param tiles_num       [I] Number of tiles buffers
     */
    IteratorCfg(int32_t order[],           
                int32_t count[],           
                int32_t first_increment[],  
                int32_t increment[],        
                int32_t last_increment[],   
                int32_t first_size[],       
                int32_t size[],             
                int32_t last_size[],        
                uint32_t tiles_num = 0
                ) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = order[i];
        m_count[i] = count[i];
        m_first_pos_inc[i] = first_increment[i];
        m_pos_inc[i] = increment[i];
        m_last_pos_inc[i] = last_increment[i];
        m_first_size[i] = first_size[i];
        m_size[i] = size[i];
        m_last_size[i] = last_size[i];
        m_diff_code[i] = CalcDiffCode(m_count[i], m_first_size[i], m_size[i], m_last_size[i]);
      }
      m_buf_tiles_num = tiles_num;
    }


    // Constructor that will compute the number of tiles in each dimension, it will also compute the increment values and sizes.
    template <typename buf_T>
    IteratorCfg(const Tensor<buf_T, iterRank>& tensor, // full tensor to be iterated
                const uint32_t tilesize[iterRank],     // size of the tile
                const int32_t order[iterRank],         // iteration order
                uint32_t skew = 0) {                   // skewing on first iteration dimension (reduction of the first tile size, used in fused processing to avoid triple buffering)
      MLI_ASSERT(tilesize[order[0]] >= skew);
      for (uint32_t i = 0; i < iterRank; ++i, skew = 0) {
        int32_t dim = order[i];
        MLI_ASSERT(tilesize[dim] <= tensor.get_dim(dim));
        m_order[i] = dim;
        m_count[i] = CEIL_DIV(tensor.get_dim(dim) + skew, tilesize[dim]);
        m_first_size[i] = m_first_pos_inc[i] = tilesize[dim] - skew;
        m_size[i] = m_pos_inc[i] = tilesize[dim];
        m_last_size[i] = (tensor.get_dim(dim) + skew - 1) % tilesize[dim] + 1;
        m_last_pos_inc[i] = tilesize[dim] * (1 - m_count[i]) + skew;
        m_diff_code[i] = CalcDiffCode(m_count[i], m_first_size[i], m_size[i], m_last_size[i]);
      }
      m_buf_tiles_num = 0;
    }


    // Constructor that updates previously constructed iterator taking into account a kernel apperture and paddings
    template <typename buf_T>
    IteratorCfg(const IteratorCfg& icfg,                // origin config
                const Tensor<buf_T, iterRank> tensor,   // full tensor
                const uint32_t effective_kernel_size[], // used to calculate the overlaps between the tiles
                const uint32_t stride[],                // used to calculate the overlaps between the tiles
                const uint32_t pre_padding[]) {         // number of virtual pixels added before each dimension
      for (uint32_t i = 0; i < iterRank; ++i) {
        int32_t dim = icfg.m_order[i];
        uint32_t tns_dim = tensor.get_dim(dim);
        m_order[i] = dim;
        m_count[i] = icfg.m_count[i];
        m_pos_inc[i] = icfg.m_pos_inc[i] * stride[dim];
        /*
         *  TODO: extend for situations where m_pos_inc[i] < pre_padding[dim]
         *  For example in case of left padding = 4 and increment = 2 - first increment is correct = 0, but second is 2, but must be also 0.
         *  To fix it some kind of pad[iterRank] variable needed and some kind of code
         *  "m_first_pos_inc[i] = MAX(0, m_pos_inc[i] - (int32_t)pre_padding[dim]);" instead of assert
         */
        MLI_ASSERT(icfg.m_first_pos_inc[i] * stride[dim] >= pre_padding[dim]);
        m_first_pos_inc[i] = icfg.m_first_pos_inc[i] * stride[dim] - (int32_t)pre_padding[dim];
        m_last_pos_inc[i] = m_count[i] > 1 ? m_pos_inc[i] * (2 - m_count[i]) - m_first_pos_inc[i] : 0;
        m_size[i] = (icfg.m_size[i] - 1) * stride[dim] + effective_kernel_size[dim];
        m_first_size[i] = MIN((icfg.m_first_size[i] - 1) * stride[dim] + effective_kernel_size[dim] - pre_padding[dim], tns_dim);
        m_last_size[i] = tns_dim + m_last_pos_inc[i];
        m_diff_code[i] = CalcDiffCode(m_count[i], m_first_size[i], m_size[i], m_last_size[i]);
      }
      m_buf_tiles_num = icfg.get_buf_tiles_num();
    }

    template <uint32_t srcIterRank>
    IteratorCfg(const IteratorCfg<srcIterRank>& icfg, uint32_t axis_to_drop) {
      static_assert(iterRank == srcIterRank - 1);
      int j = 0;
      for (uint32_t i = 0; i < srcIterRank; ++i) {
        if (i == axis_to_drop) continue;

        m_order[j] = j; // icfg.get_order(i); TODO: make it work for any iter order
        m_count[j] = icfg.get_count(i);
        m_first_size[j] = icfg.get_first_size(i);
        m_size[j] = icfg.get_size(i);
        m_last_size[j] = icfg.get_last_size(i);
        m_pos_inc[j] = icfg.get_first_inc(i);
        m_first_pos_inc[j] = icfg.get_inc(i);
        m_last_pos_inc[j] = icfg.get_last_inc(i);
        m_diff_code[j] = icfg.get_diff_code(i);
        j++;
      }
      m_buf_tiles_num = icfg.get_buf_tiles_num();
    }

    // Constructor that updates previously constructed iterator for work on buffer for only one or several tiles
    IteratorCfg(const IteratorCfg& icfg, // origin config
                bool cyclic              // cyclic buffer, even for 1 tile
               ) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = icfg.m_order[i];
        m_count[i] = icfg.m_count[i];
        if (cyclic && i == 0) {
          m_pos_inc[0] = icfg.m_pos_inc[0];
          m_first_pos_inc[0] = icfg.m_first_pos_inc[0];
          m_last_pos_inc[0] = icfg.m_last_size[0];
        } else {
          m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        }
        m_size[i] = icfg.m_size[i];
        m_first_size[i] = icfg.m_first_size[i];
        m_last_size[i] = icfg.m_last_size[i];
        m_diff_code[i] = icfg.m_diff_code[i];
      }
      m_buf_tiles_num = icfg.get_buf_tiles_num();
    }

    // Method that updates the current iterator for work on buffer for only one or several tiles
    // Does the same as previous constructor but with the current config
    void ShrinkCfg(bool cyclic = false) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        if (cyclic && i == 0) {
          m_last_pos_inc[0] = m_last_size[0];
        } else {
          m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        }
      }
    }

    // This method will remove the overlap between adjacent tiles in the first iteration dimension to avoid duplicate data transfers
    // It will update the (first)size and (first)inc parameters
    void RemoveOverlapAdjacentTiles() {
      if (m_count[0] > 1) {
        int32_t overlap = m_size[0] - m_pos_inc[0];
        m_first_pos_inc[0] = m_first_size[0];
        m_size[0] = m_pos_inc[0];
        m_last_size[0] -= overlap;
        m_last_pos_inc[0] -= overlap;
        m_diff_code[0] = CalcDiffCode(m_count[0], m_first_size[0], m_size[0], m_last_size[0]);
      }
    }

    /**
     * @brief   Specific method to add extra increments for padding handling
     *
     * @param pre_padding [I] pre-padding values on tensor dimensions
     */
    template <uint32_t tensorRank>
    void ApplyPrePadding(const uint32_t pre_padding[tensorRank]) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        int32_t dim = m_order[i];
        if (dim >= 0 && m_count[i] > 1) {
          m_first_pos_inc[i] += pre_padding[dim];
          m_last_pos_inc[i] -= pre_padding[dim];
        }
      }
    }

    template <uint32_t tensorRank>
    void ApplyAlignsToSizes(const uint32_t aligns[tensorRank]) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        int32_t dim = m_order[i];
        if (dim >= 0 && m_count[i] > 1) {
          m_first_size[i] = CEIL_RND(m_first_size[i], aligns[dim]);
          m_last_size[i] = CEIL_RND(m_last_size[i], aligns[dim]);
          m_size[i] = CEIL_RND(m_size[i], aligns[dim]);
        }
      }
    }

    /**
     * @brief 
     * 
     * Method to convert IteratorCfg granularities along specific dim
     * 
     * @param tnsDim            [I] Tensor Dimension to apply Vectorization to it.
     * @param old_vector_size   [I] Old Vector Size in bytes.
     * @param new_vector_size   [I] New Vector Size in bytes.
     */
    void ConvertGran(int32_t tnsDim, int32_t old_vector_size, int32_t new_vector_size) {
      for (uint32_t i = 0; i < iterRank; ++i) if (m_order[i] == tnsDim) {
        m_first_size[i] = CEIL_DIV(m_first_size[i] * old_vector_size, new_vector_size);
        m_size[i] = CEIL_DIV(m_size[i] * old_vector_size, new_vector_size);
        m_last_size[i] = CEIL_DIV(m_last_size[i] * old_vector_size, new_vector_size);
        m_first_pos_inc[i] = CEIL_DIV(m_first_pos_inc[i] * old_vector_size, new_vector_size);
        m_pos_inc[i] = CEIL_DIV(m_pos_inc[i] * old_vector_size, new_vector_size);
        m_last_pos_inc[i] = CEIL_DIV(m_last_pos_inc[i] * old_vector_size, new_vector_size);
      }
    }

    /**
     * @brief set config from smaller rank configuration
     *
     * @param in       [I] IteratorCfg with a smaller rank that is stored inside this cfg
     */
    template <uint32_t N>
    void set_config(const IteratorCfg<N>& in) {
        assert(N <= iterRank);
        uint32_t i = 0;
        for (; i < N; ++i) {
          m_order[i] = in.m_order[i];
          m_count[i] = in.m_count[i];
          m_size[i] = in.m_size[i];
          m_first_size[i] = in.m_first_size[i];
          m_last_size[i] = in.m_last_size[i];
          m_pos_inc[i] = in.m_pos_inc[i];
          m_first_pos_inc[i] = in.m_first_pos_inc[i];
          m_last_pos_inc[i] = in.m_last_pos_inc[i];
          m_diff_code[i] = in.m_diff_code[i];
        }
        for (; i < iterRank; ++i) {
          m_count[i] = 1;
          m_order[i] = -1;
          m_size[i] = m_first_size[i] = m_last_size[i] = 0;
          m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
          m_diff_code[i] = SZ0_SZ0_SZ0;
        }
    }

    IteratorCfg<iterRank> transpose_order(uint32_t new_order[]) const {
        // create a transposed Iterator, reordering the dimensions
        IteratorCfg<iterRank> iter;
        // change order of axes
        uint32_t c = 0;
        for (uint32_t axis = 0; axis < iterRank; axis++) {
          assert(new_order[axis] >= 0 && new_order[axis] < iterRank);
          // axis can only be selected once
          assert((c & (1 << new_order[axis])) == 0);
          c |= (1 << new_order[axis]);
          iter.m_order[axis] = m_order[axis] < 0 ? -1 : new_order[m_order[axis]];
          iter.m_count[axis] = m_count[axis];
          iter.m_size[axis] = m_size[axis];
          iter.m_first_size[axis] = m_first_size[axis];
          iter.m_last_size[axis] = m_last_size[axis];
          iter.m_pos_inc[axis] = m_pos_inc[axis];
          iter.m_first_pos_inc[axis] = m_first_pos_inc[axis];
          iter.m_last_pos_inc[axis] = m_last_pos_inc[axis];
          iter.m_diff_code[axis] = m_diff_code[axis];
        }
        return iter;
    }

    void UpdateOrder(const int32_t iteration_order[iterRank]) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = iteration_order[i];
        if (iteration_order[i] == kSkipIterDim) {
            m_first_size[i] = 0;
            m_size[i] = 0;
            m_last_size[i] = 0;
            m_diff_code[i] = SZ0_SZ0_SZ0;
        }
      }
    }

    void SetZeroIncrements(const int32_t zero_inc_mask[iterRank]) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        if (zero_inc_mask[i]) {
          m_first_pos_inc[i] = 0;
          m_pos_inc[i] = 0;
          m_last_pos_inc[i] = 0;
        }
      }
    }

    template <typename buf_T, uint32_t rank>
    void DisableTiling(const int32_t disable_mask[rank],
                       const Tensor<buf_T, rank>& tensor,
                       const int32_t skip_dim[rank] = nullptr) {
      unsigned tensor_dim = 0;
      for (uint32_t iter_dim = 0; iter_dim < rank; ++iter_dim) {
        if (skip_dim != nullptr && skip_dim[iter_dim] == kSkipIterDim) continue;

        if (disable_mask[iter_dim]) {
          uint32_t size = tensor.get_dim(tensor_dim);
          m_first_size[iter_dim] = size;
          m_size[iter_dim] = size;
          m_last_size[iter_dim] = size;
          m_diff_code[iter_dim] = SZ0_SZ0_SZ0;
        }
        tensor_dim++;
      }
    }

    template <typename Buf_T, uint32_t rank>
    void OverrideIncrements(const int32_t inc_override[iterRank], const Tensor<Buf_T, rank>& tensor) {
      for (uint32_t i = 0; i < iterRank; i++) {
        if (inc_override[i] == kSkipIterDim) continue;

        m_first_pos_inc[i] = inc_override[i];
        m_pos_inc[i] = inc_override[i];
        m_size[i] = inc_override[i];
        m_first_size[i] = inc_override[i];
        m_last_pos_inc[i] = m_count[i] > 1 ? m_pos_inc[i] * (2 - m_count[i]) - m_first_pos_inc[i] : 0;
        m_last_size[i] = tensor.get_dim(i) + m_last_pos_inc[i];
        m_diff_code[i] = CalcDiffCode(m_count[i], m_first_size[i], m_size[i], m_last_size[i]);
      }
    }

    int32_t get_order(uint32_t dim) const {
      return m_order[dim];
    }

    int32_t get_count(uint32_t dim) const {
      return m_count[dim];
    }
    int32_t get_first_inc(uint32_t dim) const {
      return m_first_pos_inc[dim];
    }
    int32_t get_inc(uint32_t dim) const {
      return m_pos_inc[dim];
    }
    int32_t get_last_inc(uint32_t dim) const {
      return m_last_pos_inc[dim];
    }
    uint32_t get_first_size(uint32_t dim) const {
      return m_first_size[dim];
    }
    uint32_t get_size(uint32_t dim) const {
      return m_size[dim];
    }
    uint32_t get_last_size(uint32_t dim) const {
      return m_last_size[dim];
    }
    TileLayoutCode get_diff_code(uint32_t dim) const {
      return m_diff_code[dim];
    }
    uint32_t get_buf_tiles_num() const {
      return m_buf_tiles_num;
    }

    void SetCount(int32_t val, uint32_t dim) {
      m_count[dim] = val;
    }

    void set_count(int32_t* count){
      for (uint32_t axis = 0; axis < iterRank; axis++) {
        m_count[axis] = count[axis];
      }
    }

  private:
    /**
     * @brief Calculate tile-size difference code along certain dimension
     *
     * Output values:
     * 0 - (First), (middle) and last tiles have the same size: (SZ0), (SZ0, SZ0) or (SZ0, SZ0, SZ0)
     * 1 - First, (middle) and last tiles have 2 different sizes: (SZ0, SZ1) or (SZ0, SZ0, SZ1)
     * 2 - First, middle and last tiles have 2 different sizes: (SZ0, SZ1, SZ0)
     * 3 - First, middle and last tiles have 2 different sizes: (SZ0, SZ1, SZ1)
     * 4 - First, middle and last tiles have 3 different sizes: (SZ0, SZ1, SZ2)
     *
     */
    TileLayoutCode CalcDiffCode(int32_t count, uint32_t first_size, uint32_t size, uint32_t last_size) {
        TileLayoutCode diff_code = SZ0_SZ0_SZ0;
        if (count > 1) {
          if (first_size != last_size) diff_code = SZ0_SZ0_SZ1;
          if (count > 2 && first_size != size) {
            if (first_size == last_size) diff_code = SZ0_SZ1_SZ0;
            else if (size == last_size) diff_code = SZ0_SZ1_SZ1;
            else diff_code = SZ0_SZ1_SZ2;
          }
        }
        return diff_code;
    }
    static constexpr int32_t kDefaultIterOrder[] = {0, 1, 2, 3, 4};
    int32_t m_order[iterRank];
    int32_t m_count[iterRank];
    int32_t m_pos_inc[iterRank];
    int32_t m_first_pos_inc[iterRank];
    int32_t m_last_pos_inc[iterRank];
    uint32_t m_size[iterRank];
    uint32_t m_first_size[iterRank];
    uint32_t m_last_size[iterRank];
    TileLayoutCode m_diff_code[iterRank];
    uint32_t m_buf_tiles_num;
};

/**
 * @brief This class implements control of the buffer attached to the
 * TensorIterator
 *
 * @tparam buf_T Type of buffer
 * @tparam iterRank Maximum rank
 */
template <class buf_T, uint32_t iterRank>
class BufferIterator {
 public:
    /**
     * @brief Construct a new Buffer Iterator object
     *
     * @param full_tensor [I] Reference to Tensor object
     * @param cfg [I] Reference to IteratorCfg object
     */
    template <uint32_t tensorRank, class src_buf_T>
    BufferIterator(const Tensor<src_buf_T, tensorRank>& full_tensor,
                   const IteratorCfg<iterRank>& cfg)
        : m_buffer_offset{0} {
      m_buffer_size = 1;
      for (uint32_t i = 0; i < iterRank; i++) {
        int32_t axis = cfg.get_order(i);
        if (axis == kSkipIterDim) continue;

        uint32_t buffer_dim = MAX(cfg.get_size(axis), cfg.get_first_size(axis));
        // If tiles number equals to 0, that means attached buffer calculated for
        // full tensor, otherwise - for number of tiles. Also if dimension of the
        // tensor/tile is 1 or 0, that simply means that dimension doesn't exist
        // in terms of buffer, so offset must be 0.
        if (cfg.get_buf_tiles_num() == 0 && full_tensor.get_dim(i) > 1) {
          m_first_offset_inc[axis] = cfg.get_first_inc(axis) == 0 ? full_tensor.get_mem_stride(axis) : cfg.get_first_inc(axis) * full_tensor.get_mem_stride(axis);
          m_offset_inc[axis] = cfg.get_inc(axis) == 0 ? full_tensor.get_mem_stride(axis) : cfg.get_inc(axis) * full_tensor.get_mem_stride(axis);
          m_last_offset_inc[axis] = cfg.get_last_inc(axis) == 0 ? full_tensor.get_mem_stride(axis) : cfg.get_last_inc(axis) * full_tensor.get_mem_stride(axis);

        } else {
          m_first_offset_inc[axis] = 0;
          m_offset_inc[axis] = 0;
          m_last_offset_inc[axis] = 0;
          m_buffer_size *= buffer_dim;
        }
        m_buffer_dec[axis] = cfg.get_buf_tiles_num() == 0
                             ? full_tensor.get_dim(axis) * full_tensor.get_mem_stride(axis)
                             : buffer_dim * full_tensor.get_mem_stride(axis);
      }
    };

    /**
     * @brief Updating BufferIterator in case of Tensor or IteratorCfg changed
     *
     * @param full_tensor [I] Reference to Tensor object
     * @param cfg         [I] Reference to IteratorCfg object
     */
    void UpdateConfig(const Tensor<buf_T, iterRank>& full_tensor,
                      const IteratorCfg<iterRank>& cfg,
                      const uint32_t buffer_offset) {
      for (uint32_t i = 0; i < iterRank; i++) {
        m_first_offset_inc[i] = cfg.get_first_inc(i)>0 ? cfg.get_first_inc(i) * full_tensor.get_mem_stride(i) :full_tensor.get_mem_stride(i);
        m_offset_inc[i] = cfg.get_inc(i)>0 ? cfg.get_inc(i) * full_tensor.get_mem_stride(i) :full_tensor.get_mem_stride(i);
        m_last_offset_inc[i] = cfg.get_last_inc(i)>0 ? cfg.get_last_inc(i) * full_tensor.get_mem_stride(i) :full_tensor.get_mem_stride(i);
        m_buffer_dec[i] = full_tensor.get_dim(i) * full_tensor.get_mem_stride(i);
      }
      m_buffer_offset = buffer_offset;
    }

    /**
     * @brief Calculating offset in elements inside memory buffer based on
     * pre-calculated offset increments per dimension
     *
     * @param dim      [I] Dimension for incrementing
     * @param tile_pos [I] Using first offset from IteratorCfg or not
     */
    void Next(const uint32_t dim, kTilePos_t tile_pos) {
      if (tile_pos == kFirstTile) {
        m_buffer_offset += m_first_offset_inc[dim];
      } else if(tile_pos == kLastTile) {
        m_buffer_offset += m_last_offset_inc[dim];
      }
      else{
        m_buffer_offset += m_offset_inc[dim];
      }
      // if m_buffer_size = 1 then no tiling
      if (m_buffer_size > 1) {
        m_buffer_offset %= m_buffer_size;
      }
    }

    /**
     * @brief Fixing offset in elements inside memory buffer when reaching the end
     * of dimension
     *
     * @param dim [IN] Dimension where end is reached
     */
    void EndOfDim(uint32_t dim) {
      if (m_buffer_dec[dim] <= m_buffer_offset) {
        m_buffer_offset -= m_buffer_dec[dim];
      }
    }

    /**
     * @brief Get the offset in elements inside memory buffer
     *
     * @return Offset in elements inside attached memory buffer considering buffer
     * cycling
     */
    uint32_t GetBufferOffset() {
      return m_buffer_offset;
    }

    uint32_t GetLastInc(uint32_t axis){
      return m_last_offset_inc[axis];
    }
    

 private:
  int32_t m_first_offset_inc[iterRank];     /**< Buffer offset increment per
                                                 dimension for the first tile */
  int32_t m_offset_inc[iterRank];           /**< Buffer offset increment per dimension for
                                                 the rest tiles */
  int32_t m_last_offset_inc[iterRank];      /**< Buffer offset increment per dimension for
                                                 the last tile */
  int32_t m_buffer_dec[iterRank];           /**< Buffer offset decrement per dimension */

  int32_t m_buffer_offset;                  /**< Offset in elements inside attached memory
                                                 buffer */
  int32_t m_buffer_size;

};

/**
 * @brief 
 *
 *
 */
template <typename buf_T, uint32_t tensorRank, uint32_t iterRank>
class TensorIterator {

  public:

    /**
      * @brief constructor
      *
      * @param tensor [I] full tensor used to create 1 tile equal tensor size
      *
      */
    TensorIterator(Tensor<buf_T, tensorRank> tensor)
      : m_full_tensor(tensor),
        m_config(tensor),
        m_buffer_itr(m_full_tensor, m_config) {
      UpdateMemstrides();
      Reset();
    }


    /**
      * @brief constructor
      *
      *
      * @param tensor [I] Full tensor which is tiled by this iterator
      * @param config [I] Iterator configuration
      */
    TensorIterator(Tensor<buf_T, tensorRank> tensor, IteratorCfg<iterRank> config)
        : m_full_tensor(tensor),
          m_config(config),
          m_buffer_itr(m_full_tensor, m_config) {
      UpdateMemstrides();
      Reset();
    }

    /**
       * @brief constructor
       *
       * This constuctor is appropriate for output tiling, for input tiling better to use
       * special constructor with effective_kernel_size parameter
       * @param tensor    [I] Full tensor which is tiled by this iterator
       * @param tile_size [I] Size of tile
       * @param order     [I] Iteration order
       */
    TensorIterator(Tensor<buf_T, tensorRank> tensor, uint32_t tile_size[], const int32_t order[])
      : m_full_tensor(tensor),
      m_config(tensor, tile_size, order),
      m_buffer_itr(m_full_tensor, m_config) {
      UpdateMemstrides();
      Reset();
    }

    /**
      * @brief constructor
      *
      * Simple TenosorIterator with default TensorIterator config
      * @param tensor           [I] Full tensor which is tiled by this TensorIterator
      * @param offset           [I] Current offset used in TensorIterator
      * @param iteration_order  [I] The iteration order used for increments
      */
    TensorIterator(const Tensor<buf_T, tensorRank>& tensor,
                   const uint32_t buffer_offset,
                   const int32_t* iteration_order)
        : m_full_tensor(tensor),
          m_config(iteration_order),
          m_buffer_itr(m_full_tensor, m_config){
      UpdateMemstrides();
      Reset();
      m_buffer_itr.UpdateConfig(m_full_tensor, m_config,buffer_offset);
    }


     /**
     * @brief constructor
     *
     * TensorIterator constructor using parameters of another TensorIterator
     * @param tensor_iterator [I] TensorIterator refrence to use its parameters in generate new TensorIterator
     */
    TensorIterator(const TensorIterator<NoBuffer, tensorRank, iterRank>& tensor_iterator)
        : m_buffer_itr(tensor_iterator.get_tensor(), tensor_iterator.get_config()) {
      Tensor<NoBuffer, tensorRank> tensor = tensor_iterator.get_tensor();
      m_full_tensor = Tensor<buf_T, tensorRank>(buf_T(), tensor);
      m_config = tensor_iterator.get_config();
      m_offset = tensor_iterator.get_offset();
      tensor_iterator.get_pos(m_pos);
      tensor_iterator.get_tile_idx(m_tile_idx);
    }

    TensorIterator(
        const TensorIterator<OffsetBuffer, tensorRank, iterRank> &tns_iter,
        uint64_t bases[],
        unsigned num_mem) 
        : m_full_tensor(tns_iter.get_tensor(), bases, num_mem),
          m_config(tns_iter.get_config()),
          m_buffer_itr(m_full_tensor, m_config) {
      m_offset = tns_iter.get_offset();
      tns_iter.get_pos(m_pos);
      tns_iter.get_tile_idx(m_tile_idx);
    }

    /**
      * @brief constructor
      * 
      * Constructor that will compute the number of tiles in each dimension, it will also compute the increment values and sizes.
      * This constructor takes into account effective kernel size and paddings.
      * Appropriate usage of this constructor is following: create TensorIterator for output tiling of conv-like kernels
      * and use this constructor to create input tiling of same conv-like kernel.
      * 
      * @param tensor                [I] Full tensor which is tiled by this iterator
      * @param tensor_iterator       [I] Base tensor iterator that defines output tiling
      * @param effective_kernel_size [I] effective kernel size (used to calculate the overlaps between the tiles)
      * @param stride                [I] strides (used to calculate the overlaps between the tiles)
      * @param pre_padding           [I] size of padding before each dimension
      */
    TensorIterator(const Tensor<buf_T, tensorRank>& tensor,
                   TensorIterator& tensor_iterator,
                   uint32_t effective_kernel_size[],
                   uint32_t stride[],
                   uint32_t pre_padding[])
      : m_full_tensor(tensor),
        m_config(tensor_iterator.get_config(), tensor, effective_kernel_size, stride, pre_padding),
        m_buffer_itr(m_full_tensor, m_config) {
      UpdateMemstrides();
      Reset();
    }

    /**
     * @brief constructor
     * 
     * Appropriate usage of this constructor is following : create IteratorCfg icfg for output tiling of conv - like kernels
     * and use this constructor to create weights or weights zero points tiling of same conv-like kernel.
     * 
     * @param tensor                     [I] Full tensor which is tiled by this iterator
     * @param tensor_iterator            [I] Base tensor iterator that defines output tiling
     * @param iteration_order            [I] for conv zps it's { -1, -1, -1, -1, 0 } to leave only Co increments, and nullptr for conv
     * @param zero_increment_mask        [I] for conv and conv zps it's {1, 1, 1, 1, 0} to make zero increments in first 4 dimensions
     */
    template<uint32_t srcTensorRank>
    TensorIterator(const Tensor<buf_T, tensorRank>& tensor,
                   TensorIterator<buf_T, srcTensorRank, iterRank>& tensor_iterator,
                   const int32_t iteration_order[iterRank] = nullptr,
                   const int32_t zero_increment_mask[iterRank] = nullptr)
      : m_full_tensor(tensor),
        m_config(tensor_iterator.get_config()),
        m_buffer_itr(tensor, m_config){

      if (zero_increment_mask != nullptr) {
        m_config.SetZeroIncrements(zero_increment_mask);
        if (srcTensorRank == iterRank) {
          MLI_ASSERT(tensorRank == iterRank || (tensorRank < iterRank && iteration_order != nullptr) );
          m_config.template DisableTiling<buf_T, tensorRank>(zero_increment_mask, tensor, iteration_order);
        }
      }
      if (iteration_order != nullptr) m_config.UpdateOrder(iteration_order);

      UpdateMemstrides();
      Reset();
    }

    /**
     * @brief constructor
     *
     * empty constructor
     */
    TensorIterator()
        : m_full_tensor(), m_config(), m_buffer_itr(m_full_tensor, m_config) {
      UpdateMemstrides();      
      Reset();
    }

    /**
     * @brief constructor
     *
     * Iterator constructor using parameters of another Iterator and tensor size
     * @param tensor_iterator [I] Iterator refrence to use its parameters in generate new iterator
     * @param Size [I] Tensor Size
     */
    TensorIterator(TensorIterator& tensor_iterator, const uint32_t size[])
        : m_config() {
      m_full_tensor = tensor_iterator.m_full_tensor.slice(size);
      m_buffer_itr = BufferIterator<buf_T, iterRank>(m_full_tensor, m_config);
      UpdateMemstrides();
      Reset();
      m_offset = tensor_iterator.m_offset;
    }

    /**
     * @brief   This method will shrink the buffer, to fit only N tiles
     *          in case of tiling. It also updates the iterator config.
     *
     * @param nTiles [I] Number of tiles fit in the shrinked buffer (usually 1 or 2)
     * @param cyclic [I] Should the shrinked buffer be cyclic
     * @param aligns [I] Alignments of the writer to this buffer (alignments of the reader doesn't matter)
     */
    uint32_t ShrinkBuffer(uint32_t nTiles,                   // Number of tiles fit in the shrinked buffer (usually 1 or 2)
                          bool cyclic,                       // Should the shrinked buffer be cyclic
                          const uint32_t aligns[tensorRank]  // Alignments of the writer to this buffer (alignments of the reader doesn't matter)
                         ) {
      uint32_t dims[tensorRank];
      m_full_tensor.get_dims(dims);
      for (uint32_t i = 0; i < iterRank; ++i) if (m_config.get_order(i) >= 0) {
        uint32_t dimsz = m_config.get_last_size(i);
        if (m_config.get_count(i) > 1) {
          dimsz = MAX(dimsz, m_config.get_first_size(i));
          if (m_config.get_count(i) > 2) {
            dimsz = MAX(dimsz, m_config.get_size(i));
          }
        }
        uint32_t align = aligns[m_config.get_order(i)];
        if (cyclic && i == 0) {
          if (nTiles > 1) {
            if (m_config.get_count(0) == 1) {
              dimsz *= nTiles;
            } else if (m_config.get_count(0) == 2) {
              dimsz = m_config.get_last_size(0) + m_config.get_first_size(0);
              for (uint32_t j = 2; j < nTiles; ++j) dimsz += static_cast<uint32_t>(m_config.get_inc(0)); // ???? Rare case
            } else {
              dimsz += MAX(m_config.get_first_size(0), static_cast<uint32_t>(m_config.get_inc(0)));
              for (uint32_t j = 2; j < nTiles; ++j) dimsz += static_cast<uint32_t>(m_config.get_inc(0));
            }
          }
          uint32_t overlap = m_config.get_size(0) - static_cast<uint32_t>(m_config.get_inc(0));
          dims[m_config.get_order(0)] = MAX(dimsz, align * nTiles + overlap);
        } else {
          dims[m_config.get_order(i)] = CEIL_RND(dimsz, align);
        }
        if (!cyclic) dims[0] *= nTiles;
      }
      Tensor<buf_T, tensorRank> new_tensor(m_full_tensor.get_buf(), dims, tensorRank);
      for (uint32_t i = 0; i < tensorRank; ++i) m_full_tensor.set_mem_stride(i, new_tensor.get_mem_stride(i));
      m_config.ShrinkCfg(cyclic);
      return new_tensor.get_total_elem_num();
    }

    void RemoveOverlapAdjacentTiles() {
      m_config.RemoveOverlapAdjacentTiles();
    }

    void ApplyPrePadding(const uint32_t pre_padding[tensorRank]) {
      m_config.template ApplyPrePadding<tensorRank>(pre_padding);
    }

    void ApplyAlignsToSizes(const uint32_t aligns[tensorRank]) {
      m_config.template ApplyAlignsToSizes<tensorRank>(aligns);
    }

    /**
     * @brief 
     * 
     * Method to convert TensorIterator granularities along inner most dimension
     * 
     * @param new_vector_size   [I] New Vector Size in bytes.
     * @param reverse_order     [I] if false Vectorization is applied on the last dim, otherwise on first dim.
     */
    void ConvertGran(int new_vector_size, bool reverse_order = false) {
      int tns_dim = reverse_order ? 0 : get_rank() - 1;
      m_config.ConvertGran(tns_dim, get_elem_size(), new_vector_size);
      m_full_tensor.ConvertGran(new_vector_size, reverse_order);
    }

    /**
     * @brief Reset the iterator to zero position
     *
     * This method will reset the internal position to zero
     */
    mli_status Reset() {
      for (uint32_t i = 0; i < iterRank; i++) {
        m_pos[i] = m_tile_idx[i] = 0;
      }
      m_offset = m_buffer_itr.GetBufferOffset();
      return MLI_STATUS_OK;
    }

    // prints the current position, size and ptr
    void Print();

    bool Next() {
      bool done = false;
      for (uint32_t r = 0; r < iterRank; ++r) {
        if (m_tile_idx[r] == m_config.get_count(r) - 1) { // Last iteration
          m_pos[r] = 0/*+= m_config.get_last_inc(r)*/; // Temporary fix, need to debug transpose conv
          //m_offset += m_config.get_last_inc(r)*m_full_tensor.get_mem_stride(m_config.get_order(r));
          m_buffer_itr.Next(r, kLastTile);
          m_buffer_itr.EndOfDim(r);
          m_tile_idx[r] = 0;
          if (r == iterRank - 1) done = true;
        } else {
          if (m_tile_idx[r] == 0) { // First iteration
            m_pos[r] += m_config.get_first_inc(r);
            //m_offset += m_config.get_first_inc(r)*m_full_tensor.get_mem_stride(m_config.get_order(r));
            m_buffer_itr.Next(r, kFirstTile);
          } else { // Middle iteration
            m_pos[r] += m_config.get_inc(r);
            //m_offset += m_config_.get_inc(r)*m_full_tensor.get_mem_stride(m_config.get_order(r));
            m_buffer_itr.Next(r, kMiddleTile);
          }
          // not at end of axis, break
          ++m_tile_idx[r];
          break;
        }
      }
      //uint32_t bufsz = m_full_tensor.get_buf().get_size();
      //while (m_offset < 0) m_offset += bufsz;
      //while (m_offset >= bufsz) m_offset -= bufsz;
      m_offset = m_buffer_itr.GetBufferOffset();
      return done;
    }

    // Return offset increment to the next tile (to increment outside the iterator)
    bool Next(int32_t& offset_inc) {
      bool done = false;
      offset_inc = 0;
      for (uint32_t r = 0; r < iterRank; ++r) {
        int32_t dim = m_config.get_order(r);
        if (dim >= 0) {
          uint32_t mem_stride = m_full_tensor.get_mem_stride(dim);
          if (m_tile_idx[r] == m_config.get_count(r) - 1) { // Last iteration
            offset_inc += m_config.get_last_inc(r)*mem_stride;
            m_tile_idx[r] = 0;
            if (r == iterRank - 1) done = true;
          } else {
            if (m_tile_idx[r] == 0) { // First iteration
              offset_inc += m_config.get_first_inc(r)*mem_stride;
            } else { // Middle iteration
              offset_inc += m_config.get_inc(r)*mem_stride;
            }
            // not at end of axis, break
            ++m_tile_idx[r];
            break;
          }
        }
      }
      return done;
    }

    // Return position increments to the next tile (to increment outside the iterator)
    bool Next(int32_t pos_inc[tensorRank]) {
      bool done = false;
      for (uint32_t r = 0; r < tensorRank; ++r) pos_inc[r] = 0;
      for (uint32_t r = 0; r < iterRank; ++r) {
        int32_t dim = m_config.get_order(r);
        if (dim >= 0) {
          if (m_tile_idx[r] == m_config.get_count(r) - 1) { // Last iteration
            pos_inc[dim] = m_config.get_last_inc(r);
            m_tile_idx[r] = 0;
            if (r == iterRank - 1) done = true;
          } else {
            if (m_tile_idx[r] == 0) { // First iteration
              pos_inc[dim] = m_config.get_first_inc(r);
            } else { // Middle iteration
              pos_inc[dim] = m_config.get_inc(r);
            }
            // not at end of axis, break
            ++m_tile_idx[r];
            break;
          }
        }
      }
      return done;
    }


    uint32_t ComputeTiledMaxBufferSize(){
      uint32_t tile_buf_shape[tensorRank];
      uint32_t buf_mem_stride[tensorRank];
      if (m_config.get_buf_tiles_num() == 0) {
        for (uint32_t dim = 0; dim < tensorRank; dim++) {
          tile_buf_shape[dim] = m_full_tensor.get_dim(dim);
          buf_mem_stride[dim] = m_full_tensor.get_mem_stride(dim);
        }
      } 
      else {
        for (uint32_t dim = 0; dim < tensorRank; dim++) {
          tile_buf_shape[dim] =
              MAX(m_config.get_first_size(dim), m_config.get_size(dim));
          buf_mem_stride[dim] = m_full_tensor.get_mem_stride(dim);
        }
      }
        return service::GetBufferSize(m_full_tensor.get_rank(), tile_buf_shape,
                                             buf_mem_stride);
    }


    /**
     * @brief Returns the subtensor of the full tensor in current iteration position
     *
     */
    Tensor<buf_T, tensorRank> GetSubTensor() {
      uint32_t pos[tensorRank];
      uint32_t copysize[tensorRank];
      // If the iterRank is less than the tensorRank we need to return the full size of non-iterable dimension
      for (uint32_t r = 0; r < tensorRank; ++r) pos[r] = 0, copysize[r] = m_full_tensor.get_dim(r);
      for (uint32_t r = 0; r < iterRank; ++r) {
        int32_t dim = m_config.get_order(r);
        if (dim == kSkipIterDim) continue;
        pos[dim] = (uint32_t)m_pos[r];
        if (m_tile_idx[r] == m_config.get_count(r) - 1) { // Last iteration
          copysize[dim] = m_config.get_last_size(r);
        } else if (m_tile_idx[r] == 0) { // First iteration
          copysize[dim] = m_config.get_first_size(r);
        } else { // Middle iteration
          // Clip here, since the penultimate tile can have less size due to big post-padding
          // Clipping of the first/last tiles is handled in the constructor
          copysize[dim] = MIN(m_full_tensor.get_dim(dim) - pos[dim], m_config.get_size(r));
        }
      }
      return m_full_tensor.slice(pos, copysize);
    }

    TensorIterator<buf_T, tensorRank, iterRank> GetSubTensorIterator() {
      uint32_t current_size[tensorRank];

      for (uint32_t i = 0; i < tensorRank; i++) {
        uint32_t tile_size = 0;
        if (m_tile_idx[i] == m_config.get_count(i) - 1) { // Last iteration
          tile_size = m_config.get_last_size(i);
        } else if (m_tile_idx[i] == 0) { // First iteration
          tile_size = m_config.get_first_size(i);
        } else { // Middle iteration
          tile_size = m_config.get_size(i);
        }
        current_size[i] =
            m_full_tensor.get_dim(i) - m_pos[i] < tile_size
                ? m_full_tensor.get_dim(i) - m_pos[i]
                : tile_size;
      }
      int32_t order[iterRank] = {0};
      for (uint32_t i = 0; i < iterRank; i++) {
        order[i] = m_config.get_order(i);
      }
      TensorIterator<buf_T, tensorRank, iterRank> sub_tensor_iterator(
          m_full_tensor.slice(current_size),m_buffer_itr.GetBufferOffset(),
          order);
          sub_tensor_iterator.m_offset = m_buffer_itr.GetBufferOffset();
      return sub_tensor_iterator;
    }

    uint32_t get_elem_size() {
      return m_full_tensor.get_elem_size();
    }

    void set_elem_size(uint32_t elem_size) {
      m_full_tensor.set_elem_size(elem_size);
    }

    TensorIterator<buf_T, tensorRank, iterRank> transpose(uint32_t new_order[]) const {
      return TensorIterator(m_full_tensor.transpose(new_order), m_config.transpose_order(new_order));
    }

    template<typename T>
    T read() {
      return m_full_tensor.template read<T>(m_offset);
    }

    template<typename T>
    void write(T data) {
      m_full_tensor.write(m_offset, data);
    }

    /**
     * @brief   This method will update the tensor memory strides based on tile size
     *          in case of tiling 
     *
     */
    void UpdateMemstrides(void) {
      if (m_config.get_buf_tiles_num() != 0) {
        uint32_t mem_stride=1;
        for (uint32_t i = iterRank; i > 0; --i)
        {
          if(m_config.get_size(m_config.get_order(i-1)) == 0) continue;
          m_full_tensor.set_mem_stride(i-1, mem_stride);
          mem_stride *= m_config.get_size(m_config.get_order(i-1));
        }        
      }
    }
    uint32_t get_dim(uint32_t dim_idx) const {
      return m_full_tensor.get_dim(dim_idx);
    }

    void get_full_shape(uint32_t shape[]) const {
      m_full_tensor.get_dims(shape);
    }

    void get_mem_strides(int32_t mem_stride[]) const {
      m_full_tensor.get_mem_strides(mem_stride);
    }

    uint32_t get_mem_stride(uint32_t dim_idx) const {
      return m_full_tensor.get_mem_stride(dim_idx);
    }

    void set_buf(const buf_T& buf) {
      m_full_tensor.set_buf(buf);
    }

    buf_T get_buf() {
      return m_full_tensor.get_buf();
    }

    void set_config(const IteratorCfg<iterRank>& config) {
      m_config = config;
    }

    const IteratorCfg<iterRank>& get_config() const {
      return m_config;
    }
    
    void SetCount(int32_t * count){
      m_config.set_count(count);
    }

    int32_t GetTensorShape(int32_t dim){
      return m_full_tensor.get_dim(dim);
    }

    int32_t get_offset() const {
      return m_offset;
    }

    void get_pos(int32_t pos[iterRank]) const {
      for (uint32_t i = 0; i < iterRank; i++) {
        pos[i] = m_pos[i];
      }
    }

    int32_t GetPos(uint32_t dim) const {
      return m_pos[dim];
    }

    void get_tile_idx(int32_t tile_idx[iterRank]) const {
      for (uint32_t i = 0; i < iterRank; i++) {
        tile_idx[i] = m_tile_idx[i];
      }
    }

    bool is_first_tile(uint32_t dim_idx) const {
      return !m_tile_idx[dim_idx];
    }

    bool is_last_tile(uint32_t dim_idx) const {
      return m_tile_idx[dim_idx] == (m_config.get_count(dim_idx) - 1);
    }

    void SetCount(int32_t val, uint32_t dim) {
       m_config.SetCount(val, dim);
    }

    void OverrideIncrements(const int32_t inc_override[iterRank]) {
      m_config.template OverrideIncrements<buf_T, tensorRank>(inc_override, m_full_tensor);
    }

    uint32_t GetTotalCount() const {
      int32_t cnt = m_config.get_count(0);
      for (uint32_t i = 1; i < iterRank; i++) {
        cnt *= m_config.get_count(i);
      }
      MLI_ASSERT(cnt > 0);
      return (uint32_t) cnt;
    }

    Tensor<buf_T, tensorRank> get_tensor() const {
      return m_full_tensor;
    }
     
    uint32_t get_rank() const {
      return m_full_tensor.get_rank();
    }

    uint32_t get_update_idx() const {
      uint32_t upd_idx = 0;
      uint32_t mul = 1;
      for (uint32_t i = 0; i < iterRank; ++i) {
        TileLayoutCode diff_code = m_config.get_diff_code(i);
        int32_t count = m_config.get_count(i);
        uint32_t modifier_tile_idx = 0;
        uint32_t num_diff = 2;
        switch (diff_code) {
          case SZ0_SZ0_SZ1:
            if (m_tile_idx[i] == count - 1) modifier_tile_idx = 1;
            break;
          case SZ0_SZ1_SZ0:
            if (m_tile_idx[i] > 0 && m_tile_idx[i] < count - 1) modifier_tile_idx = 1;
            break;
          case SZ0_SZ1_SZ1:
            if (m_tile_idx[i] > 0) modifier_tile_idx = 1;
            break;
          case SZ0_SZ1_SZ2:
            num_diff = 3;
            if (m_tile_idx[i] == count - 1) modifier_tile_idx = 2;
            else if (m_tile_idx[i] > 0) modifier_tile_idx = 1;
            break;
          default: // SZ0_SZ0_SZ0
            num_diff = 1;
        };
        upd_idx += modifier_tile_idx * mul;
        mul *= num_diff;
      }
      return upd_idx;
    }

  private:
    int32_t m_pos[iterRank];               /**< position in the full tensor */
    int32_t m_offset;            /**< offset in elements */
    Tensor<buf_T, tensorRank> m_full_tensor;
    IteratorCfg<iterRank> m_config;
    BufferIterator<buf_T, iterRank> m_buffer_itr;
    int32_t m_tile_idx[iterRank];
};

} // namespace mli

#endif // _MLI_ITERATOR_HPP_
