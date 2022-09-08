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
      }
      m_buf_tiles_num = 0;
    }
    
    // Constructor that will create config for a single tile for whole given Tensor
    template <typename buf_T>
    IteratorCfg(Tensor<buf_T, iterRank> tensor) {
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = i;
        m_count[i] = 1;
        m_size[i] = m_first_size[i] = m_last_size[i] = tensor.get_dim(i);
        m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
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
      }
      m_buf_tiles_num = tiles_num;
    }

    // Constructor that will compute the number of tiles in each dimension, it will also compute the increment values and sizes.
    template <typename buf_T>
    IteratorCfg(const Tensor<buf_T, iterRank>& tensor,        // full tensor to be iterated
                uint32_t tilesize[],                          // size of the tile
                const int32_t order[]) {                      // iteration order
      for (uint32_t i = 0; i < iterRank; ++i) {
        int32_t dim = order[i];
        m_order[i] = dim;
        m_count[i] = CEIL_DIV(tensor.get_dim(dim), tilesize[dim]);
        m_first_size[i] = m_size[i] = tilesize[dim];
        m_last_size[i] = (tensor.get_dim(dim) - 1) % tilesize[dim] + 1;
        m_first_pos_inc[i] = m_pos_inc[i] = tilesize[dim];
        m_last_pos_inc[i] = tilesize[dim] * (1 - m_count[i]);
      }
      m_buf_tiles_num = 0;
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
        int32_t dim = icfg.m_order[i];
        m_order[i] = dim;
        m_count[i] = icfg.m_count[i];
        m_pos_inc[i] = icfg.m_pos_inc[i] * stride[dim];
        m_first_pos_inc[i] = m_pos_inc[i] - pre_padding[dim];
        m_last_pos_inc[i] = m_count[i] > 1 ? m_pos_inc[i] * (1 - m_count[i]) + pre_padding[dim] : 0;
        m_size[i] = (icfg.m_size[i] - 1) * stride[dim] + effective_kernel_size[dim];
        m_first_size[i] = m_size[i] - pre_padding[dim];
        m_last_size[i] = tensor.get_dim(dim) + m_last_pos_inc[i];
      }
      m_buf_tiles_num = icfg.get_buf_tiles_num();
    }


    // Constructor that updates previously constructed iterator for work on buffer for only one or several tiles
    IteratorCfg(const IteratorCfg& icfg, // origin config
                int32_t shrinksz         // 1 or more
               ) {
      assert(shrinksz > 0);
      for (uint32_t i = 0; i < iterRank; ++i) {
        m_order[i] = icfg.m_order[i];
        m_count[i] = icfg.m_count[i];
        if (shrinksz > 1) { // cyclic buffer for double- (or more) buffering
          if (i == 0) {
            m_pos_inc[0] = icfg.m_pos_inc[0];
            m_first_pos_inc[0] = icfg.m_first_pos_inc[0];
            m_last_pos_inc[0] = icfg.m_last_size[0];
          } else {
            m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
          }
        } else { // buffer for single tile
          m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
        }
        m_size[i] = icfg.m_size[i];
        m_first_size[i] = icfg.m_first_size[i];
        m_last_size[i] = icfg.m_last_size[i];
      }
      m_buf_tiles_num = icfg.get_buf_tiles_num();
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
        }
        for (; i < iterRank; ++i) {
          m_count[i] = 1;
          m_order[i] = -1;
          m_size[i] = m_first_size[i] = m_last_size[i] = 0;
          m_pos_inc[i] = m_first_pos_inc[i] = m_last_pos_inc[i] = 0;
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
        }
        return iter;
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

    uint32_t get_buf_tiles_num() const {
      return m_buf_tiles_num;
    }

    void set_count(int32_t* count){
      for (uint32_t axis = 0; axis < iterRank; axis++) {
        m_count[axis] = count[axis];
      }
    }

  private:
    static constexpr int32_t kDefaultIterOrder[] = {0, 1, 2, 3, 4};
    int32_t m_order[iterRank];
    int32_t m_count[iterRank];
    int32_t m_pos_inc[iterRank];
    int32_t m_first_pos_inc[iterRank];
    int32_t m_last_pos_inc[iterRank];
    uint32_t m_size[iterRank];
    uint32_t m_first_size[iterRank];
    uint32_t m_last_size[iterRank];
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
    BufferIterator(Tensor<buf_T, iterRank>& full_tensor,
                  const IteratorCfg<iterRank>& cfg)
        : m_buffer_offset{0} {
      for (uint32_t i = 0; i < iterRank; i++) {
        uint32_t buffer_dim = MAX(cfg.get_size(i), cfg.get_first_size(i));

        // If tiles number equals to 0, that means attached buffer calculated for
        // full tensor, otherwise - for number of tiles. Also if dimension of the
        // tensor/tile is 1 or 0, that simply means that dimension doesn't exist
        // in terms of buffer, so offset must be 0.
        if (cfg.get_buf_tiles_num() == 0 && full_tensor.get_dim(i) > 1) {
          m_first_offset_inc[cfg.get_order(i)] = cfg.get_first_inc(cfg.get_order(i)) * full_tensor.get_mem_stride(cfg.get_order(i));
          m_offset_inc[cfg.get_order(i)] = cfg.get_inc(cfg.get_order(i)) * full_tensor.get_mem_stride(cfg.get_order(i));
          m_last_offset_inc[cfg.get_order(i)] = cfg.get_last_inc(cfg.get_order(i)) * full_tensor.get_mem_stride(cfg.get_order(i));

        } else {
          m_first_offset_inc[cfg.get_order(i)] = 0;
          m_offset_inc[cfg.get_order(i)] = 0;
          m_last_offset_inc[cfg.get_order(i)] = 0;
        }

        m_buffer_dec[cfg.get_order(i)] = cfg.get_buf_tiles_num() == 0
                              ? full_tensor.get_dim(cfg.get_order(i)) * full_tensor.get_mem_stride(cfg.get_order(i))
                              : buffer_dim * full_tensor.get_mem_stride(cfg.get_order(cfg.get_order(i)));
      }
    };

    /**
     * @brief Updating BufferIterator in case of Tensor or IteratorCfg changed
     *
     * @param full_tensor [I] Reference to Tensor object
     * @param cfg [I] Reference to IteratorCfg object
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

    // void shrink_buffer_offsets_update(Tensor<buf_T, iterRank>& full_tensor,const IteratorCfg<iterRank>& cfg){
    //   uint32_t iteration_dim = cfg.get_order(0);
    //   m_buffer_dec[iteration_dim] *= cfg.get_buf_tiles_num();
    //   if (iteration_dim > 0){
    //       full_tensor.set_mem_stride((iteration_dim - 1), full_tensor.get_mem_stride((iteration_dim - 1)*cfg.get_buf_tiles_num()));
    //       m_first_offset_inc[iteration_dim - 1] *= cfg.get_buf_tiles_num();
    //       m_offset_inc[iteration_dim - 1] *= cfg.get_buf_tiles_num();
    //       m_last_offset_inc[iteration_dim - 1] *= cfg.get_buf_tiles_num();
    //     }
      
    // }
    /**
     * @brief Calculating offset in elements inside memory buffer based on
     * pre-calculated offset increments per dimension
     *
     * @param dim [I] Dimension for incrementing
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

  int32_t m_buffer_offset;                 /**< Offset in elements inside attached memory 
                                                 buffer */
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
    * @param tensor [I] full tensor which is tiled by this iterator
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
    * Simple iterator with default iterator config will iterate in steps of 1
    * @param tensor [I] Full tensor which is tiled by this iterator
    * @param offset [I] Current offset used in TensorIterator
    * @param iteration_order [I] The iteration order used for increments
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

    TensorIterator(const TensorIterator<NoBuffer, tensorRank, iterRank>& tensor_iterator):m_buffer_itr(m_full_tensor, m_config) {
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
        unsigned num_mem) : m_buffer_itr(m_full_tensor, m_config) {
      m_full_tensor = Tensor<InternalBuffer, tensorRank>(tns_iter.get_tensor(), bases, num_mem);
      m_config = tns_iter.get_config();
      m_offset = tns_iter.get_offset();
      tns_iter.get_pos(m_pos);
      tns_iter.get_tile_idx(m_tile_idx);
    }

    /**
    * Constructor that will compute the number of tiles in each dimension, it will also compute the increment values and sizes.
    * This constructor takes into account effective kernel size and paddings.
    * Appropriate usage of this constructor is following: create IteratorCfg icfg for output tiling of conv-like kernels
    * and use this constructor to create input tiling of same conv-like kernel.
    */
    TensorIterator(const Tensor<buf_T, tensorRank>& tensor,  // full tensor
                   TensorIterator& tensor_iterator,          // origin tensor iterator
                   uint32_t effective_kernel_size[],         // used to calculate the overlaps between the tiles
                   uint32_t stride[],                        // used to calculate the overlaps between the tiles
                   uint32_t pre_padding[])                   // number of virtual pixels added before each dimension
        : m_full_tensor(tensor),
        m_config(tensor_iterator.get_config(), tensor, effective_kernel_size, stride, pre_padding),
        m_buffer_itr(m_full_tensor, m_config) {
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
      for (uint32_t r = 0; r < iterRank; r++) {
        kTilePos_t tile_pos = kMiddleTile; //initialize for middle tile
        uint32_t increment = m_config.get_inc(r);
        if(m_tile_idx[r] == m_config.get_count(r) - 1){   // check if it's a last tile
             tile_pos = kLastTile;
             increment = m_config.get_last_inc(r);
                     }
        else if (m_tile_idx[r] == 0){  //check it it's a first tile
             tile_pos = kFirstTile;
             increment = m_config.get_first_inc(r);

        }

        m_pos[r] += increment;
        m_buffer_itr.Next(r, tile_pos);
        if (m_tile_idx[r] == m_config.get_count(r) - 1) {
          m_pos[r] = 0;
          m_tile_idx[r] = 0;
          done = (r == iterRank - 1) ? true : false;
          m_buffer_itr.EndOfDim(r);

        } else {
          ++m_tile_idx[r];
          // Not at end of axis - break
          break;
        }
      }
      m_offset = m_buffer_itr.GetBufferOffset();
      return done;
    }
  
 // Return increments to the next tile (to increment outside the iterator)
    bool Next(int32_t* pos_inc) {
      bool done = false;
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



    Tensor<buf_T, tensorRank> GetSubTensor() {
      uint32_t pos[tensorRank];
      uint32_t copysize[tensorRank];
      uint32_t r = 0;
      for (r = 0; r < iterRank; r++) {
        int32_t dim = m_config.get_order(r);
        pos[dim] = (uint32_t)m_pos[r];
        if (m_tile_idx[r] == m_config.get_count(r) - 1) { // Last iteration
              copysize[dim] = m_config.get_last_size(r);
            } else if (m_tile_idx[r] == 0) { // First iteration
              copysize[dim] = m_config.get_first_size(r);
            } else { // Middle iteration
              copysize[dim] = m_config.get_size(r);
            }
      }
      return m_full_tensor.slice(pos, copysize);
    }

    TensorIterator<buf_T, tensorRank, iterRank> GetSubTensorIterator() {
      uint32_t current_size[tensorRank];

      for (uint32_t i = 0; i < tensorRank; i++) {
        current_size[i] =
            m_full_tensor.get_dim(i) - m_pos[i] < m_config.get_size(i)
                ? m_full_tensor.get_dim(i) - m_pos[i]
                : m_config.get_size(i);
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
      return m_full_tensor.get_buf().get_elem_size();
    }

    TensorIterator<buf_T, tensorRank, iterRank> transpose(uint32_t new_order[]) const {
      return TensorIterator(m_full_tensor.transpose(new_order), m_config.transpose_order(new_order));
    }

    template<typename T>
    T read(){
      return m_full_tensor.template read<T>(m_offset);
    }

    template <typename T>
    void write(T data) {
      m_full_tensor.write(m_offset, data);
    }

    /**
     * @brief   This method will update the tensor memory strides based on tile size
     *          in case of tiling 
     *
     */
      void UpdateMemstrides(void){
      if (m_config.get_buf_tiles_num() != 0) {
        uint32_t mem_stride=1;
             for (unsigned i = iterRank; i > 0; --i)
        {
          if(m_config.get_size(i-1)==0)continue;
          m_full_tensor.set_mem_stride(i-1,mem_stride);
          mem_stride*=m_config.get_size(i-1);
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

    int32_t get_offset() const {
      return m_offset;
    }

    void get_pos(int32_t pos[iterRank]) const {
      for (uint32_t i = 0; i < iterRank; i++) {
        pos[i] = m_pos[i];
      }
    }

    void get_tile_idx(int32_t tile_idx[iterRank]) const {
      for (uint32_t i = 0; i < iterRank; i++) {
        tile_idx[i] = m_tile_idx[i];
      }
    }

    bool is_first_tile(uint32_t dim_idx) const {
      return !m_tile_idx[dim_idx];
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
