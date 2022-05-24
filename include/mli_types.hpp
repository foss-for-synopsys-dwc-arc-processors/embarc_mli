/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_TYPES_HPP_
#define _MLI_TYPES_HPP_

#include <stdint.h>
#include <assert.h>


namespace snps_arc::metaware::mli {

typedef enum {
  kInvalidId = 0,
  kConv2dId,
  kPreluId,
  kMoveId,
  kDWConv2dId,
  kMaxPool2DId,
  kAddId,
  kSomeOtherKernelId
} kernel_id_t;

typedef enum class compression_mode_t {
  Uncompressed = 0,
  Compressed,
  Sparse
} compression_mode_t;

class PrivateData {
  public:
    PrivateData(kernel_id_t id) {
      kernel_id = id;
      size = 0;
    }
    kernel_id_t kernel_id;
    uint32_t size;
};

/**
 * @brief Dummy Class to be used to indicate no Buffer is used in tensor.
 *
 */
class NoBuffer {

};

/**
 * @brief Buffer type for absolute buffers
 *
 * In case of simulation this buffer 'lives' inside the x86 memory domain.
 * In case of target exectution this buffer lives in the native memory domain of the target.
 */
class Buffer {
public:
  Buffer() = default;
  Buffer(void* ptr, uint32_t size, uint32_t elem_size = 1) {
    ptr_ = reinterpret_cast<uint64_t>(ptr);
    size_ = size;
    elem_size_ = elem_size;
  }

  uint32_t get_size() const {
    return size_;
  }
  uint32_t get_elem_size() const {
    return elem_size_;
  }

  template<typename T>
  void set_ptr(T* ptr){
    ptr_ = reinterpret_cast<uint64_t>(ptr);
    elem_size_ = sizeof(T);
  }

  template<typename T>
  void set_buffer(T* ptr, uint32_t size){
    ptr_ = reinterpret_cast<uint64_t>(ptr);
    elem_size_ = sizeof(T);
    size_ = size;
  }

  template<typename T>
  T* get_ptr(){
    assert(sizeof(T) == elem_size_);
    return reinterpret_cast<T*>(ptr_);
  }

  template<typename T>
  T read(uint32_t offset){
    assert(sizeof(T) == elem_size_);
    return *(reinterpret_cast<T*>(ptr_) + offset);
  }
  template<typename T>
  void write(uint32_t offset, T data){
    assert(sizeof(T) == elem_size_);
    *(reinterpret_cast<T*>(ptr_) + offset) = data;
  }

private:
  uint64_t ptr_;
  uint32_t size_;
  uint32_t elem_size_;
};


/**
 * @brief Buffer type used to communicate memory allocatios between graph compiler and run-time
 *
 * This type contains an offset and a memory identifier.
 * The graph compiler doesn't know which (piece of) memory will be assigned to the graph.
 * It can only work with offsets inside the total requested blob of memory (for each memory type)
 * 
 * When the MLI runtime kernel object is created the bases addresses provided by the resource manager
 * are added to the offsets.
 * (e.g. the kernel private data structure contians unlinkedBuffer's and the runtime kernel object contains
 * DeviceBuffer's, and the constructor of the runtime object translates one into the other.)
 * 
 */
class OffsetBuffer {
public:
  OffsetBuffer() {
    offset_ = 0;
    mem_idx_ = 0;
    size_ = 0;
    elem_size_ = 0;
  }

  OffsetBuffer(uint32_t offset, uint32_t mem_idx, uint32_t size, uint32_t elem_size) {
    offset_ = offset;
    mem_idx_ = mem_idx;
    size_ = size;
    elem_size_ = elem_size;
  }

  uint32_t get_size() const {
    return size_;
  }
  uint32_t get_elem_size() const {
    return elem_size_;
  }
  uint32_t get_mem_idx() const {
    return mem_idx_;
  }
  uint32_t get_offset() const {
    return offset_;
  }
// read and write not possible from this buffer.

private:
  uint32_t offset_;
  uint32_t mem_idx_;
  uint32_t size_;
  uint32_t elem_size_;
};

/**
 * @brief Buffer type used inside the implementation.
 *
 * In case of simulation this buffer 'lives' inside the modelled memory domain.
 * In case of target exectuion this buffer is inside one of the target memories.
 * All access to data inside this buffer needs to go through mem_read and mem_write functions.
 * in simulation these functions will communicate with the memory model
 * on a real target these functions will map to load/store operations
 */
class InternalBuffer {

public:
  InternalBuffer() {
    ptr_ = 0;
    size_ = 0;
    elem_size_ = 0;
  }

  InternalBuffer(uint64_t ptr, uint32_t size, uint32_t elem_size) {
    ptr_ = ptr;
    size_ = size;
    elem_size_ = elem_size;
  }

  InternalBuffer(OffsetBuffer buf, uint64_t bases[], unsigned num_mems){
    unsigned memidx = buf.get_mem_idx();
    assert(memidx < num_mems);
    ptr_ = bases[memidx] + buf.get_offset();
    size_ = buf.get_size();
    elem_size_ = buf.get_elem_size();
  }

  uint32_t get_size(){
    return size_;
  }
  uint32_t get_elem_size(){
    return elem_size_;
  }
  
  template<typename T>
  void set_ptr(T* ptr){
    ptr_ = static_cast<uint64_t>(ptr);
    elem_size_ = sizeof(T);
  }

  template<typename T>
  void set_buffer(T* ptr, uint32_t size){
    ptr_ = static_cast<uint64_t>(ptr);
    elem_size_ = sizeof(T);
    size_ = size;
  }

  template<typename T>
  T* get_ptr(){
    assert(sizeof(T) == elem_size_);
    return reinterpret_cast<T*>(ptr_);
  }

  // TODO: For Read/Write If we need platform specific handling, update the implementation 
  template<typename T>
  T read(uint32_t offset){
    assert(sizeof(T) == elem_size_);
    return *(reinterpret_cast<T*>(ptr_) + offset);
  }

  template<typename T>
  void write(uint32_t offset, T data){
    assert(sizeof(T) == elem_size_);
    *(reinterpret_cast<T*>(ptr_) + offset) = data;
  }

  void inc(unsigned offset){
    ptr_ += elem_size_ * offset;
  }
private:
  uint64_t ptr_;
  uint32_t size_;
  uint32_t elem_size_;
};

template<typename buf_T, unsigned maxRank>
class Tensor {

public:
  Tensor(){
    buf_ = buf_T();
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = 0;
      mem_stride_[i] = 0;
    }
    rank_ = 0;
  }

  Tensor(buf_T buf, uint32_t shape[]){
    buf_ = buf;
    int32_t stride = 1;
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = stride;
      stride *= shape[i];
    }
    rank_ = maxRank;
  }

  Tensor(uint32_t shape[]){
    buf_ = NoBuffer();
    int32_t stride = 1;
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = stride;
      stride *= shape[i];
    }
    rank_ = maxRank;
  }

  Tensor(buf_T buf, uint32_t shape[], unsigned rank){
    assert(rank <= maxRank);
    buf_ = buf;
    int32_t stride = 1;
    for (unsigned i = 0; i < rank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = stride;
      stride *= shape[i];
    }
    rank_ = rank;
  }

  Tensor(uint32_t shape[], unsigned rank){
    assert(rank <= maxRank);
    buf_ = NoBuffer();
    int32_t stride = 1;
    for (unsigned i = 0; i < rank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = stride;
      stride *= shape[i];
    }
    rank_ = rank;
  }

  Tensor(buf_T buf, uint32_t shape[], int32_t mem_stride[]){
    buf_ = buf;
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = mem_stride[i];
    }
    rank_ = maxRank;
  }

  Tensor(uint32_t shape[], int32_t mem_stride[]){
    buf_ = NoBuffer();
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = mem_stride[i];
    }
    rank_ = maxRank;
  }

  Tensor(buf_T buf, uint32_t shape[], int32_t mem_stride[], unsigned rank){
    assert(rank <= maxRank);
    buf_ = buf;
    for (unsigned i = 0; i < rank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = mem_stride[i];
    }
    rank_ = rank;
  }

  Tensor(uint32_t shape[], int32_t mem_stride[], unsigned rank){
    assert(rank <= maxRank);
    buf_ = NoBuffer();
    for (unsigned i = 0; i < rank; i++){
      shape_[i] = shape[i];
      mem_stride_[i] = mem_stride[i];
    }
    rank_ = rank;
  }

  /* copy constructor for tensors with different rank */
  template<unsigned N>
  Tensor(Tensor<buf_T, N> in){
    static_assert( N <= maxRank, "Invalid (Input Rank > maxRank)");
    buf_ = in.get_buf();
    for (unsigned i = 0; i < N; i++){
      shape_[i] = in.get_dim(i);
      mem_stride_[i] = in.get_mem_stride(i);
    }
    for (unsigned i = N; i < maxRank; i++){
      shape_[i] = 0;
      mem_stride_[i] = 0;
    }
    rank_ = in.get_rank();
  }

  /* copy constructor for tensors with different rank/Buffer Type */
  template<unsigned N>
  Tensor(buf_T buf, Tensor<NoBuffer, N> in){
    static_assert( N <= maxRank, "Invalid (Input Rank > maxRank)");
    buf_ = buf;
    for (unsigned i = 0; i < N; i++){
      shape_[i] = in.get_dim(i);
      mem_stride_[i] = in.get_mem_stride(i);
    }
    for (unsigned i = N; i < maxRank; i++){
      shape_[i] = 0;
      mem_stride_[i] = 0;
    }
    rank_ = in.get_rank();
  }

  /* 'copy' constructors for tensors with different buffer types */
  Tensor(Tensor<OffsetBuffer, maxRank> in, uint64_t bases[], unsigned num_mems){
    // this one can only be used to create an InternalBuffer from an OffsetBuffer
    buf_ = InternalBuffer(in.get_buf(), bases, num_mems);
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = in.get_dim(i);
      mem_stride_[i] = in.get_mem_stride(i);
    }
    rank_ = in.get_rank();
  }

  uint32_t get_dim(unsigned idx) const {
    return shape_[idx];
  }
  int32_t get_mem_stride(unsigned idx) const {
    return mem_stride_[idx];
  }

  unsigned get_rank() const {
    return rank_;
  }

  buf_T get_buf() const {
    return buf_;
  }

  void set_buf(const buf_T& b) {
    buf_ = b;
  }

  unsigned get_elem_size() const {
    return buf_.get_elem_size();
  }

  Tensor<buf_T, maxRank> slice(int32_t pos[], uint32_t size[]){
    buf_T buf = buf_;
    unsigned offset = 0;
    for (unsigned i = 0; i < maxRank; i++){
      offset += pos[i] * mem_stride_[i];
    }
    buf.inc(offset);
    Tensor<buf_T, maxRank> slice_tens(buf, size, mem_stride_, rank_);
    return slice_tens;
  }

  template<typename T>
  T read(uint32_t offset){
    return buf_.template read<T>(offset);
  }

  template<typename T>
  void write(uint32_t offset, T data){
    buf_.write(offset, data);
  }

private:
  buf_T buf_;
  uint32_t shape_[maxRank];
  int32_t mem_stride_[maxRank];
  uint32_t rank_;
};

//================================================================
//
//              Kernels Configurations Definition
//
//=================================================================

struct Conv2DConfig {
    Conv2DConfig() = default;
    Conv2DConfig(uint32_t stride_ih, uint32_t stride_iw, 
                 uint32_t pad_beg_ih, uint32_t pad_beg_iw,
                 uint32_t pad_end_ih, uint32_t pad_end_iw,
                 uint32_t dilation_ih, uint32_t dilation_iw,
                 uint32_t groups) 
      : stride{stride_ih, stride_iw}
      , padding_begin{pad_beg_ih, pad_beg_iw}
      , padding_end{pad_end_ih, pad_end_iw}
      , dilation{dilation_ih, dilation_iw}
      , groups{groups}
    {}

    uint32_t stride[2];        /**< Stride along each axis [stride_IH, stride_IW]*/
    uint32_t padding_begin[2]; /**< Padding size at the begining of spatial demensions of input [pad_IH_beg, pad_IW_end]*/
    uint32_t padding_end[2];   /**< Padding size at the end of spatial demensions of input [pad_IH_end, pad_IW_end]*/
    uint32_t dilation[2];      /**< Dilation Factor [dilation_IH, dilation_IW].
                                   If set to dilation_I*>1, there will be k-1 implicitly added zero points between each
                                   filter point across appropriate dimension. If set to 1, no dilation logic is used */
    uint32_t groups;           /**< Number of groups input channels and output channels are divided into. */
};

struct DwConv2DConfig  {
    DwConv2DConfig() = default;
    DwConv2DConfig(uint32_t stride_ih, uint32_t stride_iw, 
                   uint32_t pad_beg_ih, uint32_t pad_beg_iw,
                   uint32_t pad_end_ih, uint32_t pad_end_iw,
                   uint32_t dilation_ih, uint32_t dilation_iw) 
      : stride{stride_ih, stride_iw}
      , padding_begin{pad_beg_ih, pad_beg_iw}
      , padding_end{pad_end_ih, pad_end_iw}
      , dilation{dilation_ih, dilation_iw}
    {}

    uint32_t stride[2];        /**< Stride along each axis [stride_IH, stride_IW]*/
    uint32_t padding_begin[2]; /**< Padding size at the begining of spatial demensions of input [pad_IH_beg, pad_IW_end]*/
    uint32_t padding_end[2];   /**< Padding size at the end of spatial demensions of input [pad_IH_end, pad_IW_end]*/
    uint32_t dilation[2];      /**< Dilation Factor [dilation_IH, dilation_IW].
                                    If set to dilation_I*>1, there will be k-1 implicitly added zero points between each
                                    filter point across appropriate dimension. If set to 1, no dilation logic is used */
};

struct PoolOpConfig {
    PoolOpConfig() = default;
    PoolOpConfig(uint32_t kernel_size_ih, uint32_t kernel_size_iw,
                 uint32_t stride_ih, uint32_t stride_iw, 
                 uint32_t pad_beg_ih, uint32_t pad_beg_iw,
                 uint32_t pad_end_ih, uint32_t pad_end_iw) 
      : kernel_size{kernel_size_ih, kernel_size_iw}
      , stride{stride_ih, stride_iw}
      , padding_begin{pad_beg_ih, pad_beg_iw}
      , padding_end{pad_end_ih, pad_end_iw}
    {}

    uint32_t kernel_size[2];   /**< Kernel size of pooling function [kernel_H, kernel_W] */
    uint32_t stride[2];        /**< Stride along each axis [stride_IH, stride_IW]*/
    uint32_t padding_begin[2]; /**< Padding size at the begining of spatial demensions of input [pad_IH_beg, pad_IW_end]*/
    uint32_t padding_end[2];   /**< Padding size at the end of spatial demensions of input [pad_IH_end, pad_IW_end]*/
};


enum struct LutType: int32_t {
  kSigmoid = 0,
  kTanH,
  kNegExp,
  kMish,
  kSwish,
  kGelu,
  kReciprocSqrt,
  kReciproc
};


struct TableBuiltinConfig {
  TableBuiltinConfig() = default;
  TableBuiltinConfig(LutType lut_type, bool innermost_dim_bias)
    : type{lut_type}
    , innermost_dim_bias{innermost_dim_bias}
  {}

  LutType type;             /**< Type of the table which should be used by the kernel */
  bool innermost_dim_bias;  /**<  Is bias provided per innermost dimension. if false implies per-tensor bias.
                                  Otherwise implies separate bias value per slice across innermost dimension */
};

struct RescaleConfig {
  RescaleConfig() = default;
  RescaleConfig(bool innermost_dim_rescale)
    : innermost_dim_rescale{innermost_dim_rescale}
  {}

  bool innermost_dim_rescale; /**<  Is rescaling params (scale, shift in bias,, out bias) provided per innermost dimension.
                                    if false implies per-tensor rescaling.
                                    Otherwise implies separate bias value per slice across innermost dimension */
};




struct ReduceOpConfig {
  ReduceOpConfig() = default;
  ReduceOpConfig(int32_t axis) : axis{axis} {};

  int32_t axis;   /**< Axis to reduce, in range from 0 to rank(shape1) - 1
                       Axis corresponds to index of tensor`s dimension starting from 0 to (rank - 1).
                       For instance, having feature map in HWC layout, axis == 0 corresponds to H dimension.
                       If axis < 0 the function will be applied to the whole tensor */
};


} // namespace snps_arc::metaware::mli

#endif /* _MLI_TYPES_HPP_ */
