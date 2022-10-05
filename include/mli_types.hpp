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
  kFirstTile = 0,
  kMiddleTile,
  kLastTile
} kTilePos_t;

constexpr short int kTensorBatchDim = 0;
constexpr short int kTensorHeightDim = 1;
constexpr short int kTensorWidthDim = 2;
constexpr short int kTensorChannelDim = 3;

constexpr short int kGroupTensorBatchDim = 0;
constexpr short int kGroupTensorHeightDim = 1;
constexpr short int kGroupTensorWidthDim = 2;
constexpr short int kGroupTensorGroupDim = 3;
constexpr short int kGroupTensorChannelDim = 4;

constexpr short int kTileGroupDim = 0;
constexpr short int kTileHeightDim = 1;
constexpr short int kTileWidthDim = 2;
constexpr short int kTileChannelDim = 3;

constexpr short int kKernelGroupDim = 0;
constexpr short int kKernelHeightDim = 1;
constexpr short int kKernelWidthDim = 2;
constexpr short int kKernelChannelInDim = 3;
constexpr short int kKernelChannelOutDim = 4;

constexpr short int kKernelDWHeightDim = 0;
constexpr short int kKernelDWWidthDim = 1;
constexpr short int kKernelDWChannelInDim = 2;

constexpr short int kKernelFCChannelInDim = 0;
constexpr short int kKernelFCChannelOutDim = 1;

constexpr short int kPerTensorQuantDim = -1;
constexpr short int kSkipIterDim = -1;

constexpr unsigned kMaxpoolRank = 4;
constexpr unsigned kMaxpoolIterRank = 4;

constexpr unsigned kConvIORank = 5;
constexpr unsigned kConvIOIterRank = 5;
constexpr unsigned kConvWRank = 5;
constexpr unsigned kConvWIterRank = 5;
constexpr unsigned kConvZPRank = 1;
constexpr unsigned kConvZPIterRank = 5;

constexpr unsigned kInpZPRank = 1;

constexpr unsigned kDepthwiseIORank = 5;
constexpr unsigned kDepthwiseWRank = 3;
constexpr unsigned kDepthwiseZPRank = 1;
constexpr unsigned kDepthwiseIterRank = 5;

constexpr unsigned kTransposeConvIORank = 5;
constexpr unsigned kTransposeConvIOIterRank = 5;
constexpr unsigned kTransposeConvWRank = 5;
constexpr unsigned kTransposeConvWIterRank = 5;
constexpr unsigned kTransposeConvZPRank = 1;
constexpr unsigned kTransposeConvZPIterRank = 5;

constexpr unsigned kPermuteRank = 4;
constexpr unsigned kPermuteIterRank = 4;

constexpr unsigned kClipRank = 4;
constexpr unsigned kClipIterRank = 4;
constexpr unsigned kClipParamRank = 1;
constexpr unsigned kClipParamIterRank = 1;

constexpr unsigned kRescaleRank = 4;
constexpr unsigned kRescaleIterRank = 4;
constexpr unsigned kRescaleParamRank = 1;
constexpr unsigned kRescaleParamIterRank = 1;

constexpr unsigned kPreluRank = 4;
constexpr unsigned kPreluIterRank = 4;
constexpr unsigned kPreluParamRank = 2;
constexpr unsigned kPreluParamIterRank = 2;

constexpr unsigned kReduceSumRank = 4;
constexpr unsigned kReduceSumIterRank = 4;
constexpr short int kMatMulRank = 2;
constexpr short int kMatMulIterRank = 2;

constexpr short int kArgMaxInRank = 4;
constexpr short int kArgMaxInIterRank = 4;
constexpr short int kArgMaxOutRank = 3;
constexpr short int kArgMaxOutIterRank = 3;

constexpr short int kTableBuiltinIORank = 4;
constexpr short int kTableBuiltinIOIterRank = 4;

constexpr short int kBiasRank = 1;
constexpr short int kBiasIterRank = 1;
constexpr unsigned kEltwiseIterRank = 4;
constexpr unsigned kEltwiseRank = 4;

constexpr short int kReduceMaxRank = 4;
constexpr short int kReduceMaxIterRank = 4;

constexpr unsigned kMoveBroadcastRank = 5;
constexpr unsigned kMoveBroadcastIterRank = 5;

constexpr short int kResizeDim = 2;
constexpr short int kResizeBilinearRank = 4;
constexpr short int kResizeBilinearIterRank = 4;

constexpr unsigned kMoveRank = 5;
constexpr unsigned kMoveIterRank = 5;

typedef enum : uint32_t {
  kInvalidId = 0,
  kNopId,
  kConv2dId,
  kPreluId,
  kMoveId,
  kDWConv2dId,
  kMaxPool2DId,
  kFullyConnectedId,
  kSumPool2DId,
  kAddId,
  kSubId,
  kMulId,
  kMaxId,
  kMinId,
  kRescaleId,
  kClipId,
  kReduceMaxId,
  kTransConv2DId,
  kPermuteId,
  kReduceSumId,
  kArgMaxId,
  kTableBuiltinId,
  kMatMulId,
  kMoveBroadcastId,
  kResizeBilinearId,
} kernel_id_t;

typedef enum class compression_mode_t {
  Uncompressed = 0,
  Compressed,
  Sparse
} compression_mode_t;

class PrivateData {
  public:
    PrivateData() : kernel_id(kInvalidId), size(0) {}
    PrivateData(kernel_id_t id, uint32_t s = 0) : kernel_id(id), size(s) {}
    kernel_id_t kernel_id;
    uint32_t size;
    bool issue_enable{false};
    bool prefetch_enable{false};
    uint16_t reserved;
};

enum MoveDataDirection {
  kMoveDataDirectionInput,
  kMoveDataDirectionOutput
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

  /**
   * @brief 
   * 
   * @param offset [I] offset is in number of elements (not in bytes)
   */
  void inc(unsigned offset) {
    ptr_ += elem_size_ * offset;
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
  T read(uint32_t offset) const{
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
 * @brief Buffer type used to communicate memory allocations between graph compiler and run-time
 *
 * This type contains an offset and a memory identifier.
 * The graph compiler doesn't know which (piece of) memory will be assigned to the graph.
 * It can only work with offsets inside the total requested blob of memory (for each memory type)
 * 
 * When the MLI runtime kernel object is created the bases addresses provided by the resource manager
 * are added to the offsets.
 * (e.g. the kernel private data structure contains unlinkedBuffer's and the runtime kernel object contains
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

  void set_elem_size(uint32_t elem_size) {
    assert(elem_size <= size_);
    elem_size_ = elem_size;
  }

  void set_size(uint32_t size) {
    size_ = size;
  }

  uint32_t get_mem_idx() const {
    return mem_idx_;
  }

  uint32_t get_offset() const {
    return offset_;
  }

  /**
   * @brief 
   * 
   * @param offset [I] offset is in number of elements (not in bytes)
   */
  void inc(unsigned offset) {
    offset_ += elem_size_ * offset;
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

  template <typename T>
  InternalBuffer(T* ptr, uint32_t size) {
    set_ptr(ptr);
    size_ = size;
  }

  uint32_t get_size() const {
    return size_;
  }

  uint32_t get_elem_size() const {
    return elem_size_;
  }

  void set_elem_size(uint32_t elem_size) {
    elem_size_ = elem_size;
  }

  template <typename T>
  void set_ptr(T* ptr) {
    ptr_ = reinterpret_cast<uint64_t>(ptr);
    elem_size_ = sizeof(T);
  }

  template <typename T>
  void set_buffer(T* ptr, uint32_t size) {
    ptr_ = reinterpret_cast<uint64_t>(ptr);
    elem_size_ = sizeof(T);
    size_ = size;
  }

  template <typename T>
  T* get_ptr() {
    assert(sizeof(T) == elem_size_);
    return reinterpret_cast<T*>(ptr_);
  }

  template <typename T>
  const T* get_ptr() const {
    assert(sizeof(T) == elem_size_);
    return reinterpret_cast<const T*>(ptr_);
  }

  // TODO: For Read/Write If we need platform specific handling, update the implementation
  template <typename T>
  T read(uint32_t offset) const {
    assert(sizeof(T) == elem_size_);
    return *(reinterpret_cast<T*>(ptr_) + offset);
  }

  template <typename T>
  void write(uint32_t offset, T data) {
    assert(sizeof(T) == elem_size_);
    *(reinterpret_cast<T*>(ptr_) + offset) = data;
  }

  /**
   * @brief 
   * 
   * @param offset [I] offset is in number of elements (not in bytes)
   */
  void inc(unsigned offset) {
    ptr_ += elem_size_ * offset;
  }

private:
  uint64_t ptr_;
  uint32_t size_;
  uint32_t elem_size_;
};

/**
 * @brief Tensor type - main data descriptor for all MLI_CS kernel operands and algorithms
 * 
 * @tparam buf_T type of the buffer handled by tensor
 * @tparam maxRank maximum rank of the tensor instance might be represent
 */
template <typename buf_T, unsigned maxRank>
class Tensor {
 public:

  /**
  * A default constructor for a tensor
  */
  Tensor() : buf_{buf_T()}, offset_(0), shape_{0}, mem_stride_{0}, rank_{0} {}

  /**
  * The completely specialized Tensor constructor
  */
  Tensor(buf_T buf, uint32_t shape[], int32_t mem_stride[], uint32_t rank)
      : buf_{buf}, offset_(0), shape_{0}, mem_stride_{0}, rank_{rank} {
    assert(rank_ <= maxRank);
    for (uint32_t i = 0; i < rank; ++i) {
      shape_[i] = shape[i];
      mem_stride_[i] = mem_stride[i];
    }
  }

  /**
  * The Specialized constructor for tensors with contiguous data
  */
  Tensor(buf_T buf, uint32_t shape[], uint32_t rank) 
      : buf_{buf}, offset_(0), shape_{0}, mem_stride_{0}, rank_{rank} {
    assert(rank_ <= maxRank);
    int32_t stride = 1;
    for (uint32_t cur_dim = rank_; cur_dim > 0; --cur_dim) {
      const uint32_t idx = cur_dim - 1;
      shape_[idx] = shape[idx];
      mem_stride_[idx] = stride;
      if (shape[idx] > 0) 
        stride *= shape[idx];
    }
  }

  /**
  * Various partly specialized constructors
  */
  Tensor(uint32_t shape[]) 
      : Tensor(NoBuffer(), shape, static_cast<uint32_t>(maxRank))  {}

  Tensor(buf_T buf, uint32_t shape[]) 
      : Tensor(buf, shape, static_cast<uint32_t>(maxRank)) {}

  Tensor(uint32_t shape[], uint32_t rank) 
      : Tensor(NoBuffer(), shape, rank) {}

  Tensor(buf_T buf, uint32_t shape[], int32_t mem_stride[]) 
      :  Tensor(buf, shape, mem_stride, static_cast<uint32_t>(maxRank)) {}

  Tensor(uint32_t shape[], int32_t mem_stride[]) 
      :  Tensor(NoBuffer(), shape, mem_stride, static_cast<uint32_t>(maxRank)) {}

  Tensor(uint32_t shape[], int32_t mem_stride[], uint32_t rank) 
      : Tensor(NoBuffer(), shape, mem_stride, rank) {}

  /* copy constructor for tensors with different rank */
  template <unsigned N>
  Tensor(Tensor<buf_T, N> in) {
    static_assert( N <= maxRank, "Invalid (Input Rank > maxRank)");
    buf_ = in.get_buf();
    offset_ = in.get_offs();
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

  /* copy constructor for almost same tensors, but different shape */
  template <unsigned N>
  Tensor(Tensor<buf_T, N> in, uint32_t shape[]) {
    static_assert(N <= maxRank, "Invalid (Input Rank > maxRank)");
    buf_ = in.get_buf();
    offset_ = in.get_offs();
    for (unsigned i = 0; i < N; i++) {
      shape_[i] = shape[i];
      mem_stride_[i] = in.get_mem_stride(i);
    }
    for (unsigned i = N; i < maxRank; i++) {
      shape_[i] = 0;
      mem_stride_[i] = 0;
    }
    rank_ = in.get_rank();
  }

  /* copy constructor for tensors with different rank/Buffer Type */
  template <unsigned N>
  Tensor(buf_T buf, Tensor<NoBuffer, N> in) {
    static_assert( N <= maxRank, "Invalid (Input Rank > maxRank)");
    buf_ = buf;
    offset_ = in.get_offs();
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
  Tensor(Tensor<OffsetBuffer, maxRank> in, uint64_t bases[], unsigned num_mems) {
    // this one can only be used to create an InternalBuffer from an OffsetBuffer
    buf_ = InternalBuffer(in.get_buf(), bases, num_mems);
    offset_ = in.get_offs();
    for (unsigned i = 0; i < maxRank; i++){
      shape_[i] = in.get_dim(i);
      mem_stride_[i] = in.get_mem_stride(i);
    }
    rank_ = in.get_rank();
  }

  uint32_t get_dim(unsigned idx) const {
    return shape_[idx];
  }

  void set_dim(unsigned idx, uint32_t shape) {
    shape_[idx] = shape;
  }

  void set_dims(uint32_t shape[]) {
    for (uint32_t i = 0; i < maxRank; i++) {
      shape_[i] = shape[i];
    }
  }

  void get_dims(uint32_t shape[]) const {
    for (uint32_t i = 0; i < maxRank; i++) {
      shape[i] = shape_[i];
    }
  }

  int32_t get_mem_stride(unsigned idx) const {
    return mem_stride_[idx];
  }
  
  void set_mem_stride(uint32_t idx, uint32_t val) {
    mem_stride_[idx] = val;
  }

  void set_mem_stride(unsigned idx, int32_t mem_stride) {
    mem_stride_[idx] = mem_stride;
  }

  void get_mem_strides(int32_t mem_strides[]) const {
    for (uint32_t i = 0; i < maxRank; i++) {
      mem_strides[i] = mem_stride_[i];
    }
  }

  uint32_t get_offs() const {
    return offset_;
  }

  void set_offs(uint32_t offset) {
    offset_ = offset;
  }

  uint32_t get_rank() const {
    return rank_;
  }

  void set_rank(uint32_t rank) {
    assert(rank <= maxRank);
    rank_ = rank;
  }

  buf_T get_buf() const {
    return buf_;
  }

  void set_buf(const buf_T& b) {
    buf_ = b;
  }
  
  uint32_t get_total_elem_num() const {
    uint32_t total_elem_num = 1;
    for (int i = 0; i < rank_; i++) total_elem_num *= shape_[i];
    return total_elem_num;
  }

  uint32_t get_elem_size() const {
    return buf_.get_elem_size();
  }

  uint32_t get_offset(uint32_t pos[]){
    uint32_t offset = 0;
    for (unsigned i = 0; i < rank_; ++i){
      offset += pos[i] * mem_stride_[i];
    }
    return offset;
  }

  Tensor<buf_T, maxRank> slice(uint32_t pos[], uint32_t size[]){
    uint32_t offset = get_offset(pos);
    Tensor<buf_T, maxRank> slice_tens(buf_, size, mem_stride_, rank_);
    slice_tens.set_offs(offset_ + offset);
    return slice_tens;
  }

  Tensor<buf_T, maxRank> slice(uint32_t offset, uint32_t size[]) {
    buf_T buf = buf_;
    buf.inc(offset);
    Tensor<buf_T, maxRank> slice_tens(buf, size, mem_stride_, rank_);
    return slice_tens;
  }
    
  Tensor<buf_T, maxRank> slice(uint32_t size[]) {
    buf_T buf = buf_;
    Tensor<buf_T, maxRank> slice_tens(buf, size, mem_stride_, rank_);
    return slice_tens;
  }

  Tensor<buf_T, maxRank> transpose(uint32_t new_order[]) const {
    // create a transposed Tensor, reordering the dimensions
    Tensor<buf_T, maxRank> tns;
    // change order of axes
    uint32_t c = 0;
    for (uint32_t axis = 0; axis < maxRank; axis++) {
      assert(new_order[axis] >= 0 && new_order[axis] < maxRank);
      // axis can only be selected once
      assert((c & (1 << new_order[axis])) == 0);
      c |= (1 << new_order[axis]);
      tns.shape_[axis] = shape_[new_order[axis]];
      tns.mem_stride_[axis] = mem_stride_[new_order[axis]];
    }
    tns.offset_ = offset_;
    tns.buf_ = buf_;
    tns.rank_ = rank_;
    return tns;
  }

  // split 'axis' dimension into 'chunks' chunks. place chunks dimension to the left or to the right of the splittd one
  Tensor<buf_T, maxRank+1> split(uint32_t axis, uint32_t chunks, bool left = false) const {
    assert(axis < maxRank);
    Tensor<buf_T, maxRank+1> tns;
    int s = 0;
    for (uint32_t r = 0; r < maxRank; ++r) {
      if (r < axis || r > axis) {
        tns.set_dim(s, shape_[r]);
        tns.set_mem_stride(s, mem_stride_[r]);
      } else {
        // split axis in 2
        assert(shape_[r]%chunks == 0);
        if (left) {
          tns.set_dim(s, chunks);
          tns.set_mem_stride(s, (shape_[r]/chunks)*mem_stride_[r]);
          ++s;
          tns.set_dim(s, shape_[r]/chunks);
          tns.set_mem_stride(s, mem_stride_[r]);
        } else {
          tns.set_dim(s, shape_[r]/chunks);
          tns.set_mem_stride(s, mem_stride_[r]);
          ++s;
          tns.set_dim(s, chunks);
          tns.set_mem_stride(s, (shape_[r]/chunks)*mem_stride_[r]);
        }
      }
      ++s;
    }
    tns.set_offs(offset_);
    tns.set_buf(buf_);
    tns.set_rank(rank_ + 1);
    return tns;
  }

  // combine 'axis' and 'axis'+1 dimensions into one dimension if possible
  Tensor<buf_T, maxRank-1> combine(uint32_t axis) const {
    assert(axis < maxRank - 1);
    Tensor<buf_T, maxRank-1> tns;
    int s = 0;
    for (int r = 0; r < maxRank; ++r) {
      if (r < axis || r > axis) {
        tns.set_dim(s, shape_[r]);
        tns.set_mem_stride(s, mem_stride_[r]);
      } else {
        // combine 2 adjacent axis into 1
        assert(mem_stride_[r+1] == shape_[r]*mem_stride_[r]);
        tns.set_dim(s, shape_[r]*shape_[r+1]);
        tns.set_mem_stride(s, mem_stride_[r]);
        ++r;
      }
      ++s;
    }
    tns.set_offs(offset_);
    tns.set_buf(buf_);
    tns.set_rank(rank_ - 1);
    return tns;
  }

  template <typename T>
  T read(uint32_t offset) const {
    return buf_.template read<T>(offset + offset_*get_elem_size()/sizeof(T));
  }

  template <typename T>
  void write(uint32_t offset, T data) {
    buf_.template write<T>(offset + offset_*get_elem_size()/sizeof(T), data);
  }

private:
  buf_T buf_;
  uint32_t offset_;
  uint32_t shape_[maxRank];
  int32_t mem_stride_[maxRank];
  uint32_t rank_;
};

// Quantized Tensor
template <typename buf_T, unsigned rank>
struct QTensor {
  Tensor<buf_T, rank> t;
  buf_T zp;
  int quant_axis;
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
    uint32_t padding_begin[2]; /**< Padding size at the beginning of spatial dimensions of input [pad_IH_beg, pad_IW_beg]*/
    uint32_t padding_end[2];   /**< Padding size at the end of spatial dimensions of input [pad_IH_end, pad_IW_end]*/
    uint32_t dilation[2];      /**< Dilation Factor [dilation_IH, dilation_IW].
                                   If set to dilation_I*>1, there will be k-1 implicitly added zero points between each
                                   filter point across appropriate dimension. If set to 1, no dilation logic is used */
    uint32_t groups;           /**< Number of groups input channels and output channels are divided into. */
};

struct DwConv2DConfig {
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
    uint32_t padding_begin[2]; /**< Padding size at the beginning of spatial dimensions of input [pad_IH_beg, pad_IW_beg]*/
    uint32_t padding_end[2];   /**< Padding size at the end of spatial dimensions of input [pad_IH_end, pad_IW_end]*/
    uint32_t dilation[2];      /**< Dilation Factor [dilation_IH, dilation_IW].
                                    If set to dilation_I*>1, there will be k-1 implicitly added zero points between each
                                    filter point across appropriate dimension. If set to 1, no dilation logic is used */
};

struct TransposeConv2DConfig {
    TransposeConv2DConfig() = default;
    TransposeConv2DConfig(uint32_t stride_ih, uint32_t stride_iw,
                          uint32_t pad_beg_ih, uint32_t pad_beg_iw,
                          uint32_t pad_end_ih, uint32_t pad_end_iw)
        : stride{stride_ih, stride_iw},
          padding_begin{pad_beg_ih, pad_beg_iw},
          padding_end{pad_end_ih, pad_end_iw} {}

    uint32_t stride[2]; /**< Stride along each axis [stride_IH, stride_IW]*/
    uint32_t padding_begin[2]; /**< Padding size at the beginning of spatial dimensions of input [pad_IH_beg, pad_IW_beg]*/
    uint32_t padding_end[2]; /**< Padding size at the end of spatial dimensions of input [pad_IH_end, pad_IW_end]*/
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
    uint32_t stride[2];        /**< Stride along each axis [stride_IH, stride_IW] */
    uint32_t padding_begin[2]; /**< Padding size at the beginning of spatial dimensions of input [pad_IH_beg, pad_IW_beg] */
    uint32_t padding_end[2];   /**< Padding size at the end of spatial dimensions of input [pad_IH_end, pad_IW_end] */
};


enum struct LutType: int32_t {
  kSigmoid = 0,
  kTanH,
  kNegExp,
  kMish,
  kSwish,
  kGelu,
  kReciprocSqrt,
  kReciproc,
  kHswish
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
  RescaleConfig(int32_t axis) : axis{axis} {};

  int32_t axis; /**< An axis along which the function will be computed.
                     Axis corresponds to index of tensor`s dimension starting from 0.
                     For instance, having future map in HWC layout, axis == 0 corresponds to H dimension.
                     If axis < 0 the function will be applied to the whole tensor */
};

struct ReduceOpConfig {
  ReduceOpConfig() = default;
  ReduceOpConfig(int32_t axis) : axis{axis} {};

  int32_t axis;   /**< An axis along which the function will be computed.
                       Axis corresponds to index of tensor`s dimension starting from 0.
                       For instance, having future map in HWC layout, axis == 0 corresponds to H dimension.
                       If axis < 0 the function will be applied to the whole tensor */
};

/**
 * @brief Permute layer config definition
 *
 * Data structure to provide the permutation order to functions.
 */
struct PermuteOpConfig {
  PermuteOpConfig() = default;
  PermuteOpConfig(uint8_t *perm_dim) {
    for(uint32_t i = 0; i < kPermuteRank; i++) {
      this->perm_dim[i] = perm_dim[i];
    }
  }

  uint8_t perm_dim[kPermuteRank];   /**< A permutation array. Dimensions order for output tensor. */
};

struct ArgMaxConfig {
  ArgMaxConfig() = default;
  ArgMaxConfig(int32_t axis) : axis{axis} {};

  int32_t axis;   /**< An axis along which the function will be computed.
                       Axis corresponds to index of tensor`s dimension starting from 0.
                       For instance, having future map in HWC layout, axis == 0 corresponds to H dimension.
                       If axis < 0 the function will be applied to the whole tensor */
};

struct PreluOpConfig {
  PreluOpConfig() = default;
  PreluOpConfig(int32_t axis) : axis{axis} {};

  int32_t axis; /**< An axis along which the function will be computed.
                     Axis corresponds to index of tensor`s dimension starting from 0.
                     For instance, having future map in HWC layout, axis == 0 corresponds to H dimension.
                     If axis < 0 the function will be applied to the whole tensor */
};

struct ResizeOpConfig {
  ResizeOpConfig() = default;
  ResizeOpConfig(int16_t *stride, int16_t *offset, int8_t shift) {
    for(int8_t i = 0; i < kResizeDim; i++) {
      this->stride[i] = stride[i];
      this->offset[i] = offset[i];
    }
    this->shift = shift;
  }

  int16_t stride[kResizeDim];    /**< [stride_H, stride_W] */
  int16_t offset[kResizeDim];    /**< [offset_H, offset_W] */
  int8_t shift;         /**< Shift value (for fractional stride and offset) */

};

} // namespace snps_arc::metaware::mli

#endif /* _MLI_TYPES_HPP_ */
