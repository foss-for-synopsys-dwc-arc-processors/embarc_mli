/*
* Copyright 2022, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/
#ifndef _MLI_REF_RUNTIME_API_HPP_
#define _MLI_REF_RUNTIME_API_HPP_

#include "mli_runtime_api.hpp"
#include "mli_ref_compiler_api.hpp"
#include "mli_iterator.hpp"
#include "mli_ref_private_types.hpp"

namespace lib_mli = ::snps_arc::metaware::mli;

namespace snps_arc::metaware::mli::ref {

using lib_mli::ExecutionInterface;
using lib_mli::PrivateData;

class Conv2d : public ExecutionInterface {

public:
    Conv2d(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    Conv2dMetadata m_metadata;
    // element size of input feature map
    uint32_t m_i_elem_size;
    // element size of weights
    uint32_t m_w_elem_size;
    // element size of output
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the DepthwiseConv2d kernel xop interpreter interface
 *
 *
 */
class DepthwiseConv2d : public ExecutionInterface {

public:
    /**
     * @brief constructor for the DepthwiseConv2d
     *
     * This Method will create and initialize the object using the information
     * stored in the kernel_private_data_buffer that has been computed at compile time
     * by the get_kernel_private_data() method.
     *
     * @param kernel_private_data_buffer [I] pointer to the compiletime computed initialization data
     * @param size        [I] Size of the data is used to check for coding errors
     * @param membases[]  [I] The kernel private data may contain offsets inside a (vector) memory.
     *                        At run-time specific locations in memory are allocated for
     *                        the graph, the membase array contains the is the start of
     *                        each memory region.
     *                        The init method will add this base to all the memory offsets
     *                        inside the descriptor according to the memory number associated
     *                        with that offset.
     *                        Each platform can have different (number of) memories. For mli
     *                        this is completely transparant. Compiler needs to use the same
     *                        memory id's when attaching the buffers as are used by the
     *                        xop-interpreter to set the membases.
     */
    DepthwiseConv2d(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    DepthwiseConv2dMetadata m_metadata;
    // element size of input feature map
    uint32_t m_i_elem_size;
    // element size of weights
    uint32_t m_w_elem_size;
    // element size of output
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Move kernel xop interpreter interface
 *
 *
 */

class Move : public ExecutionInterface {

public:
    /**
     * @brief constructor to create a move run-time object from a private data buffer from the Move_CS class
     *
     * This Method will create and initialize the object using the information
     * stored in the kernel_private_data_buffer that has been computed at compile time
     * by the get_kernel_private_data() method.
     *
     * @param kernel_private_data_buffer [I] pointer to the compiletime computed initialization data
     * @param size        [I] Size of the data is used to check for coding errors
     * @param membases[]  [I] The kernel private data may contain offsets inside a (vector) memory.
     *                        At run-time specific locations in memory are allocated for
     *                        the graph, the membase array contains the is the start of
     *                        each memory region.
     *                        The init method will add this base to all the memory offsets
     *                        inside the descriptor according to the memory number associated
     *                        with that offset.
     *                        Each platform can have different (number of) memories. For mli
     *                        this is completely transparant. Compiler needs to use the same
     *                        memory id's when attaching the buffers as are used by the
     *                        xop-interpreter to set the membases.
     */
    Move(void* kernel_private_data_buffer, size_t size,
         uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    TensorIterator<Move_CS::kMaxRank> m_src_it;
    TensorIterator<Move_CS::kMaxRank> m_dst_it;
    IteratorCfg<Move_CS::kMaxRank> m_src_it_cfg;
    IteratorCfg<Move_CS::kMaxRank> m_dst_it_cfg;

    template <typename buf_T, unsigned N>
    void CopySrcToDst(Tensor<buf_T, N> src, Tensor<buf_T, N> dst);

    TensorIterator<Move_CS::kMaxRank> GetSrcTensorTileItr(
        void* kernel_private_data_buffer, uint64_t membases[],
        int num_mems);

    TensorIterator<Move_CS::kMaxRank> GetDstTensorTileItr(
        void* kernel_private_data_buffer, uint64_t membases[],
        int num_mems);
};

/**
 * @brief This class implements the MaxPool2D kernel xop interpreter interface
 *
 *
 */
class MaxPool2D : public ExecutionInterface {

public:
    MaxPool2D(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_pool_cfg m_cfg;
    mli_tensor m_input;
    mli_tensor m_output;
    int32_t m_input_batch_offset;
    int32_t m_output_batch_offset;
    uint32_t m_batch_number;
    uint32_t m_io_elem_size;
};


class FullyConnected : public ExecutionInterface {

public:
    /**
     * @brief constructor for the FullyConnected
     *
     * This Method will create and initialize the object using the information
     * stored in the kernel_private_data_buffer that has been computed at compile time
     * by the get_kernel_private_data() method.
     *
     * @param kernel_private_data_buffer [I] pointer to the compiletime computed initialization data
     * @param size        [I] Size of the data is used to check for coding errors
     * @param membases[]  [I] The kernel private data may contain offsets inside a (vector) memory.
     *                        At run-time specific locations in memory are allocated for
     *                        the graph, the membase array contains the is the start of
     *                        each memory region.
     *                        The init method will add this base to all the memory offsets
     *                        inside the descriptor according to the memory number associated
     *                        with that offset.
     *                        Each platform can have different (number of) memories. For mli
     *                        this is completely transparant. Compiler needs to use the same
     *                        memory id's when attaching the buffers as are used by the
     *                        xop-interpreter to set the membases.
     * @param num_mems    [I] Number of elements in the membases array.
     */
    FullyConnected(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    FullyConnectedMetadata m_metadata;
    // element size of input feature map
    uint32_t m_i_elem_size;
    // element size of weights
    uint32_t m_w_elem_size;
    // element size of output
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the SumPool2D kernel xop interpreter interface
 *
 *
 */
class SumPool2D : public ExecutionInterface {

public:
    SumPool2D(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_pool_cfg m_cfg;
    mli_tensor m_input;
    mli_tensor m_output;
    int32_t m_input_batch_offset;
    int32_t m_output_batch_offset;
    uint32_t m_batch_number;
    uint32_t m_i_elem_size;
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Add kernel xop interpreter interface
 *
 *
 */
class Add : public ExecutionInterface {

public:
    Add(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_tensor m_input_left;
    mli_tensor m_input_right;
    mli_tensor m_output;
    uint32_t m_i_elem_size;
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Sub kernel xop interpreter interface
 *
 *
 */
class Sub : public ExecutionInterface {

public:
    Sub(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_tensor m_input_left;
    mli_tensor m_input_right;
    mli_tensor m_output;
    uint32_t m_i_elem_size;
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Mul kernel xop interpreter interface
 *
 *
 */
class Mul : public ExecutionInterface {

public:
    Mul(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_tensor m_input_left;
    mli_tensor m_input_right;
    mli_tensor m_output;
    uint32_t m_i_elem_size;
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Max kernel xop interpreter interface
 *
 *
 */
class Max : public ExecutionInterface {

public:
    Max(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_tensor m_input_left;
    mli_tensor m_input_right;
    mli_tensor m_output;
    uint32_t m_i_elem_size;
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Min kernel xop interpreter interface
 *
 *
 */
class Min : public ExecutionInterface {

public:
    Min(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    mli_tensor m_input_left;
    mli_tensor m_input_right;
    mli_tensor m_output;
    uint32_t m_i_elem_size;
    uint32_t m_o_elem_size;
};

/**
 * @brief This class implements the Rescale kernel xop interpreter interface
 *
 *
 */
class Rescale : public ExecutionInterface {

public:
    Rescale(void* kernel_private_data_buffer, size_t size, uint64_t membases[], int num_mems);

    mli_status Issue() override;

    mli_status Prefetch() override;

    mli_status Update() override;

private:
    RescaleMetadata m_metadata;

    uint32_t m_in_elem_size;
    uint32_t m_out_elem_size;
};

} // namespace snps_arc::metaware::mli::ref

#endif // _MLI_REF_RUNTIME_API_HPP_
