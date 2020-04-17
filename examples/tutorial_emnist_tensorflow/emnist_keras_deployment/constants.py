import inspect

PROJECT_NAME = ''
PROJECT_NAME_CAPS = ''
KERNEL_TYPE= ''
DEBUG_VERSION = False
NUM_CLASSES = 0
NORM_VALUE = 0
FRAC_BITS = 0
TEST_SAMPLE = 0

WEIGHTS_FILE_NAME = ''
CONSTANTS_FILE_NAME = ''
TEST_FILE_NAME = ''
MODEL_H_FILE_NAME = ''
MODEL_C_FILE_NAME = ''
ML_API_FILE_NAME = ''

MODEL_FOLDER = 'model'
IR_TEST_FOLDER = 'idx'
TEST_DATASET_FOLDER = 'small_test_base'

def create(project_name, num_classes, kernel_type, debug, norm_value, frac_bits, test_sample, ir_test_folder, test_dataset_folder):
    global PROJECT_NAME
    global PROJECT_NAME_CAPS
    global KERNEL_TYPE
    global DEBUG_VERSION
    global NUM_CLASSES
    global NORM_VALUE
    global FRAC_BITS
    global TEST_SAMPLE
    global IR_TEST_FOLDER
    global TEST_DATASET_FOLDER

    global WEIGHTS_FILE_NAME
    global CONSTANTS_FILE_NAME
    global MODEL_H_FILE_NAME
    global MODEL_C_FILE_NAME
    global TEST_FILE_NAME
    global ML_API_FILE_NAME

    PROJECT_NAME = project_name
    PROJECT_NAME_CAPS = str.upper(PROJECT_NAME)
    KERNEL_TYPE = kernel_type
    DEBUG_VERSION = debug
    NUM_CLASSES = num_classes
    NORM_VALUE = norm_value
    FRAC_BITS = frac_bits
    TEST_SAMPLE = test_sample
    IR_TEST_FOLDER = ir_test_folder
    TEST_DATASET_FOLDER = test_dataset_folder

    WEIGHTS_FILE_NAME = PROJECT_NAME + '_coefficients_small.c'
    CONSTANTS_FILE_NAME = PROJECT_NAME + '_constants.h'
    TEST_FILE_NAME = PROJECT_NAME + '_ref_inout.h'
    MODEL_H_FILE_NAME = PROJECT_NAME + '_model.h'
    MODEL_C_FILE_NAME = PROJECT_NAME + '_model_chw.c'
    ML_API_FILE_NAME = 'ml_api_' + PROJECT_NAME + '_tensorflow_main.c'


def execution_log_decorator(func):
    def wrapper(*args, **kwargs):
        print('Executing ', func.__name__)
        func(*args, **kwargs)
        print('Done\n')
    return wrapper


# Weights file constants
WEIGHTS_FILE_INCLUDES = inspect.cleandoc(
    '''
    #include "{constants_file_name}"
    '''
)

CONVOLUTION_WEIGHTS_ARRAY_NAME = 'const w_type _W{type} L{layer_idx}_conv_wt_buf[CONV{layer_idx}_W_ELEMENTS]'
DENSE_WEIGHTS_ARRAY_NAME = 'const w_type _W{type} L{layer_idx}_fc_wt_buf[FC{layer_idx}_W_ELEMENTS]'
CONVOLUTION_BIASES_ARRAY_NAME = 'const w_type _W{type} L{layer_idx}_conv_bias_buf[CONV{layer_idx}_B_ELEMENTS]'
DENSE_BIASES_ARRAY_NAME = 'const w_type _W{type} L{layer_idx}_fc_bias_buf[FC{layer_idx}_B_ELEMENTS]'

WEIGHTS_QUANT_WRAPPER = 'L{}_WQ({: 2.9f})'
BIASES_QUANT_WRAPPER = 'L{}_BQ({: 2.9f})'

WEIGHTS_BASIC_WRAPPER = '{}'
BIASES_BASIC_WRAPPER = '{}'


# Test file constants
TEST_FILE_DEFINES = inspect.cleandoc(
    '''
    #ifndef _{project_name_caps}_REF_INOUT_H_
    #define _{project_name_caps}_REF_INOUT_H_
    #include "{project_name}_model.h"
    
    #define QMN(type, fraq, val)   (type)(val * (1u << (fraq)) + ((val >= 0)? 0.5f: -0.5f))
    #define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)
    #define INQ(val)   ((unsigned char)val)
    #define PBQ(val)  (val)
    '''
)

TEST_FILE_DEFINES_INPUT = inspect.cleandoc(
    '''
    #define IN_IMG_{test_sample}_SHAPE {{{shape0},{shape1},{shape2}}}
    #define IN_IMG_{test_sample}_RANK ({rank})
    #define IN_IMG_{test_sample} {{\\
    '''
)

TEST_FILE_DEFINES_OUTPUT = inspect.cleandoc(
    '''
    #define OUT_PROB_{test_sample}_SHAPE {{{shape0}}}
    #define OUT_PROB_{test_sample}_RANK ({rank})
    #define OUT_PROB_{test_sample} {{\\
    
    '''
)

TEST_FILE_INPUT_QUANT_WRAPPER = 'INQ({: 2.9f})'
TEST_FILE_OUTPUT_QUANT_WRAPPER = 'PBQ({: 2.9f})'
TEST_FILE_FOOTER = '#endif // _{}_REF_INOUT_H_'


# Constants.h file constants
CONSTANTS_CONVOLUTION_WEIGHTS = 'extern const w_type _W{type} L{layer_idx}_conv_wt_buf[];'
CONSTANTS_DENSE_WEIGHTS = 'extern const w_type _W{type} L{layer_idx}_fc_wt_buf[];'
CONSTANTS_CONVOLUTION_BIASES = 'extern const w_type _W{type} L{layer_idx}_conv_bias_buf[];'
CONSTANTS_DENSE_BIASES = 'extern const w_type _W{type} L{layer_idx}_fc_bias_buf[];'

CONSTANTS_FILE_GUARD_BEG = inspect.cleandoc(
    '''
    #ifndef _{project_name_caps}_CONSTANTS_H_
    #define _{project_name_caps}_CONSTANTS_H_
    '''
)

CONSTANTS_FILE_INCLUDES = inspect.cleandoc(
    '''
    #include "mli_config.h"
    #include "{project_name}_model.h"
    '''
)

CONSTANTS_W_EL_TYPE_8 = inspect.cleandoc(
    '''
    #define W_EL_TYPE (MLI_EL_FX_8)
    typedef int8_t w_type;
    '''
)

CONSTANTS_W_EL_TYPE_16 = inspect.cleandoc(
    '''
    #define W_EL_TYPE (MLI_EL_FX_16)
    typedef int16_t w_type;
    '''
)

CONSTANTS_FILE_DEFINES = inspect.cleandoc(
    '''
    // Defining data sections attributes
    //===================================
    #if (ARC_PLATFORM == V2DSP_XY)
    #if defined (__GNUC__) && !defined (__CCAC__)
    // ARC GNU tools
    // Model Weights attribute
    #define _Wdata_attr __attribute__((section(".mli_model")))
    #define _W  _Wdata_attr

    // Model Weights (part 2) attribute
    #define _W2data_attr __attribute__((section(".mli_model_p2")))
    #define _W2  _W2data_attr

    // Bank X (XCCM) attribute
    #define __Xdata_attr __attribute__((section(".Xdata")))
    #define _X  __Xdata_attr

    // Bank Y (YCCM) attribute
    #define __Ydata_attr __attribute__((section(".Ydata")))
    #define _Y  __Ydata_attr

    // Bank Z (DCCM) attribute
    #define __Zdata_attr __attribute__((section(".Zdata")))
    #define _Z  __Zdata_attr

    #else
    // Metaware tools
    // Model Weights attribute
    #define _Wdata_attr __attribute__((section(".mli_model")))
    #define _W __xy _Wdata_attr

    // Model Weights (part 2) attribute
    #define _W2data_attr __attribute__((section(".mli_model_p2")))
    #define _W2 __xy _W2data_attr

    // Bank X (XCCM) attribute
    #define __Xdata_attr __attribute__((section(".Xdata")))
    #define _X __xy __Xdata_attr

    // Bank Y (YCCM) attribute
    #define __Ydata_attr __attribute__((section(".Ydata")))
    #define _Y __xy __Ydata_attr

    // Bank Z (DCCM) attribute
    #define __Zdata_attr __attribute__((section(".Zdata")))
    #define _Z __xy __Zdata_attr
    #endif // if defined (__GNUC__) && !defined (__CCAC__)

    #else
    #define _X __attribute__((section(".mli_ir_buf")))
    #define _Y __attribute__((section(".mli_ir_buf")))
    #define _Z __attribute__((section(".mli_ir_buf")))
    #define _W __attribute__((section(".mli_model")))
    #define _W2 __attribute__((section(".mli_model")))
    #endif

    //======================================================
    //
    // Common data transform (Qmn) defines
    //
    //======================================================

    #define QMN(type, fraq, val)   (type)(val * (1u << (fraq)) + ((val >= 0)? 0.5f: -0.5f))
    #define FRQ_BITS(int_part, el_type) ((sizeof(el_type)*8)-int_part-1)

    //======================================================
    //
    // Common data transform (Qmn) defines
    //
    //======================================================
    '''
)

CONSTANTS_CONV_DEF = inspect.cleandoc(
    '''
    #define CONV{layer_idx}_W_INT   ({w_bit})
    #define CONV{layer_idx}_B_INT   ({b_bit})
    #define CONV{layer_idx}_OUT_INT ({o_bit})
    '''
)

CONSTANTS_DENSE_DEF = inspect.cleandoc(
    '''
    #define FC{layer_idx}_W_INT   ({w_bit})
    #define FC{layer_idx}_B_INT   ({b_bit})
    #define FC{layer_idx}_OUT_INT ({o_bit})
    '''
)

CONSTANTS_CONV_SHAPES = inspect.cleandoc(
    '''
    // CONV{layer_idx}
    //================================================
    #define CONV{layer_idx}_W_SHAPE {{{s1},{s2},{s3},{s4}}}
    #define CONV{layer_idx}_W_ELEMENTS ({s1}*{s2}*{s3}*{s4})
    #define CONV{layer_idx}_W_RANK (4)
    
    #define CONV{layer_idx}_W_FRAQ   (FRQ_BITS(CONV{layer_idx}_W_INT, w_type))
    #define L{layer_idx}_WQ(val)   QMN(w_type, CONV{layer_idx}_W_FRAQ, val)
    
    #define CONV{layer_idx}_B_ELEMENTS ({s1})
    #define CONV{layer_idx}_B_SHAPE {{{s1}}}
    #define CONV{layer_idx}_B_RANK (1)
    
    #define CONV{layer_idx}_B_FRAQ   (FRQ_BITS(CONV{layer_idx}_B_INT, w_type))
    #define L{layer_idx}_BQ(val)   QMN(w_type, CONV{layer_idx}_B_FRAQ, val)
    
    #define CONV{layer_idx}_OUT_FRAQ (FRQ_BITS(CONV{layer_idx}_OUT_INT, d_type))
    '''
)

CONSTANTS_DENSE_SHAPES = inspect.cleandoc(
    '''
    // FC{layer_idx}
    //================================================
    #define FC{layer_idx}_W_SHAPE {{{s1},{s2}}}
    #define FC{layer_idx}_W_ELEMENTS ({s1}*{s2})
    #define FC{layer_idx}_W_RANK (2)
    
    #define FC{layer_idx}_W_FRAQ   (FRQ_BITS(FC{layer_idx}_W_INT, w_type))
    #define L{layer_idx}_WQ(val)   QMN(w_type, FC{layer_idx}_W_FRAQ, val)
    
    #define FC{layer_idx}_B_ELEMENTS ({s1})
    #define FC{layer_idx}_B_SHAPE {{{s1}}}
    #define FC{layer_idx}_B_RANK (1)
    
    #define FC{layer_idx}_B_FRAQ   (FRQ_BITS(FC{layer_idx}_B_INT, w_type))
    #define L{layer_idx}_BQ(val)   QMN(w_type, FC{layer_idx}_B_FRAQ, val)
    
    #define FC{layer_idx}_OUT_FRAQ (FRQ_BITS(FC{layer_idx}_OUT_INT, d_type))
    '''
)

CONSTANTS_FILE_GUARD_END = '#endif  //_{project_name_caps}_CONSTANTS_H_'


# Model.h constants
MODEL_H_GUARD_BEG = inspect.cleandoc(
    '''
    #ifndef _{project_name_caps}_MODEL_H_
    #define _{project_name_caps}_MODEL_H_
    '''
)

MODEL_H_FILE_INCLUDES = inspect.cleandoc(
    '''
    #include <stdint.h>
    #include "mli_types.h"
    '''
)

MODEL_H_FILE_DEFINES = inspect.cleandoc(
    '''
    //=============================================
    //
    // Model interface
    //
    //=============================================
    // Input tensor. To be filled with input image by user befor calling inference function (cifar10_cf_net).
    #define IN_POINTS ({is1} * {is2} * {is3})
    extern mli_tensor * const {project_name}_cf_net_input;
    
    // Output tensor for model. Will be filled with probabilities vector by model
    #define OUT_POINTS ({os1})
    extern mli_tensor * const {project_name}_cf_net_output;
    
    extern char const letters[OUT_POINTS]; 
    
    // Model inference function
    //
    // Get input data from cifar10_cf_net_input tensor (FX format), fed it to the neural network,
    // and writes results to cifar10_cf_net_output tensor (FX format). It is user responsibility
    // to prepare input tensor correctly before calling this function and get result from output tensor
    // after function finish
    //
    // params:
    // debug_ir_root -  Path to intermediate vectors prepared in IDX format (hardcoded names). 
    //                  Provides opportunity to analyse intermediate results in terms of 
    //                  similarity with reference data. If path is incorrect it outputs only profiling data
    //                  If NULL is passed, no messages will be printed in inference
    void {project_name}_cf_net();

    void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs);
    void all_pred(float *pred_data);
    '''  
) 

MODEL_H_FILE_DEFINES_DBG = inspect.cleandoc(
    '''
    //=============================================
    //
    // Model interface
    //
    //=============================================
    // Input tensor. To be filled with input image by user befor calling inference function (cifar10_cf_net).
    #define IN_POINTS ({is1} * {is2} * {is3})
    extern mli_tensor * const {project_name}_cf_net_input;
    
    // Output tensor for model. Will be filled with probabilities vector by model
    #define OUT_POINTS ({os1})
    extern mli_tensor * const {project_name}_cf_net_output;
    
    extern char const letters[OUT_POINTS]; 
    
    // Model inference function
    //
    // Get input data from cifar10_cf_net_input tensor (FX format), fed it to the neural network,
    // and writes results to cifar10_cf_net_output tensor (FX format). It is user responsibility
    // to prepare input tensor correctly before calling this function and get result from output tensor
    // after function finish
    //
    // params:
    // debug_ir_root -  Path to intermediate vectors prepared in IDX format (hardcoded names). 
    //                  Provides opportunity to analyse intermediate results in terms of 
    //                  similarity with reference data. If path is incorrect it outputs only profiling data
    //                  If NULL is passed, no messages will be printed in inference
    void {project_name}_cf_net();

    void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs);
    void all_pred(float *pred_data);
    '''  
) 

MODEL_H_D_TYPE_8 = inspect.cleandoc(
    '''
    typedef int8_t d_type;
    '''
)

MODEL_H_D_TYPE_16 = inspect.cleandoc(
    '''
    typedef int16_t d_type;
    '''
)

MODEL_H_GUARD_END = '\n#endif  //_{project_name_caps}_MODEL_H_'


# Model.c constants
MODEL_C_FILE_INCLUDES = inspect.cleandoc(
    '''
    #include "{project_name}_model.h"

    #include <stdint.h>
    #include <stdio.h>
    #include <string.h>
    #include <assert.h>
    
    #include "mli_api.h"
    #include "mli_types.h"
    #include "mli_config.h"
    
    #include "{project_name}_constants.h"
    '''
)

MODEL_C_FILE_INCLUDES_DBG = inspect.cleandoc(
    '''
    #include "{project_name}_model.h"

    #include <stdint.h>
    #include <stdio.h>
    #include <string.h>
    #include <assert.h>
    
    #include "mli_api.h"
    #include "mli_types.h"
    #include "mli_config.h"
    
    #include "{project_name}_constants.h"
    #include "tests_aux.h"
    #include "tensor_transform.h"
    '''
)

MODEL_C_D_EL_TYPE_8 = inspect.cleandoc(
    '''
    #define D_EL_TYPE (MLI_EL_FX_8)
    '''
)

MODEL_C_D_EL_TYPE_16 = inspect.cleandoc(
    '''
    #define D_EL_TYPE (MLI_EL_FX_16)
    '''
)

MODEL_C_FILE_DEFINES = inspect.cleandoc(
    '''
    //==============================================================
    //
    //
    // Data related to the Module
    //
    //
    //==============================================================
    
    const char debug_ir_root[] = "model/idx";
    
    // Intermediate data buffers (enough size for max intermediate results)
    //==============================
    #define IR_BUF_SZ_MOST ({most_buf_s0}*{most_buf_s1}*{most_buf_s2})
    #define IR_BUF_SZ_NEXT ({next_buf_s0}*{next_buf_s1}*{next_buf_s2})
    
    #pragma Data(".nn_ir_data_1")
    static d_type  x_mem_buf[IR_BUF_SZ_MOST];
    #pragma Data()
    
    #pragma Data(".nn_ir_data_2")
    static d_type  y_mem_buf[IR_BUF_SZ_NEXT];
    #pragma Data()
    '''
)

MODEL_C_IO_TENSORS = inspect.cleandoc(
    '''
    // Module Input/Output tensors and their's external interface
    //============================================================
    static mli_tensor input = {{
        .data = (void *)x_mem_buf,
        .capacity = sizeof(d_type) * IN_POINTS,
        .shape = {{{is0}, {is1}, {is2}}},
        .rank = 3,
        .el_type = D_EL_TYPE,
        .el_params.fx.frac_bits = {in_frac_bits},
    }};
    static mli_tensor output = {{
        .data = (void *)y_mem_buf,
        .capacity = sizeof(d_type) * OUT_POINTS,
        .shape = {{{os0}}},
        .rank = 1,
        .el_type = D_EL_TYPE,
        .el_params.fx.frac_bits = {out_frac_bits}, 
    }};
    
    // Interface variables: Available to user via main model header
    //===========================================================
    mli_tensor * const {project_name}_cf_net_input = &input;
    mli_tensor * const {project_name}_cf_net_output = &output;

    char const letters[26] = {{'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}};
    
    
    //==============================================================
    //  Model description and configuration
    //==============================================================
    #pragma Data(".mli_data")
    
    // Configuration objects for layers
    //===============================================
    
    static const mli_permute_cfg permute_hwc2chw_cfg = {{
            .perm_dim = {{2, 0, 1}} // 2 0 1
    }};
    '''
)

MODEL_C_CONV_CFG = inspect.cleandoc(
    '''
    static const mli_conv2d_cfg shared_conv_cfg = {{
        .stride_height = {stride_height}, .stride_width = {stride_width},
        .padding_bottom = {padding_bottom}, .padding_top = {padding_top},
        .padding_left = {padding_left}, .padding_right = {padding_right},
        .relu.type = MLI_RELU_GEN
    }};
    '''
)

MODEL_C_POOL_CFG = inspect.cleandoc(
    '''
    static const mli_pool_cfg shared_pool_cfg = {{
        .kernel_height = {kernel_height}, .kernel_width = {kernel_width},
        .stride_height = {stride_height}, .stride_width = {stride_width},
        .padding_bottom = {padding_bottom}, .padding_top = {padding_top},
        .padding_left = {padding_left}, .padding_right = {padding_right}
    }};
    '''
)

MODEL_C_CONV_TENSOR = inspect.cleandoc(
    '''
    static const mli_tensor L{layer_idx}_conv_wt = {{
        .data = (void *)L{layer_idx}_conv_wt_buf,
        .capacity = CONV{layer_idx}_W_ELEMENTS * sizeof(w_type),
        .shape = CONV{layer_idx}_W_SHAPE,
        .rank = CONV{layer_idx}_W_RANK,
        .el_type = W_EL_TYPE,
        .el_params.fx.frac_bits = CONV{layer_idx}_W_FRAQ,
    }};
    
    static const mli_tensor L{layer_idx}_conv_bias = {{
        .data = (void *)L{layer_idx}_conv_bias_buf,
        .capacity = CONV{layer_idx}_B_ELEMENTS * sizeof(w_type),
        .shape = CONV{layer_idx}_B_SHAPE,
        .rank = CONV{layer_idx}_B_RANK,
        .el_type = W_EL_TYPE,
        .el_params.fx.frac_bits = CONV{layer_idx}_B_FRAQ,
    }};
    '''
)

MODEL_C_DENSE_TENSOR = inspect.cleandoc(
    '''
    static mli_tensor L{layer_idx}_fc_wt = {{
        .data = (void *)L{layer_idx}_fc_wt_buf,
        .capacity = FC{layer_idx}_W_ELEMENTS * sizeof(w_type),
        .shape = FC{layer_idx}_W_SHAPE,
        .rank = FC{layer_idx}_W_RANK,
        .el_type = W_EL_TYPE,
        .el_params.fx.frac_bits = FC{layer_idx}_W_FRAQ,
    }};
    
    static mli_tensor L{layer_idx}_fc_bias = {{
        .data = (void *)L{layer_idx}_fc_bias_buf,
        .capacity = FC{layer_idx}_B_ELEMENTS * sizeof(w_type),
        .shape = FC{layer_idx}_B_SHAPE,
        .rank = FC{layer_idx}_B_RANK,
        .el_type = W_EL_TYPE,
        .el_params.fx.frac_bits = FC{layer_idx}_B_FRAQ,
    }};
    '''
)

MODEL_C_IR_TENSORS = inspect.cleandoc(
    '''
    // Intermediate result tensors
    //===============================================
    static mli_tensor ir_tensor_X = {
        .data = (void *)x_mem_buf,
        .capacity = sizeof(x_mem_buf),
        .shape = {0, 0, 0, 0},
        .rank = 4,
        .el_type = D_EL_TYPE,
        .el_params.fx.frac_bits = FRQ_BITS(0, d_type),
    };
    
    static mli_tensor ir_tensor_Y = {
        .data = (void *)y_mem_buf,
        .capacity = sizeof(y_mem_buf),
        .shape = {0, 0, 0, 0},
        .rank = 4,
        .el_type = D_EL_TYPE,
        .el_params.fx.frac_bits = FRQ_BITS(0, d_type),
    };
    
    #pragma Data()
    '''
)

MODEL_C_FUNC_DECL_DBG = inspect.cleandoc(
    '''
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out);
    
    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out);
    
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out);
    static const mli_relu_cfg relu_cfg = {.type = MLI_RELU_GEN};
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out);
    
    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out);
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor  *weights,
            const mli_tensor  *bias,
            mli_tensor *out);
    
    //  Check kernel result. Debug function
    //==============================================================
    static void check_result(
            const char * ir_root,
            const char * ref_file,
            mli_tensor *pred_tsr,
            unsigned cycles,
            mli_status ret_code);
    '''
)

MODEL_C_FUNC_DECL = inspect.cleandoc(
    '''
    static void emnist_preprocessing();

    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out);
    
    static inline mli_status softmax(const mli_tensor *in, mli_tensor *out);
    
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out);
    static const mli_relu_cfg relu_cfg = {.type = MLI_RELU_GEN};
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out);
    
    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out);
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor  *weights,
            const mli_tensor  *bias,
            mli_tensor *out);
    '''
)

MODEL_C_NETWORK = inspect.cleandoc(
    '''
    //==============================================================
    //
    //  EMNIST graph based on Keras example.
    //  Layer-by-Layer execution for CHW layput
    //
    //==============================================================
    void {project_name}_cf_net() {{
    '''
)

MODEL_C_NETWORK_END = inspect.cleandoc(
    '''
    }
    '''
)

MODEL_C_NETWORK_DBG = inspect.cleandoc(
    '''
    void {project_name}_cf_net() {{
    '''
)

MODEL_C_NETWORK_PREPROCESS = '''
        preprocessing(&input);
    '''

MODEL_C_PERMUTE = '''
        mli_krn_permute_fx(&input, &permute_hwc2chw_cfg, &ir_tensor_Y);
    '''

MODEL_C_NETWORK_CONV = '''
        ir_tensor_{out_tensor}.el_params.fx.frac_bits = CONV{layer_idx}_OUT_FRAQ;
        conv2d_chw(&ir_tensor_{in_tensor}, &L{layer_idx}_conv_wt, &L{layer_idx}_conv_bias, &shared_conv_cfg, &ir_tensor_{out_tensor});
    '''


MODEL_C_NETWORK_CONV_DBG = '''
        ir_tensor_{out_tensor}.el_params.fx.frac_bits = CONV{layer_idx}_OUT_FRAQ;
        ret = conv2d_chw(&ir_tensor_{in_tensor}, &L{layer_idx}_conv_wt, &L{layer_idx}_conv_bias, &shared_conv_cfg, &ir_tensor_{out_tensor});
        check_result(debug_ir_root, "ir_acti{layer_idx}.idx", &ir_tensor_{out_tensor}, 0, ret);
    '''


MODEL_C_NETWORK_MAXPOOL = '''
        maxpool_chw(&ir_tensor_{in_tensor}, &shared_pool_cfg, &ir_tensor_{out_tensor});
    '''


MODEL_C_NETWORK_MAXPOOL_DBG = '''
        ret = maxpool_chw(&ir_tensor_{in_tensor}, &shared_pool_cfg, &ir_tensor_{out_tensor});
        check_result(debug_ir_root, "ir_pool{layer_idx}.idx", &ir_tensor_{out_tensor}, 0, ret);
    '''


MODEL_C_NETWORK_DENSE = '''
        ir_tensor_{out_tensor}.el_params.fx.frac_bits = FC5_OUT_FRAQ;
        fully_connected(&ir_tensor_{in_tensor}, &L{layer_idx}_fc_wt, &L{layer_idx}_fc_bias, &ir_tensor_{out_tensor});
    '''


MODEL_C_NETWORK_DENSE_DBG = '''
        ir_tensor_{out_tensor}.el_params.fx.frac_bits = FC{layer_idx}_OUT_FRAQ;
        ret = fully_connected(&ir_tensor_{in_tensor}, &L{layer_idx}_fc_wt, &L{layer_idx}_fc_bias, &ir_tensor_{out_tensor});
        check_result(debug_ir_root, "ir_dense{layer_idx}.idx", &ir_tensor_{out_tensor}, 0, ret);
    '''

MODEL_C_NETWORK_RELU = '''
        relu(&ir_tensor_{in_tensor}, &relu_cfg, &ir_tensor_{out_tensor});
    '''


MODEL_C_NETWORK_RELU_DBG = '''
        ret = relu(&ir_tensor_{in_tensor}, &relu_cfg, &ir_tensor_{out_tensor});
        check_result(debug_ir_root, "ir_acti{layer_idx}.idx", &ir_tensor_{out_tensor}, 0, ret);
    '''

MODEL_C_NETWORK_SOFTMAX = '''
        softmax(&ir_tensor_{in_tensor}, &output);
    '''

MODEL_C_NETWORK_SOFTMAX_DBG = '''
        ret = softmax(&ir_tensor_{in_tensor}, &output);
        check_result(debug_ir_root, "ir_acti{layer_idx}.idx", &output, 0, ret);  
    '''

MODEL_C_NETWORK_CYCLES_DBG = '''
        unsigned cycles = 0;
    '''

MODEL_C_PERMUTE_DBG = '''
        mli_status ret = MLI_STATUS_OK;
        ret = mli_krn_permute_fx(&input, &permute_hwc2chw_cfg, &ir_tensor_Y);
        cycles += cycle_cnt;
'''

MODEL_C_PRINT_CYCLES_DBG = '''
        printf("\\n\\n\\tTotal: %u cycles\\n\\n", cycles);
'''

MODEL_C_NETWORK_CLOSE = '''
        }
    }
    '''

MODEL_C_FUNC_CHECK_RESULT = inspect.cleandoc(
    '''
    static void check_result(
        const char * ir_root,
        const char * ref_file,
        mli_tensor *pred_tsr,
        unsigned cycles,
        mli_status ret_code) {
        if (ret_code != MLI_STATUS_OK) {
            printf("ERROR: MLI Code for %s (%d) is not OK\\n", ref_file, ret_code);
            assert(0);
        }

        if (ir_root != NULL) {
            ref_to_pred_output err;
            test_status test_result = measure_ref_to_pred(ir_root, ref_file, *pred_tsr, &err);
            if (test_result == TEST_PASSED) {
                printf("%s: \\n\\tS/N=%-10.1f (%-4.1f db)\\n\\tMAX_ABS_ERROR=%-10.1f\\n",
                        ref_file,
                        err.ref_vec_length / err.noise_vec_length,
                        err.ref_to_noise_snr,
                        err.max_abs_err);
            }
            else if (test_result == TEST_FAILED) {
                printf("ERROR: Test suit returns FAILD code for %s\\n", ref_file);
                assert(0);
            } else {
                printf("%s(w/o IR check):\\t%u cycles\\n", ref_file, cycles);
            }
        }
    }
    '''
)

#MODEL_C_FUNC_RUN_INFERENCE = inspect.cleandoc(
#    '''
#    void run_{project_name}_cf_net() {{
#        PROFILE({project_name}_cf_net());
#        printf("\\n\\n\\tTotal: %u cycles \\n\\n", cycle_cnt);
#    }}
#    '''
#)

MODEL_C_FUNC_DEFS_8_SPEC = inspect.cleandoc(
    '''
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {{
        return mli_krn_maxpool_chw_fx8_k{pool_dim1}x{pool_dim1}_krnpad(in, cfg, out);
    }}
    
    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {{
        return mli_krn_softmax_fx8(in, out);
    }}
    
    static const mli_relu_cfg relu_cfg = {{.type = MLI_RELU_GEN}};
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {{
        return mli_krn_relu_fx8(in, cfg, out);
    }}
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {{
        return mli_krn_permute_fx8(in, cfg, out);
    }}


    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out) {{
        return mli_krn_conv2d_chw_fx8_k{kernel_dim1}x{kernel_dim2}_str{kernel_stride}_krnpad(in, weights, bias, cfg, out);
    }}
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            mli_tensor *out) {{
        return mli_krn_fully_connected_fx8(in, weights, bias, out);
    }}
    '''
)

MODEL_C_FUNC_DEFS_16_SPEC = inspect.cleandoc(
    '''
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {{
        return mli_krn_maxpool_chw_fx16_k{pool_dim1}x{pool_dim2}_krnpad(in, cfg, out); // TODO check if not 3x3
    }}

    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {{
        return mli_krn_softmax_fx16(in, out);
    }}
    
    static const mli_relu_cfg relu_cfg = {{.type = MLI_RELU_GEN}};
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {{
        return mli_krn_relu_fx16(in, cfg, out);
    }}
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {{
        return mli_krn_permute_fx16(in, cfg, out);
    }}
 
    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out) {{
        return mli_krn_conv2d_chw_fx16_k{kernel_dim1}x{kernel_dim2}_str{kernel_stride}_krnpad(in, weights, bias, cfg, out);
    }}
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            mli_tensor *out) {{
        return mli_krn_fully_connected_fx16(in, weights, bias, out);
    }}
    '''
)

MODEL_C_FUNC_DEFS_816_SPEC = inspect.cleandoc(
    '''
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {{
        return mli_krn_maxpool_chw_fx16_k{pool_dim1}x{pool_dim1}_krnpad(in, cfg, out);
    }}
    
    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {{
        return mli_krn_softmax_fx16(in, out);
    }}
    
    static const mli_relu_cfg relu_cfg = {{.type = MLI_RELU_GEN}};
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {{
        return mli_krn_relu_fx16(in, cfg, out);
    }}
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {{
        return mli_krn_permute_fx16(in, cfg, out);
    }}

    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out) {{
        return mli_krn_conv2d_chw_fx8w16d(in, weights, bias, cfg, out);
    }}
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            mli_tensor *out) {{
        return mli_krn_fully_connected_fx8w16d(in, weights, bias, out);
    }}
    '''
)

MODEL_C_FUNC_DEFS_8_GEN = inspect.cleandoc(
    '''
    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {{
        return mli_krn_softmax_fx8(in, out);
    }}
    
    static const mli_relu_cfg relu_cfg = {{.type = MLI_RELU_GEN}};
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {{
        return mli_krn_relu_fx8(in, cfg, out);
    }}
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {{
        return mli_krn_permute_fx8(in, cfg, out);
    }}
    
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {{
        /* GENERIC VERSION OF KERNEL IS USED BELOW. REPLACE IT WITH SPECIALIZED ONE FOR BETTER PERFORMANCE. */
        return mli_krn_maxpool_chw_fx8_generic(in, cfg, out);
    }}


    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out) {{
        /* GENERIC VERSION OF KERNEL IS USED BELOW. REPLACE IT WITH SPECIALIZED ONE FOR BETTER PERFORMANCE. */
        return mli_krn_conv2d_chw_fx8_generic(in, weights, bias, cfg, out);
    }}
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            mli_tensor *out) {{
        return mli_krn_fully_connected_fx8(in, weights, bias, out);
    }}
    '''
)

MODEL_C_FUNC_DEFS_16_GEN = inspect.cleandoc(
    '''
    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {{
        return mli_krn_softmax_fx16(in, out);
    }}
    
    static const mli_relu_cfg relu_cfg = {{.type = MLI_RELU_GEN}};
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {{
        return mli_krn_relu_fx16(in, cfg, out);
    }}
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {{
        return mli_krn_permute_fx16(in, cfg, out);
    }}
    
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {{
        return mli_krn_maxpool_chw_fx16_generic(in, cfg, out);
    }}


    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out) {{
        return mli_krn_conv2d_chw_fx16_generic(in, weights, bias, cfg, out);
    }}
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            mli_tensor *out) {{
        return mli_krn_fully_connected_fx16(in, weights, bias, out);
    }}
    '''
)

MODEL_C_FUNC_DEFS_816_GEN = inspect.cleandoc(
    '''
    static inline mli_status softmax(const mli_tensor *in,  mli_tensor *out) {{
        return mli_krn_softmax_fx16(in, out);
    }}
    
    static const mli_relu_cfg relu_cfg = {{.type = MLI_RELU_GEN}};
    static inline mli_status relu(const mli_tensor *in, const mli_relu_cfg *cfg, mli_tensor *out) {{
        return mli_krn_relu_fx16(in, cfg, out);
    }}
    
    static inline mli_status mli_krn_permute_fx(const mli_tensor *in, const mli_permute_cfg *cfg, mli_tensor *out) {{
        return mli_krn_permute_fx16(in, cfg, out);
    }}
    
    static inline mli_status maxpool_chw(const mli_tensor *in, const mli_pool_cfg *cfg, mli_tensor *out) {{
        return mli_krn_maxpool_chw_fx16_generic(in, cfg, out);
    }}


    static inline mli_status conv2d_chw(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            const mli_conv2d_cfg *cfg,
            mli_tensor *out) {{
        return mli_krn_conv2d_chw_fx8w16d_generic(in, weights, bias, cfg, out);
    }}
    
    static inline mli_status fully_connected(
            const mli_tensor *in,
            const mli_tensor *weights,
            const mli_tensor *bias,
            mli_tensor *out) {{
        return mli_krn_fully_connected_fx8w16d(in, weights, bias, out);
    }}
    '''
)

MODEL_C_AUX_FUNC = inspect.cleandoc(
    '''
    static void tensor_to_float (const mli_tensor * src, float *dst, uint32_t dst_size) {{
        const float scale_val = 1.0f / (float) (1u << (src->el_params.fx.frac_bits));
        if (src->el_type == MLI_EL_FX_16) {{
            int16_t *src_arr = src->data;
            for (int idx = 0; idx < dst_size; idx++)
                dst[idx] = (float) (scale_val * src_arr[idx]);
        }} else {{
            int8_t *src_arr = src->data;
            for (int idx = 0; idx < dst_size; idx++)
                dst[idx] = (float) (scale_val * src_arr[idx]);
        }}
    }}


    void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs) {{
        uint8_t flags[OUT_POINTS] = {{0}};
        float pred_data[OUT_POINTS] = {{0}};
        //d_type* const out = (d_type * const){project_name}_cf_net_output->data;
        tensor_to_float({project_name}_cf_net_output, pred_data, OUT_POINTS);
        for (int top = 0; top < n; top++) {{
            float max = -1;
            uint8_t max_idx = -1;
            
            for (int idx = 0; idx < OUT_POINTS; idx++) {{
                if(pred_data[idx] > max && flags[idx] != 1) {{
                    max = pred_data[idx];
                    max_idx = idx;
                }}
            }}
    
            top_letters[top] = letters[max_idx];
            top_letters_probs[top] = pred_data[max_idx];
            flags[max_idx] = 1;
        }}
    }}


    void all_pred(float *pred_data) {{
        tensor_to_float({project_name}_cf_net_output, pred_data, OUT_POINTS);
    }}
    '''
)
MODEL_C_TOP_N_FUNC = inspect.cleandoc(
    '''
    void top_n_pred(int8_t n, char *top_letters, float *top_letters_probs) {{
        uint8_t flags[OUT_POINTS] = {{0}};
        float pred_data[OUT_POINTS] = {{0}};
        //d_type* const out = (d_type * const){project_name}_cf_net_output->data;
        tensor_to_float({project_name}_cf_net_output, pred_data, OUT_POINTS);
        for (int top = 0; top < n; top++) {{
            float max = -1;
            uint8_t max_idx = -1;
            
            for (int idx = 0; idx < OUT_POINTS; idx++) {{
                if(pred_data[idx] > max && flags[idx] != 1) {{
                    max = pred_data[idx];
                    max_idx = idx;
                }}
            }}
    
            top_letters[top] = letters[max_idx];
            top_letters_probs[top] = pred_data[max_idx];
            flags[max_idx] = 1;
        }}
    }}
    '''
)

MODEL_C_ALL_PRED_FUNC = inspect.cleandoc(
    '''
    void all_pred(float *pred_data) {{
        tensor_to_float({project_name}_cf_net_output, pred_data, OUT_POINTS);
    }}
    '''
)

MODEL_C_CONVERT_FLOAT_FUNC = inspect.cleandoc(
    '''
    static void tensor_to_float (const mli_tensor * src, float *dst, uint32_t dst_size) {
        const float scale_val = 1.0f / (float) (1u << (src->el_params.fx.frac_bits));
        if (src->el_type == MLI_EL_FX_16) {
            int16_t *src_arr = src->data;
            for (int idx = 0; idx < dst_size; idx++)
                dst[idx] = (float) (scale_val * src_arr[idx]);
        } else {
            int8_t *src_arr = src->data;
            for (int idx = 0; idx < dst_size; idx++)
                dst[idx] = (float) (scale_val * src_arr[idx]);
        }
    }
    '''
)


# Main file constants
ML_API_MAIN_FILE_INCLUDES = inspect.cleandoc(
    '''
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <assert.h>
    
    #include "mli_types.h"
    
    #include "{project_name}_ref_inout.h"
    #include "{project_name}_model.h"
    #include "examples_aux.h"
    #include "tests_aux.h"
    '''
)

ML_API_FILE_MAIN = inspect.cleandoc(
    '''
    #include <stdint.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <assert.h>
    
    #include "mli_types.h"
    
    #include "{project_name}_ref_inout.h"
    #include "{project_name}_model.h"
    #include "examples_aux.h"
    #include "tests_aux.h"

    #if defined (__GNUC__) && !defined (__CCAC__)
    extern int start_init(void);
    #endif // if defined (__GNUC__) && !defined (__CCAC__)
    static const char kOutFilePostfix[] = "_out";
    
    const unsigned char kSingleIn[IN_POINTS] = IN_IMG_{test_sample};
    const float kSingleOutRef[OUT_POINTS] = OUT_PROB_{test_sample};
    
    static void {project_name}_preprocessing(const void * image_, mli_tensor * net_input_);
    
    void run_{project_name}_cf_net();
    
    #define EXAMPLE_MAX_MODE (3)
    int mode=3; // emulation argc for GNU toolchain
    char param[EXAMPLE_MAX_MODE][256];// emulation argv for GNU toolchain
    //========================================================================================
    //
    // MAIN
    //
    //========================================================================================
    int main(int argc, char ** argv ) {{
    #if defined (__GNUC__) && !defined (__CCAC__)
    //ARC GNU tools
        if (0 != start_init() ){{
            printf("ERROR: init proccesor\\n");
            //Error init proccesor;
            return 1;
        }}
    //fill mode and param from cmd line script before use
    
    #else
    //Metaware tools
    //fill mode and param from argc and argv
        if ( argc <= EXAMPLE_MAX_MODE) {{
            mode=argc;
        
            for(int i=0; i < mode; i++) {{
                memcpy( &param[i][0], argv[i], strlen(argv[i]) );
            }}
        }}        
    #endif // if defined (__GNUC__) && !defined (__CCAC__)
       
        //checking that variables are set
        if(mode == 0){{
            printf("ERROR: mode not set up\\n");
    #if defined (__GNUC__) && !defined (__CCAC__)        
    //ARC GNU tools
            printf("Please set up mode \\n");
            printf("Please check that you use mdb_com_gnu script with correct setups\\n");
    #else
    //Metaware tools    
            printf("App command line:\\n"
                    "\\t%s \\n\\t\\tProcess single hardcoded vector\\n\\n"
                    "\\t%s <input_test_base.idx> \\n\\t\\tProcess testset from file and \\n"
                    "\\t\\t output model results to <input_test_base.idx_out> file\\n\\n", argv[0], argv[0]);
    #endif // if defined (__GNUC__) && !defined (__CCAC__)                 
            return 2; //Error: mode not set       
        }}  
        
        for(int i=0; i < mode; i++) {{
           if (param[i][0] == 0){{
               printf("param[%d][0] not set.\\n", i);
               if (i==0) printf("Please set up dummy string for check.\\n");
               if (i==1) printf("Please set up input IDX file.\\n");
               if (i==2) printf("Please set up labels IDX file.\\n");
               return 2; //Error: param not set
           }}
        }}
    
        switch (mode) {{
        // No Arguments for app. Process single hardcoded input
        // Print various measures to stdout
        //=========================================================
        case 1:
            printf("HARDCODED INPUT PROCESSING\\n");
            model_run_single_in(kSingleIn, kSingleOutRef,
                    {project_name}_cf_net_input, {project_name}_cf_net_output,
                    {project_name}_preprocessing, run_{project_name}_cf_net, "NN Performance");
            break;
    
        // APP <input_test_base.idx>
        // Output vectors will be written to <input_test_base.idx_out> file
        //=================================================================
        case 2:
            printf("Input IDX testset to output IDX set\\n");
            char * out_path = malloc(strlen(param[1]) + strlen(kOutFilePostfix) + 1);
            if (out_path == NULL) {{
                printf("mem allocation failed\\n");
                break;
            }}
            out_path[0] = 0;
            strcat(out_path, param[1]);
            strcat(out_path, kOutFilePostfix);
    
            model_run_idx_base_to_idx_out(param[1], out_path,
                    {project_name}_cf_net_input, {project_name}_cf_net_output,
                    {project_name}_preprocessing, run_{project_name}_cf_net, NULL);
            free(out_path);
            break;
    
        // APP <input_test_base.idx> <input_test_labels.idx>
        // Calculate accuracy of the model
        //=================================================================
        case 3:
            printf("ACCURACY CALCULATION on Input IDX testset according to IDX labels set\\n");
            model_run_acc_on_idx_base(param[1], param[2],
                    {project_name}_cf_net_input, {project_name}_cf_net_output,
                    {project_name}_preprocessing, run_{project_name}_cf_net, NULL);
            break;
    
        // Unknown format
        //=================================================================
        default:
            printf("App command line:\\n"
                    "\\t%s \\n\\t\\tProcess single hardcoded vector\\n\\n"
                    "\\t%s <input_test_base.idx> \\n\\t\\tProcess testset from file and \\n"
                    "\\t\\t output model results to <input_test_base.idx_out> file\\n\\n", argv[0], argv[0]);
            break;
        }}
        printf("FINISHED\\n");
    
        return 0;
    }}
    
    //========================================================================================
    //
    // Other internal functions and routines
    //
    //========================================================================================
    static void {project_name}_preprocessing(const void* image_, mli_tensor* net_input_) {{
        const unsigned char* in = image_;
        d_type* const dst = (d_type * const)net_input_->data;
        for (int idx = 0; idx < IN_POINTS; idx++) {{
            dst[idx] = (d_type)((int)in[idx]);
        }}
    }}
    
    
    void run_{project_name}_cf_net(const char * info_str) {{
        PROFILE({project_name}_cf_net());
        if (info_str != NULL) {{
            printf("\\n%s\\n\\tTotal: %u cycles \\n\\n", info_str, cycle_cnt);
        }}
    }}
    '''
)


# Make file constants
MAKE_FILE_MAIN = inspect.cleandoc(
    '''
    #
    # Copyright 2019-2020, Synopsys, Inc.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-3-Clause license found in
    # the LICENSE file in the root directory of this source tree.
    #

    # default toolchain
    TOOLCHAIN ?= mwdt

    # MLI Library Root Directory
    EMBARC_MLI_DIR ?= ../..

    # default hardware config file
    TCF_FILE  ?= $(EMBARC_MLI_DIR)/hw/em9d.tcf

    # Directories and files
    SRC_DIRS    = . \\
                  ./model\\
                  $(EMBARC_MLI_DIR)/examples/auxiliary

    INC_DIRS    = . \\
                  ./model\\
                  $(EMBARC_MLI_DIR)/include\\
                  $(EMBARC_MLI_DIR)/examples/auxiliary

    EXT_LIBS_DIR ?= $(EMBARC_MLI_DIR)/bin
    EXT_LIBS     ?= $(EMBARC_MLI_DIR)/bin/libmli.a
    OUT_DIR      ?= ./bin
    BUILD_DIR    ?= ./obj
    OUT_NAME     ?= example_emnist_tensorflow
    ifeq ($(TOOLCHAIN),mwdt)
    # MWDT specific options
    CFLAGS       =  -Hnocopyr -Hpurge -Hheap=8K -Hstack=1K -Hfxapi -e_start -Bgrouplib -Hldopt=-q -Hsdata0 -Xdsp_ctrl=postshift,guard,convergent -Hdense_prologue

    # use compact CRT
    CFLAGS      += -Hcl -Hcrt_argv -Hcrt_fast_memcpy -Hcrt_fast_memset -Hxcheck -Hcrt_initbss

    else
    PREBUILT_LIB ?= $(EMBARC_MLI_DIR)/examples/prebuilt/libmli.a

    # GNU toolchain specific options - correct it according to your target platform settings (see build_configuration.txt for input)
    #Iot DevKit config
    CFLAGS       =  -mcpu=em4_fpuda -mlittle-endian -mcode-density -mdiv-rem -mswap -mnorm -mmpy-option=6 -mbarrel-shifter -mxy

    # The embARC MLI Library specific options it according to your target platform settings 
    #(EM5D or EM7D platform)
    #CFLAGS      += -DV2DSP
    #(EM9D or EM11D platform)
    CFLAGS      += -DV2DSP_XY
    #(HS45D or HS47D platform)
    #CFLAGS      += -DV2DSP_WIDE

    # GNU toolchain linker specific options
    LDFLAGS      = --defsym=__DEFAULT_HEAP_SIZE=8k 
    LDFLAGS     += --defsym=__DEFAULT_STACK_SIZE=1k
    LDFLAGS     += -Map $(OUT_DIR)/$(OUT_NAME).map

    #specific options for run the example with the MetaWare Debuger on the nSim simulator. 
    DBG_OPTS     = -cmd="read mdb_com_gnu"
    endif

    .PHONY: clean all lib cleanall app
    .DEFAULT_GOAL := all

    all: lib app 

    $(EXT_LIBS): $(EXT_LIBS_DIR)
        @echo Copy prebuilt library $(PREBUILT_LIB) to $@
        @$(CP) $(call fix_platform_path,$(PREBUILT_LIB)) $(call fix_platform_path,$@)

    $(EXT_LIBS_DIR):
        $(MKDIR) $(call fix_platform_path, $@ )

    include $(EMBARC_MLI_DIR)/build/rules.mk

    ifeq ($(TOOLCHAIN),mwdt)
    lib:
        @ $(MAKE) generic_lib -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS) TCF_FILE="$(TCF_FILE)"
    else
    lib: $(EXT_LIBS)
    endif

    app: generic_app

    run: generic_run

    clean:
        @echo Cleaning application $(OUT_NAME)...
        -@$(RM) $(call fix_platform_path,$(OBJS))

    cleanall: clean
        @echo Cleaning all files ..
        -@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).elf)
        -@$(RM) $(call fix_platform_path,$(OUT_DIR)/$(OUT_NAME).map)
        +$(MAKE) clean -C $(EMBARC_MLI_DIR)$(PS)lib$(PS)make$(PS)
    '''
)


FUNC_PREPROCESS_OUTER = inspect.cleandoc(
    '''
    void {project_name}_preprocessing(const void* image_, mli_tensor* net_input_) {{
        const unsigned char* in = image_;
        d_type* const dst = (d_type * const)net_input_->data;

        if (net_input_->el_params.fx.frac_bits == {frac_bits}) {{
            for (int idx = 0; idx < IN_POINTS; idx++) {{
                dst[idx] = (d_type)((int)in[idx] - {norm_value});
            }}
        }}
        else if (net_input_->el_params.fx.frac_bits > {frac_bits}) {{
            int shift_left = net_input_->el_params.fx.frac_bits - {frac_bits};
            for (int idx = 0; idx < IN_POINTS; idx++) {{
                dst[idx] = (d_type)((int)in[idx] - {norm_value}) << shift_left;
            }}
        }}
        else {{
            int shift_right = {frac_bits} - net_input_->el_params.fx.frac_bits;
            for (int idx = 0; idx < IN_POINTS; idx++) {{
                dst[idx] = (d_type)((int)in[idx] - {norm_value}) >> shift_right;
            }}
        }}
    }}
    '''
)

FUNC_PREPROCESS_INNER = inspect.cleandoc(
    '''    
    static void preprocessing(mli_tensor* net_input_) {{
        d_type* const dst = (d_type * const)net_input_->data;
        if (net_input_->el_params.fx.frac_bits == {frac_bits}) {{
            for (int idx = 0; idx < IN_POINTS; idx++) {{
                dst[idx] = dst[idx] - {norm_value};
            }}
        }}
        else if (net_input_->el_params.fx.frac_bits > {frac_bits}) {{
            int shift_left = net_input_->el_params.fx.frac_bits - {frac_bits};
            for (int idx = 0; idx < IN_POINTS; idx++) {{
                dst[idx] = (dst[idx] - {norm_value}) << shift_left;
            }}
        }}
        else {{
            int shift_right = {frac_bits} - net_input_->el_params.fx.frac_bits;
            for (int idx = 0; idx < IN_POINTS; idx++) {{
                dst[idx] = (dst[idx] - {norm_value}) >> shift_right;
            }}
        }}
    }}
    '''
)