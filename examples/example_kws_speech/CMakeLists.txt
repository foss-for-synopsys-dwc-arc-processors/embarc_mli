#
# Copyright 2020, Synopsys, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-3-Clause license found in
# the LICENSE file in the root directory of this source tree.
#

include(../../cmake/settings.cmake)

file(GLOB temp
    ../auxiliary/*.cc
    ../auxiliary/*.c
)
add_executable(example_kws_speech
    main.cc
    simple_kws_postprocessor.cc
    wav_file_guard.cc
    audio_features/audio_features.cc
    kws/dsconv_lstm_nn/dsconv_lstm_nn_model_data.cc
    kws/dsconv_lstm_nn/dsconv_lstm_nn_model_impl.cc
    ${temp}
)

target_link_libraries(example_kws_speech PUBLIC mli)

target_include_directories(example_kws_speech PRIVATE
    ../../examples/auxiliary
    kws
    audio_features
)

if (ARC)
set(EXAMPLE_FLAGS
    -Hnocopyr
    -Hpurge
    -Hheap=4K
    -Hstack=4K
    -Hdsplib
    -Hfxapi
    -e_start
    -Bgrouplib
    -Hldopt=-q
    -Hsdata0
    -Xdsp_ctrl=postshift,guard,convergent
    -Hdense_prologue
)
else()
set(EXAMPLE_FLAGS)
endif()

target_compile_options(example_kws_speech PRIVATE
    ${MLI_PLATFORM_COMPILE_OPTIONS}
    ${EXAMPLE_FLAGS}
)

target_link_options(example_kws_speech PRIVATE
    ${MLI_PLATFORM_LINK_OPTIONS}
    ${EXAMPLE_FLAGS}
)

if (ARC)
target_link_options(example_kws_speech PRIVATE
    -m
    -Coutput=./elf_files/example_kws_speech.map
)
endif()

set_target_properties(example_kws_speech
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "./elf_files$<0:>"
)

