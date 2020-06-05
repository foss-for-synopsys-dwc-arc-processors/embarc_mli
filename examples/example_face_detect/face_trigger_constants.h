/*
* Copyright 2019-2020, Synopsys, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-3-Clause license found in
* the LICENSE file in the root directory of this source tree.
*
*/

#ifndef _FACE_TRIGGER_CONST_H_
#define _FACE_TRIGGER_CONST_H_

//======================================================
//
// Hardcoded Weights and Biases
//
//======================================================

#define CONV1_WEIGHTS { \
        0x75, 0x01, 0x10, 0xFF, 0xB5, 0x04, 0x29, 0x0E, 0x7E, 0x0E, 0x6F, 0x07, 0x33, 0x02, 0x26, 0xFB, 0x72, 0xF5, 0xBF, 0xF3, 0xCC, 0xF1, 0x8D, 0xFA, 0xBB, 0x00, 0xD7, 0xFC, 0xAD, 0xF3, \
        0x28, 0xE9, 0x8F, 0xE4, 0xF8, 0xF2, 0xA1, 0x01, 0xBA, 0x06, 0x10, 0x14, 0xDA, 0x20, 0x55, 0x1D, 0x72, 0x0B, 0xA4, 0x00, 0x6E, 0xFF, 0x7A, 0x03, 0xAC, 0x0C, 0x26, 0x0E, 0x29, 0x06, \
        0xFF, 0xFE, 0x75, 0xF9, 0x5A, 0xF2, 0x65, 0xEF, 0x12, 0xF2, 0x92, 0xFA, 0x40, 0xFE, 0x90, 0xFD, 0x82, 0x01, 0x0C, 0x03, 0xE4, 0xFF, 0x0A, 0xFF, 0x56, 0xFD, 0xEF, 0xFA, 0x94, 0x01, \
        0xA6, 0x06, 0x3F, 0x01, 0x93, 0xFE, 0x5C, 0xFF, 0x75, 0xFB, 0x43, 0xFE, 0x4B, 0x05, 0x0B, 0x03, 0xEB, 0xFF, 0x11, 0x01, 0x15, 0xFE, 0x65, 0xFD, 0x54, 0x01, 0x44, 0x01, 0x52, 0x00, \
        0x5B, 0x00, 0x2B, 0xFF, 0xED, 0xFE, 0x85, 0xFF, 0x25, 0x00, 0xBD, 0x00, 0x90, 0xFF, 0x2C, 0xFF, 0xBA, 0xFF, 0xE2, 0xFF, 0x91, 0x00, 0xCD, 0x00, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0x01, 0x00, 0x01, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x02, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x03, 0x00, 0x03, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, \
        0x01, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x01, 0x00, 0xFD, 0xFF, 0xFE, 0xFF, 0x00, 0x00, 0xFE, 0xFF, 0xFE, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x06, 0xDA, 0x07, 0x2D, 0x03, 0xC7, 0x03, 0xE3, 0x08, 0x86, 0x06, 0x64, 0x07, 0x99, 0x06, 0x1C, 0xFE, 0xFF, 0xFE, 0x7B, 0x07, 0x63, 0x07, \
        0x6C, 0x05, 0xD3, 0x03, 0x0E, 0xFC, 0xFD, 0xFB, 0xB9, 0x01, 0x63, 0x03, 0x47, 0x07, 0xD6, 0x06, 0xC2, 0xFC, 0xCF, 0xFC, 0x75, 0x05, 0xD9, 0x05, 0x5C, 0x02, 0x22, 0xFA, 0x8D, 0xEB, \
        0x6F, 0xED, 0xFC, 0xFD, 0x53, 0x04, 0x34, 0xFF, 0x5F, 0xF8, 0xE9, 0xEF, 0x65, 0xF1, 0xA9, 0xFB, 0x01, 0x01}

#define CONV2_A_WEIGHTS { \
        0x16, 0x00, 0x20, 0xF8, 0xA9, 0x03, 0x9F, 0x0B, 0xF3, 0xFA, 0xBA, 0xF4, 0xA2, 0x07, 0xDB, 0x0D, 0x0E, 0xF7, 0x10, 0xFC, 0x8E, 0x06, 0x8C, 0x01, 0x32, 0xFC, 0x76, 0xFF, 0x95, 0x02, \
        0x50, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0xFF, 0xFF, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x00, \
        0x00, 0x00, 0xFE, 0xFF, 0x0F, 0x00, 0x06, 0x00, 0xC8, 0xFF, 0xD2, 0xFF, 0x0B, 0x00, 0x0E, 0x00, 0xAA, 0xFF, 0xA7, 0xFF, 0x06, 0x00, 0x0B, 0x00, 0xAF, 0xFF, 0xAA, 0xFF, 0x0A, 0x00, \
        0x03, 0x00, 0xCD, 0xFF, 0xD4, 0xFF, 0x40, 0xFF, 0x25, 0xFE, 0xA8, 0xFD, 0xC4, 0xFE, 0x0C, 0x00, 0x3C, 0xFD, 0xEA, 0xFB, 0xBA, 0xFE, 0x34, 0xFF, 0xCB, 0xFD, 0xFF, 0xFE, 0x68, 0x00, \
        0x68, 0xFE, 0xB4, 0xFE, 0xBD, 0x00, 0x72, 0x00, 0x01, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0x03, 0x00, 0x02, 0x00, 0xFD, 0xFF, 0xFD, 0xFF, 0x03, 0x00, \
        0x04, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0x02, 0x00, 0x04, 0x00, 0xFF, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFE, 0xFF, 0xFD, 0xFF, 0xFD, 0xFF, 0xFC, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, \
        0x00, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF}

#define CONV2_B_WEIGHTS { \
        0x44, 0x00, 0x55, 0x00, 0x5C, 0x00, 0x4B, 0x00, 0xA3, 0x00, 0xAD, 0x00, 0xA1, 0x00, \
        0x97, 0x00, 0xAD, 0x00, 0xA2, 0x00, 0x6E, 0x00, 0x79, 0x00, 0x4D, 0x00, 0x4A, 0x00, 0x2A, 0x00, 0x2D, 0x00, 0xD7, 0x00, 0xF1, 0x00, 0x3D, 0x00, 0x23, 0x00, 0x42, 0x00, 0xF8, 0xFF, \
        0xEA, 0xFF, 0x33, 0x00, 0xD9, 0xFE, 0x9F, 0xFE, 0x0A, 0xFF, 0x45, 0xFF, 0x6F, 0xFF, 0x98, 0xFF, 0x5E, 0xFF, 0x35, 0xFF, 0xF2, 0xFF, 0xF4, 0xFF, 0x39, 0x00, 0x38, 0x00, 0xF1, 0xFF, \
        0xE9, 0xFF, 0x5D, 0x00, 0x66, 0x00, 0x07, 0x00, 0xD4, 0xFF, 0x2A, 0x00, 0x5E, 0x00, 0x08, 0x00, 0xDF, 0xFF, 0x06, 0x00, 0x30, 0x00, 0x06, 0x00, 0xF0, 0xFF, 0xE9, 0xFF, 0xFF, 0xFF, \
        0x12, 0x00, 0xE8, 0xFF, 0xC1, 0xFF, 0xEB, 0xFF, 0x0A, 0x00, 0x00, 0x00, 0xE5, 0xFF, 0xF0, 0xFF, 0xFF, 0xFF, 0x07, 0x00, 0x0C, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFE, 0xFF, 0xFF, 0xFF, \
        0x01, 0x00, 0xFF, 0xFF, 0xFB, 0xFF, 0xFC, 0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, \
        0x00, 0x00, 0xFF, 0xFF, 0x01, 0x00, 0x00, 0x00, 0xFC, 0xFF, 0xFC, 0xFF, 0x02, 0x00, 0x01, 0x00, 0xFD, 0xFF, 0xFE, 0xFF, 0x02, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x03, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x01, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0x01, 0x00, 0xFF, 0xFF, 0xFD, 0xFF, 0xFE, 0xFF, 0x01, 0x00, 0x04, 0x00, 0x00, 0x00, 0xFC, 0xFF, 0x00, 0x00, 0x06, 0x00, 0x02, 0x00, 0xFE, 0xFF, 0x00, 0x00, 0x01, 0x00, \
        0xFF, 0xFF, 0xE0, 0xFE, 0xC7, 0xFE, 0x85, 0xFF, 0x9E, 0xFF, 0xDB, 0xFE, 0x7B, 0xFF, 0xFE, 0xFF, 0x5E, 0xFF, 0x2F, 0x00, 0xC1, 0x00, 0x9D, 0x00, 0x0B, 0x00, 0x34, 0x00, 0x0D, 0x00, \
        0x24, 0x00, 0x4B, 0x00, 0x15, 0x00, 0x90, 0x00, 0x9C, 0x00, 0x21, 0x00, 0x4C, 0x00, 0x53, 0x00, 0x91, 0x00, 0x8B, 0x00, 0x8C, 0x00, 0x2E, 0x00, 0x52, 0x00, 0xB0, 0x00, 0x55, 0x00, \
        0x6B, 0x00, 0x5C, 0x00, 0x47, 0x00, 0x0D, 0x00, 0x03, 0x00, 0xF5, 0xFF, 0x00, 0x00, 0x0D, 0x00, 0x07, 0x00, 0xF0, 0xFF, 0xF6, 0xFF, 0x0F, 0x00, 0x1B, 0x00, 0xFA, 0xFF, 0xEE, 0xFF, \
        0x0F, 0x00, 0x16, 0x00, 0xFF, 0xFF, 0xF7, 0xFF, 0xF7, 0xFF, 0xE8, 0xFF, 0xE7, 0xFF, 0xF6, 0xFF, 0x03, 0x00, 0xF3, 0xFF, 0xC8, 0xFF, 0xD8, 0xFF, 0x15, 0x00, 0x09, 0x00, 0xB8, 0xFF, \
        0xC4, 0xFF, 0x09, 0x00, 0xFE, 0xFF, 0xD7, 0xFF, 0xE2, 0xFF}

#define CONV3_WEIGHTS  { \
        0xD2, 0xFF, 0x87, 0x00, 0x40, 0x00, 0xC4, 0xFF, 0x74, 0xFF, 0x62, 0x00, 0x0E, 0x00, 0x44, 0x00, 0xED, 0xFF, 0x35, 0x00, 0x08, 0x00, 0x4F, 0x00, 0xF6, 0xFF, 0x38, 0x00, 0x07, 0x00, \
        0x19, 0x00, 0x04, 0x00, 0xCD, 0x00, 0xE8, 0xFF, 0x1A, 0xFF, 0xA8, 0xFF, 0x77, 0x00, 0x2E, 0x00, 0x73, 0x00, 0x82, 0xFF, 0x8A, 0xFF, 0x0B, 0x00, 0xA7, 0x00, 0x7E, 0x00, 0x37, 0x00, \
        0xCF, 0xFF, 0xC8, 0xFF, 0x0B, 0x00, 0x98, 0x00, 0xD0, 0x00, 0x24, 0x00, 0xF8, 0xFF, 0xEE, 0xFF, 0xF0, 0xFF, 0xE7, 0xFF, 0xE1, 0xFF, 0xE5, 0xFF, 0xEB, 0xFF, 0xF8, 0xFF, 0x15, 0x00, \
        0x18, 0x00, 0xFB, 0xFF, 0x05, 0x00, 0xFA, 0xFF, 0x04, 0x00, 0x1E, 0x00, 0x17, 0x00, 0xF3, 0xFF, 0xFE, 0xFF, 0xFE, 0xFF, 0xF0, 0xFF, 0x0F, 0x00, 0x13, 0x00, 0xEA, 0xFF, 0x0C, 0x00, \
        0xE7, 0xFF, 0xF2, 0xFF, 0xF5, 0xFF, 0xF0, 0xFF, 0xEB, 0xFF, 0x0C, 0x00, 0x05, 0x00, 0x07, 0x00, 0x05, 0x00, 0x06, 0x00, 0xFA, 0xFF, 0x08, 0x00, 0xEC, 0xFF, 0x21, 0x00, 0x04, 0x00, \
        0x01, 0x00, 0x0B, 0x00, 0x14, 0x00, 0xF6, 0xFF, 0x2A, 0x00, 0xEF, 0xFF, 0x09, 0x00, 0x25, 0x00, 0xDA, 0xFF, 0x14, 0x00, 0x01, 0x00, 0xDA, 0xFF, 0x0A, 0x00, 0xFF, 0xFF, 0xDD, 0xFF, \
        0x21, 0x00, 0xF2, 0xFF, 0xEB, 0xFF, 0x2C, 0x00, 0x16, 0x00, 0xDC, 0xFF, 0x0F, 0x00, 0xEA, 0xFF, 0xDF, 0xFF, 0x1A, 0x00, 0x02, 0x00, 0xF8, 0xFF, 0x24, 0x00, 0xF4, 0xFF, 0xE7, 0xFF, \
        0x2F, 0x00, 0x2C, 0x00, 0x07, 0x00, 0x93, 0xFF, 0xAF, 0xFF, 0xC1, 0x01, 0x22, 0xFE, 0xC5, 0xFE, 0xC0, 0xFD, 0x82, 0xFF, 0x9C, 0x00, 0x8C, 0x02, 0x54, 0x00, 0x32, 0x01, 0x85, 0x01, \
        0xCC, 0x00, 0xCC, 0xFE, 0x6C, 0xFE, 0xD7, 0xFF, 0x5E, 0x00, 0x8F, 0xFF, 0x98, 0xFF, 0x15, 0x00, 0x79, 0x02, 0x74, 0xFF, 0xB3, 0x02, 0x71, 0xFF, 0x8B, 0x00, 0x54, 0x00, 0x67, 0xFF, \
        0x2C, 0xFF, 0xB4, 0xFF, 0x23, 0x00, 0x37, 0x00, 0x7C, 0x01, 0x9E, 0xFF, 0x62, 0xFE, 0x77, 0xFF, 0x4C, 0xFE, 0x0F, 0x00, 0x0E, 0x00, 0x11, 0x00, 0x0C, 0x00, 0x23, 0x00, 0x09, 0x00, \
        0x23, 0x00, 0x04, 0x00, 0x01, 0x00, 0x09, 0x00, 0x10, 0x00, 0x15, 0x00, 0xF4, 0xFF, 0x04, 0x00, 0x1D, 0x00, 0x1E, 0x00, 0x05, 0x00, 0x08, 0x00, 0x14, 0x00, 0x11, 0x00, 0x1C, 0x00, \
        0xF8, 0xFF, 0x10, 0x00, 0x26, 0x00, 0xF6, 0xFF, 0x18, 0x00, 0xF7, 0xFF, 0xFB, 0xFF, 0x0A, 0x00, 0xF5, 0xFF, 0xEE, 0xFF, 0xFD, 0xFF, 0x0E, 0x00, 0xEE, 0xFF, 0x10, 0x00, 0xED, 0xFF, \
        0xE5, 0xFF, 0x13, 0x00, 0xE7, 0xFF, 0x0C, 0x00, 0xF2, 0xFF, 0xF8, 0xFF, 0xFB, 0xFF, 0x1D, 0x00, 0x03, 0x00, 0x11, 0x00, 0xE4, 0xFF, 0x10, 0x00, 0xFE, 0xFF, 0x1F, 0x00, 0x03, 0x00, \
        0x02, 0x00, 0x11, 0x00, 0xFA, 0xFF, 0xF9, 0xFF, 0xED, 0xFF, 0x18, 0x00, 0xF7, 0xFF, 0xF1, 0xFF, 0x0D, 0x00, 0x12, 0x00, 0xEA, 0xFF, 0x00, 0x00, 0xEB, 0xFF, 0xFD, 0xFF, 0x0E, 0x00, \
        0x09, 0x00, 0x0E, 0x00, 0xFA, 0xFF, 0x07, 0x00, 0x0D, 0x00, 0xFD, 0xFF, 0xF7, 0xFF, 0x1A, 0x00, 0xF9, 0xFF, 0xFD, 0xFF, 0x01, 0x00, 0x0D, 0x00, 0xFC, 0xFF, 0xF9, 0xFF, 0xEE, 0xFF, \
        0x16, 0x00, 0x1A, 0x00, 0x15, 0x00, 0xF0, 0xFF, 0x11, 0x00, 0x12, 0x00, 0x10, 0x00, 0x05, 0x00, 0x12, 0x00, 0xED, 0xFF, 0xFB, 0xFF, 0x1E, 0x00, 0xEB, 0xFF, 0xE9, 0xFF, 0xF7, 0xFF, \
        0x17, 0x00, 0x13, 0x00, 0xF4, 0xFF, 0x18, 0x00, 0xF0, 0xFF, 0x03, 0x00, 0xFB, 0xFF, 0x11, 0x00, 0x0C, 0x00, 0x1E, 0x00, 0xF9, 0xFF, 0xE7, 0xFF, 0xE9, 0xFF, 0xE2, 0xFF, 0x11, 0x00, \
        0x1B, 0x00, 0x18, 0x00, 0x0C, 0x00, 0x07, 0x00, 0x10, 0x00, 0x17, 0x00, 0x02, 0x00, 0x00, 0x00, 0xE0, 0xFF, 0x00, 0x00, 0x06, 0x00, 0x1F, 0x00, 0xFC, 0xFF, 0x16, 0x00, 0x09, 0x00, \
        0xEB, 0xFF, 0x19, 0x00, 0xDD, 0xFF, 0x11, 0x00, 0x0E, 0x00, 0x0B, 0x00, 0xFC, 0xFF, 0x0C, 0x00, 0x18, 0x00, 0xEF, 0xFF, 0x07, 0x00, 0x0C, 0x00, 0xE9, 0xFF, 0x02, 0x00, 0x04, 0x00, \
        0xFF, 0xFF, 0xEE, 0xFF, 0x18, 0x00, 0xD6, 0x00, 0x7A, 0x00, 0x7F, 0xFF, 0xF7, 0xFE, 0xAE, 0x02, 0xD0, 0xFE, 0x8F, 0x00, 0x6C, 0x00, 0x55, 0xFD, 0x35, 0xFF, 0xC6, 0x00, 0xF9, 0x00, \
        0x9B, 0xFE, 0x99, 0x01, 0x92, 0xFD, 0x8D, 0x01, 0x1E, 0xFF, 0xB2, 0xFF, 0x2E, 0xFE, 0x6D, 0x01, 0x9C, 0xFF, 0x36, 0x01, 0xB9, 0xFF, 0x57, 0xFE, 0xBC, 0xFE, 0x45, 0x01, 0x94, 0xFF, \
        0xD4, 0x00, 0x14, 0x01, 0x24, 0xFF, 0xC2, 0xFE, 0x63, 0x00, 0xDF, 0x00, 0xB2, 0x00, 0x57, 0x00, 0xC4, 0xFE, 0x18, 0x00, 0xE2, 0xFF, 0x1B, 0x00, 0x26, 0x00, 0xE9, 0xFF, 0x06, 0x00, \
        0x02, 0x00, 0xEC, 0xFF, 0x1A, 0x00, 0x1C, 0x00, 0x20, 0x00, 0xF7, 0xFF, 0xE5, 0xFF, 0x10, 0x00, 0xEF, 0xFF, 0x07, 0x00, 0xE7, 0xFF, 0x2F, 0x00, 0x0C, 0x00, 0x08, 0x00, 0x2E, 0x00, \
        0xDB, 0xFF, 0xF3, 0xFF, 0x22, 0x00, 0xD2, 0xFF, 0xE5, 0xFF, 0x2B, 0x00, 0xE9, 0xFF, 0xEE, 0xFF, 0x38, 0x00, 0xD8, 0xFF, 0x31, 0x00, 0x38, 0x00, 0xDF, 0xFF, 0xEC, 0xFF, 0x0F, 0x00, \
        0xE4, 0xFF, 0x20, 0x00, 0x00, 0x00, 0xE5, 0xFF, 0x0C, 0x00, 0x07, 0x00, 0xEE, 0xFF, 0x06, 0x00, 0x13, 0x00, 0x04, 0x00, 0x01, 0x00, 0xED, 0xFF, 0x0D, 0x00, 0x0E, 0x00, 0x17, 0x00, \
        0xF8, 0xFF, 0x1E, 0x00, 0x08, 0x00, 0xF6, 0xFF, 0x23, 0x00, 0xE8, 0xFF, 0xED, 0xFF, 0xE9, 0xFF, 0xF4, 0xFF, 0x1B, 0x00, 0xF4, 0xFF, 0xEE, 0xFF, 0xEF, 0xFF, 0x10, 0x00, 0xEA, 0xFF, \
        0xEC, 0xFF, 0xFF, 0xFF, 0xF8, 0xFF, 0x1E, 0x00, 0x14, 0x00, 0xF0, 0xFF, 0xED, 0xFF, 0xF4, 0xFF, 0xFA, 0xFF, 0x19, 0x00, 0x09, 0x00, 0xE8, 0xFF, 0xF7, 0xFF, 0xED, 0xFF, 0xFA, 0xFF, \
        0x10, 0x00, 0xEE, 0xFF, 0xEC, 0xFF, 0xEA, 0xFF, 0xFA, 0xFF, 0xEC, 0xFF, 0xEF, 0xFF, 0x1C, 0x00, 0x14, 0x00, 0x03, 0x00, 0x04, 0x00, 0xFB, 0xFF, 0xEC, 0xFF, 0xFD, 0xFF, 0x0F, 0x00, \
        0xE7, 0xFF, 0xEE, 0xFF, 0xEB, 0xFF, 0x1A, 0x00, 0xFB, 0xFF, 0x0A, 0x00, 0x23, 0x00, 0x28, 0x00, 0x09, 0x00, 0x23, 0x00, 0x17, 0x00, 0x16, 0x00, 0x11, 0x01, 0x41, 0xFF, 0x2A, 0xFF, \
        0x73, 0xFD, 0xD9, 0xFE, 0x41, 0x01, 0x24, 0x01, 0xFD, 0xFF, 0x6D, 0x01, 0xD2, 0x01, 0xB7, 0xFF, 0x27, 0x01, 0x4E, 0x01, 0x9B, 0x00, 0x56, 0xFE, 0x33, 0xFE, 0x1D, 0x01, 0x34, 0x01, \
        0x19, 0xFF, 0x4A, 0xFE, 0x6E, 0xFF, 0x32, 0x00, 0x73, 0xFF, 0x2D, 0xFF, 0x06, 0x00, 0x7D, 0xFF, 0xA1, 0xFF, 0x54, 0xFF, 0x69, 0x00, 0x43, 0x00, 0x45, 0xFF, 0xC7, 0xFF, 0x2D, 0x00, \
        0x7B, 0x00, 0x7B, 0x00, 0x17, 0xFF, 0xD7, 0xFF, 0xFA, 0xFF, 0x09, 0x00, 0xDB, 0xFF, 0xDE, 0xFF, 0x03, 0x00, 0xFA, 0xFF, 0xEF, 0xFF, 0xE0, 0xFF, 0xFD, 0xFF, 0xEF, 0xFF, 0x06, 0x00, \
        0xFF, 0xFF, 0xE3, 0xFF, 0x0E, 0x00, 0xF0, 0xFF, 0x28, 0x00, 0x16, 0x00, 0x0A, 0x00, 0x1D, 0x00, 0xFD, 0xFF, 0x0D, 0x00, 0xF3, 0xFF, 0x03, 0x00, 0x0B, 0x00, 0xF3, 0xFF, 0xF7, 0xFF, \
        0xF7, 0xFF, 0x18, 0x00, 0x10, 0x00, 0x2C, 0x00, 0x13, 0x00, 0x11, 0x00, 0x14, 0x00, 0x23, 0x00, 0xE3, 0xFF}

#define FC4_WEIGHTS {0x40, 0xFC, 0x59, 0x00, 0x89, 0x00, 0x74, 0xF8, 0x8C, 0x00, 0x66, 0x00, 0x6F, 0x00, 0x79, 0x00, 0x33, 0x07, 0xAE, 0x00, 0x70, 0x00, 0x77, 0x00, 0x19, 0xF9, 0x7A, 0x00}


#define CONV1_BIAS {0x99, 0xFC, 0xD5, 0xFD, 0x0D, 0x00, 0x4B, 0xFF}
#define CONV2_A_BIAS {0xE7, 0xFF, 0xED, 0xFF, 0x2A, 0x00, 0xF7, 0xFF, 0xFA, 0xFF, 0x12, 0x00, 0xE8, 0xFF, 0xEB, 0xFF}
#define CONV2_B_BIAS {0xB5, 0xFF, 0xE3, 0xFF, 0xF2, 0xFF, 0x06, 0x00, 0x27, 0xFE, 0x47, 0x00}
#define CONV3_BIAS  {0x49, 0xFF, 0xE1, 0xFF, 0xF9, 0xFF, 0x7D, 0x01, 0xF5, 0xFF, 0xE3, 0xFF, 0xE8, 0xFF, 0x1D, 0x00, 0xA9, 0xFF, 0x0B, 0x00, 0x1E, 0x00, 0x1C, 0x00, 0xB2, 0x03, 0x22, 0x00}
#define FC4_BIAS  {0x60, 0x01}


//======================================================
//
// User activation lookup table
//
//======================================================

#define ACT_LUT_SIZE (1024)
#define ACT_LUT_IDX_BITS (10)
#define ACT_LUT { \
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F, 0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, \
        0x9D, 0x9E, 0x9F, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB6, 0xB7, \
        0xB8, 0xB9, 0xBA, 0xBA, 0xBB, 0xBC, 0xBD, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1, 0xC1, 0xC2, 0xC3, 0xC3, 0xC4, 0xC5, 0xC6, 0xC6, 0xC7, 0xC8, 0xC8, 0xC9, 0xCA, 0xCA, 0xCB, 0xCC, 0xCC, 0xCD, \
        0xCE, 0xCE, 0xCF, 0xCF, 0xD0, 0xD1, 0xD1, 0xD2, 0xD2, 0xD3, 0xD4, 0xD4, 0xD5, 0xD5, 0xD6, 0xD6, 0xD7, 0xD8, 0xD8, 0xD9, 0xD9, 0xDA, 0xDA, 0xDB, 0xDB, 0xDC, 0xDC, 0xDD, 0xDD, 0xDD, \
        0xDE, 0xDE, 0xDF, 0xDF, 0xE0, 0xE0, 0xE1, 0xE1, 0xE1, 0xE2, 0xE2, 0xE3, 0xE3, 0xE4, 0xE4, 0xE4, 0xE5, 0xE5, 0xE5, 0xE6, 0xE6, 0xE7, 0xE7, 0xE7, 0xE8, 0xE8, 0xE8, 0xE9, 0xE9, 0xE9, \
        0xEA, 0xEA, 0xEA, 0xEB, 0xEB, 0xEB, 0xEB, 0xEC, 0xEC, 0xEC, 0xED, 0xED, 0xED, 0xED, 0xEE, 0xEE, 0xEE, 0xEE, 0xEF, 0xEF, 0xEF, 0xEF, 0xF0, 0xF0, 0xF0, 0xF0, 0xF1, 0xF1, 0xF1, 0xF1, \
        0xF1, 0xF2, 0xF2, 0xF2, 0xF2, 0xF3, 0xF3, 0xF3, 0xF3, 0xF3, 0xF3, 0xF4, 0xF4, 0xF4, 0xF4, 0xF4, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF5, 0xF6, 0xF6, 0xF6, 0xF6, 0xF6, 0xF6, 0xF6, 0xF7, \
        0xF7, 0xF7, 0xF7, 0xF7, 0xF7, 0xF7, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF8, 0xF9, 0xF9, 0xF9, 0xF9, 0xF9, 0xF9, 0xF9, 0xF9, 0xF9, 0xFA, 0xFA, 0xFA, 0xFA, 0xFA, 0xFA, \
        0xFA, 0xFA, 0xFA, 0xFA, 0xFA, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFB, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, 0xFC, \
        0xFC, 0xFC, 0xFC, 0xFC, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFE, 0xFE, 0xFE, 0xFE, \
        0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, \
        0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, \
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, \
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, \
        0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, \
        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, \
        0x03, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, \
        0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x09, \
        0x09, 0x09, 0x09, 0x09, 0x09, 0x09, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0B, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, 0x0D, \
        0x0E, 0x0E, 0x0E, 0x0E, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x10, 0x10, 0x10, 0x10, 0x11, 0x11, 0x11, 0x11, 0x12, 0x12, 0x12, 0x12, 0x13, 0x13, 0x13, 0x13, 0x14, 0x14, 0x14, 0x15, 0x15, \
        0x15, 0x15, 0x16, 0x16, 0x16, 0x17, 0x17, 0x17, 0x18, 0x18, 0x18, 0x19, 0x19, 0x19, 0x1A, 0x1A, 0x1B, 0x1B, 0x1B, 0x1C, 0x1C, 0x1C, 0x1D, 0x1D, 0x1E, 0x1E, 0x1F, 0x1F, 0x1F, 0x20, \
        0x20, 0x21, 0x21, 0x22, 0x22, 0x23, 0x23, 0x23, 0x24, 0x24, 0x25, 0x25, 0x26, 0x26, 0x27, 0x27, 0x28, 0x28, 0x29, 0x2A, 0x2A, 0x2B, 0x2B, 0x2C, 0x2C, 0x2D, 0x2E, 0x2E, 0x2F, 0x2F, \
        0x30, 0x31, 0x31, 0x32, 0x32, 0x33, 0x34, 0x34, 0x35, 0x36, 0x36, 0x37, 0x38, 0x38, 0x39, 0x3A, 0x3A, 0x3B, 0x3C, 0x3D, 0x3D, 0x3E, 0x3F, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x43, 0x44, \
        0x45, 0x46, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, \
        0x5F, 0x60, 0x61, 0x62, 0x63, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, \
        0x7C, 0x7D, 0x7E, 0x7F}



#endif    // _FACE_TRIGGER_CONST_H_
