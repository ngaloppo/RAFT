// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "cpu_kernel.hpp"
#include "op.hpp"
#include <details/ie_exception.hpp>
#include <ie_layouts.h>
#include "ie_parallel.hpp"

using namespace TemplateExtension;

//! [cpu_implementation:ctor]
GridSampleImpl::GridSampleImpl(const std::shared_ptr<ngraph::Node> &node) {
    try {
        auto castedNode = std::dynamic_pointer_cast<GridSampleOp>(node);
        if (!castedNode)
            THROW_IE_EXCEPTION << "Cannot create implementation for unknown operation!";
        if (castedNode->inputs().size() != 2 || castedNode->outputs().size() != 1)
            THROW_IE_EXCEPTION << "Cannot create implementation for operation with incorrect number of inputs or outputs!";
        if (castedNode->get_input_partial_shape(0).is_dynamic() || castedNode->get_output_partial_shape(0).is_dynamic())
            THROW_IE_EXCEPTION << "Cannot create implementation for op with dynamic shapes!";
        if (castedNode->get_input_shape(0).size() != 4 || castedNode->get_output_shape(0).size() != 4)
            THROW_IE_EXCEPTION << "Operation supports only 4d tensors for input and output.";
        if (castedNode->get_input_element_type(0) != ngraph::element::f32 || castedNode->get_output_element_type(0) != ngraph::element::f32)
            THROW_IE_EXCEPTION << "Operation supports only FP32 tensors.";
        inShapes.resize(2);
        for (int i = 0; i < 2; ++i)
            inShapes[i] = castedNode->get_input_shape(i);
        outShape = castedNode->get_output_shape(0);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        error = ex.what();
    }
}
//! [cpu_implementation:ctor]

//! [cpu_implementation:getSupportedConfigurations]
InferenceEngine::StatusCode GridSampleImpl::getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig> &conf,
                                                                         InferenceEngine::ResponseDesc *resp) noexcept {
     std::vector<InferenceEngine::DataConfig> inDataConfig;
     std::vector<InferenceEngine::DataConfig> outDataConfig;
     InferenceEngine::SizeVector order = {0, 1, 2, 3};
     // Allow any offset before data
     size_t offset((std::numeric_limits<size_t>::max)());

     // Input shape
     for (const auto& shape : inShapes)
     {
         InferenceEngine::DataConfig inpConf;
         inpConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, shape, {shape, order, offset});
         inDataConfig.push_back(inpConf);
     }

     // Output shape
     InferenceEngine::DataConfig outConf;
     outConf.desc = InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, outShape, {outShape, order, offset});
     outDataConfig.push_back(outConf);

     InferenceEngine::LayerConfig layerConfig;
     layerConfig.inConfs = inDataConfig;
     layerConfig.outConfs = outDataConfig;

     conf.push_back(layerConfig);
     return InferenceEngine::StatusCode::OK;
}
//! [cpu_implementation:getSupportedConfigurations]

//! [cpu_implementation:init]
InferenceEngine::StatusCode GridSampleImpl::init(InferenceEngine::LayerConfig &config, InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        if (config.inConfs.size() != 2 || config.outConfs.size() != 1) {
            THROW_IE_EXCEPTION << "Operation cannot be initialized with incorrect number of inputs/outputs!";
        }

        if (config.inConfs[0].desc.getDims().size() != 4 || config.outConfs[0].desc.getDims().size() != 4) {
            THROW_IE_EXCEPTION << "Operation can be initialized only with 4d input/output tensors!";
        }

        if (config.outConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32 ||
                config.inConfs[0].desc.getPrecision() != InferenceEngine::Precision::FP32)  {
            THROW_IE_EXCEPTION << "Operation supports only FP32 precisions!";
        }
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        if (resp) {
            strncpy(resp->msg, error.c_str(), sizeof(resp->msg) - 1);
            resp->msg[sizeof(resp->msg)-1] = 0;
        }
        return InferenceEngine::GENERAL_ERROR;
    }

    return InferenceEngine::OK;
}
//! [cpu_implementation:init]

//! [cpu_implementation:execute]
InferenceEngine::StatusCode GridSampleImpl::execute(std::vector<InferenceEngine::Blob::Ptr> &inputs,
                                                    std::vector<InferenceEngine::Blob::Ptr> &outputs,
                                                    InferenceEngine::ResponseDesc *resp) noexcept {
    const float* inpData  = inputs[0]->cbuffer().as<float*>();
    const float* gridData = inputs[1]->cbuffer().as<float*>();
    float* outData = outputs[0]->buffer().as<float*>();

    std::vector<size_t> inpDims = inputs[0]->getTensorDesc().getDims();
    std::vector<size_t> outDims = outputs[0]->getTensorDesc().getDims();

    const int batch     = outDims[0];
    const int channels  = outDims[1];
    const int height    = outDims[2];
    const int width     = outDims[3];
    const int inpHeight = inpDims[2];
    const int inpWidth  = inpDims[3];
    const int inpPlane = inpHeight * inpWidth;
    const int outPlane = height * width;
    InferenceEngine::parallel_for(batch, [&](int d) {
        const float* inp  = inpData + d * inpPlane;
        const float* grid = gridData + d * outPlane * 2;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int offset = y * width + x;

                float input_x = (grid[offset * 2] + 1) * inpWidth;
                int x0 = static_cast<int>(input_x);
                int x1 = std::min(x0 + 1, inpWidth - 1);

                float input_y = (grid[offset * 2 + 1] + 1) * inpHeight;
                int y0 = static_cast<int>(input_y);
                int y1 = std::min(y0 + 1, inpHeight - 1);

                const float* inp_row0 = inp + y0 * inpWidth;
                const float* inp_row1 = inp + y1 * inpWidth;
                float* out = outData + d * outPlane + offset;
                for (int c = 0; c < channels; ++c) {
                    *out = inp_row0[x0] +
                           (input_y - y0) * (inp_row1[x0] - inp_row0[x0]) +
                           (input_x - x0) * (inp_row0[x1] - inp_row0[x0] +
                           (input_y - y0) * (inp_row1[x1] - inp_row0[x1] - inp_row1[x0] + inp_row0[x0]));
                    out += outPlane;
                }
            }
        }
    });
    return InferenceEngine::OK;
}
//! [cpu_implementation:execute]
