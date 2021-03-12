import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import torch._C as _C
OperatorExportTypes = _C._onnx.OperatorExportTypes

DEVICE = 'cpu'

def load_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert RAFT model to OpenVINO")
    parser.add_argument('-m', '--model', help="restore checkpoint")
    parser.add_argument('-o', '--output_path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    model = load_model(args)
    height = 436
    width = 1024
    input = torch.randn(1, 3, height, width)
    padder = InputPadder(input.shape)
    input1, input2 = padder.pad(input, input)
    torch.onnx.export(model, (input1, input2), args.output_path, input_names=['input1', 'input2'], output_names=['flow_low', 'flow_up'], verbose=True, operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=11)
    # torch.onnx.export(model, (input1, input2), args.output_path, input_names=['input1', 'input2'], output_names=['flow_low', 'flow_up'], verbose=True, opset_version=11)
    print(f"Model converted to ONNX, output: {args.output_path}")
