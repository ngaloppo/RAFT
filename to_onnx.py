import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cpu'

def load_model(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location=torch.device(DEVICE)))
    model = model.module
    model.to(DEVICE)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert RAFT model to OpenVINO")
    parser.add_argument('-m', '--model', help="restore checkpoint", required=True)
    parser.add_argument('-o', '--output_path', help="output path", required=True)
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--shape', type=int, nargs='+', default=[1, 3, 436, 1024])
    parser.add_argument('--iters', type=int, default=20)
    args = parser.parse_args()

    model = load_model(args)
    inp = Variable(torch.randn(args.shape))
    model.eval()

    with torch.no_grad():
        padder = InputPadder(inp.shape)
        input1, input2 = padder.pad(inp, inp)
        torch.onnx.export(model, (input1, input2), args.output_path, 
                input_names=['input1', 'input2'], 
                output_names=['flow_low', 'flow_up'], 
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, 
                opset_version=11,
                verbose=True)
        print(f"Model converted to ONNX, output: {args.output_path}")
