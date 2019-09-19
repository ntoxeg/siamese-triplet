from PIL import Image
from utils import parse_image, extract_bbox
import os
from datasets import COCODataset

CENTERNET_TASK = "ctdet"
CENTERNET_MODEL_PATH = "/home/adrian/projects/CenterNet/models/ctdet_coco_dla_2x.pth"

import sys

sys.path.extend(
    [
        "/home/adrian/projects/CenterNet/src",
        "/home/adrian/projects/CenterNet/src/centernet/models/networks/DCNv2",
    ]
)

from centernet.detectors.detector_factory import detector_factory
from centernet.opts import opts

opt = opts().init([CENTERNET_TASK, "--load_model", CENTERNET_MODEL_PATH])
import torch

detector = detector_factory[opt.task](opt)
opt.data_dir = "/home/adrian/data"
cocodata = COCODataset(opt, "val")


datain = "data/coconut/train"
dataout = "data/classy_coconut/train"
data_dir = "data"
data_split = "train"
annFile = f"{data_dir}/coco/annotations/instances_{data_split}2017.json"


def run(datain, dataout):
    for klass in os.listdir(datain):
        if not os.path.isdir(os.path.join(dataout, klass)):
            os.makedirs(os.path.join(dataout, klass))
        for imgfile in os.listdir(os.path.join(datain, klass)):
            image = Image.open(os.path.join(datain, klass, imgfile))
            visual_parse = parse_image(detector, image)

            klass_bboxes = visual_parse[cocodata.class_name.index(klass)]
            if len(klass_bboxes) == 0:
                continue
            bbox = klass_bboxes[0, :4].tolist()
            klassimg = extract_bbox(image, bbox)
            try:
                klassimg.save(os.path.join(dataout, klass, imgfile))
            except SystemError:
                continue


run(datain, dataout)

datain = "data/coconut/val"
dataout = "data/classy_coconut/val"
data_dir = "data"
data_split = "val"
annFile = f"{data_dir}/coco/annotations/instances_{data_split}2017.json"

run(datain, dataout)
