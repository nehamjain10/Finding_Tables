import PIL
import torch, torchvision
from os import listdir
from tqdm.notebook import tqdm
from detectron2.engine import DefaultTrainer
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from PIL import Image

# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2.structures import BoxMode

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def save_detected_tables(image,bnd_boxes_tables):
    tables_detected=[]
    os.makedirs("output_images",exist_ok=True)
    count=0
    cv2.imwrite("output_images/tables_detected.png",image) 
    for i in bnd_boxes_tables:
        i=i.numpy()
        xmin=int(i[0])
        ymin=int(i[1])
        xmax=int(i[2])
        ymax=int(i[3])
        new_img=image[ymin:ymax,xmin:xmax]
        path=f"output_images/cropped_table{count}.png"
        cv2.imwrite(path,new_img)
        count+=1
        tables_detected.append(path)
    return tables_detected

def get_predictor(model_weights,threshold=0.75):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("data/All_X152.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_weights
    predictor = DefaultPredictor(cfg)
    return predictor,cfg


def Table_Detection(image_path,model_weights):
    bnd_boxes_tables=[]
    threshold=0.75
    im = cv2.imread(image_path)
    print("DETECTING TABLES................")
    while len(bnd_boxes_tables)==0 and threshold>=0.5:
        predictor,cfg=get_predictor(model_weights,threshold)
        outputs = predictor(im)
        bnd_boxes_tables=list(outputs["instances"].pred_boxes.to("cpu"))
        threshold=threshold-0.25
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im1=out.get_image()[:, :, ::-1]
    table_imgs=save_detected_tables(im1,bnd_boxes_tables)
    print("File path of saved tables are:  ",table_imgs)
    return table_imgs