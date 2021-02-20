from models.model_layout import LayoutLM
from models.table_location_predictor import Table_Detection
import sys
import argparse
import cv2
import PIL
import warnings
warnings.filterwarnings("ignore")
from pre.preprocess import preprocess_image
# class to turn the keys of a dict into attributes (thanks Stackoverflow)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



if __name__=="__main__":


    args = {'local_rank': -1,
            'overwrite_cache': True,
            'data_dir': '/content/data',
            'model_name_or_path':'data/model_layoutLM.pt',
            'max_seq_length': 512,
            'model_type': 'layoutlm',}
    args = AttrDict(args)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Image file path (.PNG)')
    parser.add_argument('--layoutLM_model_path',default='data/model_layoutLM.pt', type=str,help='.pt file for model weights')
    parser.add_argument('--table_detection_model_path',default='data/model_detectronV2.pth', type=str,help='.pt file for model weights')
    parser.add_argument('--config',default="microsoft/layoutlm-base-uncased",type = str,help='model configure path json file')
    arguments = parser.parse_args()
    
    preprocess_image(arguments.image_path)
    
    detected_tables = Table_Detection("output_images/processed_image.png",arguments.table_detection_model_path)
    
    
    
    for path in detected_tables:    
        layout_model = LayoutLM(path, arguments.layoutLM_model_path, arguments.config)
        layout_model.setup_data(args)
        Image = layout_model.inference()

    