from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pytesseract
from transformers import LayoutLMForTokenClassification,LayoutLMTokenizer,LayoutLMConfig
import torch
#from utils import *

def openImage(path):
    image = Image.open(path)
    return image


def getOCRdata(image):
    width, height = image.size
    w_scale = 1000/width
    h_scale = 1000/height

    ocr_df = pytesseract.image_to_data(image, output_type='data.frame') \
                
    ocr_df = ocr_df.dropna().assign(left_scaled = ocr_df.left*w_scale,
                        width_scaled = ocr_df.width*w_scale,
                        top_scaled = ocr_df.top*h_scale,
                        height_scaled = ocr_df.height*h_scale,
                        right_scaled = lambda x: x.left_scaled + x.width_scaled,
                        bottom_scaled = lambda x: x.top_scaled + x.height_scaled)

    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    return ocr_df, width, height

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def getBoxes(df, width, height):
    words = list(df.text)
    coordinates = df[['left', 'top', 'width', 'height','text']]
    actual_boxes = []; named_boxes =[]
    for idx, row in coordinates.iterrows():
        x, y, w, h, text = tuple(row) # the row comes in (left, top, width, height) format
        actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+widght, top+height) to get the actual box 
        text_box =actual_box +[text]
        actual_boxes.append(actual_box)
        named_boxes.append(text_box )

    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))
    return boxes, words,actual_boxes




def convert_example_to_features(image, words, boxes, actual_boxes, tokenizer, args, cls_token_box=[0, 0, 0, 0],
                                 sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0]):
      width, height = image.size

      tokens = []
      token_boxes = []
      actual_bboxes = [] # we use an extra b because actual_boxes is already used
      token_actual_boxes = []
      for word, box, actual_bbox in zip(words, boxes, actual_boxes):
          word_tokens = tokenizer.tokenize(word)
          tokens.extend(word_tokens)
          token_boxes.extend([box] * len(word_tokens))
          actual_bboxes.extend([actual_bbox] * len(word_tokens))
          token_actual_boxes.extend([actual_bbox] * len(word_tokens))

      # Truncation: account for [CLS] and [SEP] with "- 2". 
      special_tokens_count = 2 
      if len(tokens) > args.max_seq_length - special_tokens_count:
          tokens = tokens[: (args.max_seq_length - special_tokens_count)]
          token_boxes = token_boxes[: (args.max_seq_length - special_tokens_count)]
          actual_bboxes = actual_bboxes[: (args.max_seq_length - special_tokens_count)]
          token_actual_boxes = token_actual_boxes[: (args.max_seq_length - special_tokens_count)]

      # add [SEP] token, with corresponding token boxes and actual boxes
      tokens += [tokenizer.sep_token]
      token_boxes += [sep_token_box]
      actual_bboxes += [[0, 0, width, height]]
      token_actual_boxes += [[0, 0, width, height]]
      
      segment_ids = [0] * len(tokens)

      # next: [CLS] token
      tokens = [tokenizer.cls_token] + tokens
      token_boxes = [cls_token_box] + token_boxes
      actual_bboxes = [[0, 0, width, height]] + actual_bboxes
      token_actual_boxes = [[0, 0, width, height]] + token_actual_boxes
      segment_ids = [1] + segment_ids

      input_ids = tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = args.max_seq_length - len(input_ids)
      input_ids += [tokenizer.pad_token_id] * padding_length
      input_mask += [0] * padding_length
      segment_ids += [tokenizer.pad_token_id] * padding_length
      token_boxes += [pad_token_box] * padding_length
      token_actual_boxes += [pad_token_box] * padding_length

      assert len(input_ids) == args.max_seq_length
      assert len(input_mask) == args.max_seq_length
      assert len(segment_ids) == args.max_seq_length
      #assert len(label_ids) == args.max_seq_length
      assert len(token_boxes) == args.max_seq_length
      assert len(token_actual_boxes) == args.max_seq_length
      
      return input_ids, input_mask, segment_ids, token_boxes, token_actual_boxes

class LayoutLM(torch.nn.Module):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self,image_path, model_path,config_path, num_labels =13, args= None):
        super(LayoutLM, self).__init__()
        self.image = openImage(image_path)
        self.args = args
        self.tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

        config = LayoutLMConfig.from_pretrained(config_path)
        self.model = LayoutLMForTokenClassification.from_pretrained(model_path, config=config)
        self.model.to(device)

        self.input_ids= None; self.attention_mask= None; self.token_type_ids= None; self.bboxes= None; self.token_actual_boxes =None 

    def setup_data(self, args):

        ocr_df, width, height = getOCRdata(self.image)
        boxes, words,actual_boxes = getBoxes(ocr_df, width, height)
        input_ids, input_mask, segment_ids, token_boxes, self.token_actual_boxes = \
            convert_example_to_features(image=self.image, words=words, boxes=boxes, actual_boxes=actual_boxes, tokenizer=self.tokenizer, args=self.args)

        self.input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        self.attention_mask = torch.tensor(input_mask, device=device).unsqueeze(0)
        self.token_type_ids = torch.tensor(segment_ids, device=device).unsqueeze(0)
        self.bbox = torch.tensor(token_boxes, device=device).unsqueeze(0)
        self.outputs = self.model(input_ids=self.input_ids, bbox=self.bbox, attention_mask=self.attention_mask, token_type_ids=self.token_type_ids)
        assert self.outputs != None ,"Setup failed"
        print('Setup done')

    def inference(self):
        token_predictions = self.outputs.logits.argmax(-1).squeeze().tolist()
        word_level_predictions = [] # let's turn them into word level predictions
        final_boxes = []
        for id, token_pred, box in zip(self.input_ids.squeeze().tolist(), token_predictions, self.token_actual_boxes):
            if (self.tokenizer.decode([id]).startswith("##")) or (id in [self.tokenizer.cls_token_id, 
                                                                    self.tokenizer.sep_token_id, 
                                                                    self.tokenizer.pad_token_id]):
                # skip prediction + bounding box

                continue
            else:
                word_level_predictions.append(token_pred)
                final_boxes.append(box)

        label2color = {'i-question':'blue', 'i-answer':'green', 'i-header':'orange','b-question':'blue',\
            'b-answer':'green', 'b-header':'orange', 'e-question':'blue', 'e-answer':'green', 'e-header':'orange',\
               's-question':'blue', 's-answer':'green', 's-header':'orange','other':'violet'}

        label_map ={0: 'B-ANSWER',
                    1: 'B-HEADER',
                    2: 'B-QUESTION',
                    3: 'E-ANSWER',
                    4: 'E-HEADER',
                    5: 'E-QUESTION',
                    6: 'I-ANSWER',
                    7: 'I-HEADER',
                    8: 'I-QUESTION',
                    9: 'O',
                    10: 'S-ANSWER',
                    11: 'S-HEADER',
                    12: 'S-QUESTION'}

        draw = ImageDraw.Draw(self.image)
        for prediction, box in zip(word_level_predictions, final_boxes):
            predicted_label = iob_to_label(label_map[prediction]).lower()
            if predicted_label !='other':
                draw.rectangle(box, outline=label2color[predicted_label])
                draw.text((box[0] + 20, box[1] - 20), text=predicted_label, fill=label2color[predicted_label], font=font)

        return Image




