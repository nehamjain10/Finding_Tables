# Subex Hackathon

> An end to end Deep Learning approach for table detection and structure recognition from invoice documents

## Results: 1st Place out of 150+ participants
<br>

## 1. Introduction
Finding Tables is an automatic table recognition method for interpretation of tabular data in document images. We present an improved deep learning-based end to end approach for solving both problems of table detection and structure recognition using a finetuned model on our invoice data which is already pretrained on TableBank. Finding Tables is a Faster RCNN High-Resolution Network that detects the regions of tables. For our structure recognition we propose an entirely novel approach leveraging the SOTA methods in NLP. We basically feed in the image and we get question answers pairs and on the basis of that we can transform it into tables.


## 2. Setting it all up 

<br>
Setting up LayoutLM:
<p>
<pre>

git clone -b remove_torch_save https://github.com/NielsRogge/unilm.git
cd unilm/layoutlm
pip install unilm/layoutlm
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install ./transformers
</pre>

<b>Code is developed under following library dependencies</b> <br>

Torch==1.7.0+cu101 <br>
Torchvision==0.8.1+cu101<br>
Cuda = 10.1<br>

<br> To install Detectron V2
<pre>
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
</pre>

<br>


<br> Installing other dependencies
<pre>
pip install -r requirements.txt
</pre>

<br>

<p> Please download weights for Detectron V2 and LayoutLM and keep it data folder</p> 


Detectronv2 weights: <a href="https://drive.google.com/file/d/1Mn9E4Ylrc-p850TqhCbAK2OF6ltSxSUv/view?usp=sharing"> Detectron_finetuned_model_weights </a> 

LayoutLM <a href="https://drive.google.com/file/d/1ec84bdkRZ2aSatGBSbph3SHND-rsMF_A/view?usp=sharing"> LayoutLM weights </a>

## 3. Running inference

![alt text](imgs/Subex.jpg)


To test custom images on our model, go inside the folder and run the command "python run_inference.py 00017.PNG (path of image file)"


## 4. Examples

Original Image:

![alt text](imgs/00017.PNG)


Detecting Images:

![alt text](imgs/output_image.png)


Example from Structure Recognition:

![alt text](imgs/Finding_Tables.png)

![alt text](imgs/output1.jpg)


<br>
<p>

If you are having troubles getting it to work, please feel free to contact me or raise an issue

</p>
<br>
<p>
Neham (nehamjain2002@gmail.com) & Tanay (dixittanay@gmail.com)
