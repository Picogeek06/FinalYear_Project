import os
import shutil

from fastai.vision import SegmentationLabelList,ImageList,Image,open_image,Path,get_image_files,image2np,pil2tensor,get_transforms,ResizeMethod,to_np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 

from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import cascaded_union
from collections import defaultdict

from pathlib import Path

from tqdm import tqdm_notebook  

from fastai.basic_train import load_learner

from skimage import measure

from flask import Flask, request, render_template, send_from_directory
app = Flask(__name__)

class SegLabelListCustom(SegmentationLabelList):
    def open(self, fn): return open_mask(fn, div=True)

class SegItemListCustom(ImageList):
    _label_cls = SegLabelListCustom

def combo_loss(pred, targ):
    bce_loss = CrossEntropyFlat(axis=1)
    return bce_loss(pred,targ) + dice_loss(pred,targ)

def dice_loss(input, target):
    smooth = 1.
    input = input[:,1,None].sigmoid()
    iflat = input.contiguous().view(-1).float()
    tflat = target.view(-1).float()
    intersection = (iflat * tflat).sum()
    return (1 - ((2. * intersection + smooth) / ((iflat + tflat).sum() +smooth)))

def get_pred(learn,img):
    #t_img = Image(pil2tensor(tile,np.float32))
    a,b,outputs = learn.predict(img)
    im = image2np(outputs.sigmoid())
    return im



APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
    
    segmentaion_model_path = './Segmentation_model'
    learn = load_learner(segmentaion_model_path)
    image_path = Path(f'./images/{filename}')
    img2,img = cv2.imread(destination),open_image(destination)
    mask = get_pred(learn,img)
    contours = measure.find_contours(mask[:,:,0], 0.002)
    contour1 = []
    for i in range(len(contours)):
      ele = []
      for j in range(len(contours[i])):
        y,x = contours[i][j]
        ele.append([x,y])
      contour1.append(np.array(ele))
    poly1=[]
    for i in contour1:
      c = np.expand_dims(i.astype(np.float32), 1)
      # Convert it to UMat obj
      c = cv2.UMat(c)
      area = cv2.contourArea(c)
      if area>50:
        #poly.append(Polygon(i))
        poly1.append(i)
    df_contour = pd.DataFrame()
    _id = []
    img_names = []
    contour_data = []
    for i in range(len(poly1)):
      x,y,w,h = cv2.boundingRect(np.int32([poly1[i]]))
      ROI = img2[y:y+h+50, x:x+w+50]
      _id.append(i)
      img_names.append(f'IMG_025_0{i}.png')
      contour_data.append(poly1[i])
      cv2.imwrite(f"./crop_file/IMG_025_0{i}.png",ROI)
    df_contour['id'] = _id
    df_contour['image_names'] = img_names
    df_contour['contour'] = contour_data
    classification_model_path = './classification_model'
    learn_classi = load_learner(classification_model_path)
    tfms = get_transforms(flip_vert=True, max_rotate=0.2, max_warp=0., max_zoom=1.1, max_lighting=0.4)
    test_path = Path(f'./crop_file')
    test_fns = [o for o in sorted(test_path.iterdir()) if '.png' in o.name]   
    preds = []
    pred_classes = []
    image_names = []
    for fn in tqdm_notebook(test_fns):
      try:
        img = open_image(fn)
        img = img.apply_tfms(tfms[1],resize_method=ResizeMethod.SQUISH, padding_mode='zeros')
        pred_class,pred_idx,outputs = learn_classi.predict(img)
        image_names.append(fn.name)
        preds.append(list(to_np(outputs)))
        pred_classes.append(str(pred_class))
      except Exception as exc:
        print(f'{exc}') 
        preds.append([-1,-1,-1,-1])
        pred_classes.append('error')
    df_pred_img = pd.DataFrame()
    df_pred_img['id'] = range(len(image_names))
    df_pred_img['image_names'] = image_names
    df_pred_img['predict_classes'] = pred_classes 

    fig, ax = plt.subplots(figsize=(10,10))
    for i,j in enumerate(image_names):
      x = df_pred_img[df_pred_img['image_names']==j]['predict_classes'].values[0]
      y = df_contour[df_contour['image_names']==j]['contour'].values[0]
      if x=='Complete':
        final_plot = cv2.drawContours(img2, [y.astype(int)], 0, (0,255,0), 3)
      elif x=='Incomplete':
        final_plot = cv2.drawContours(img2, [y.astype(int)], 0, (255,0,0), 3)
      elif x=='Foundation':
        final_plot = cv2.drawContours(img2, [y.astype(int)], 0, (0,0,255), 3)
      else:
        final_plot = cv2.drawContours(img2, [y.astype(int)], 0, (255,255,255), 3)
    cv2.imwrite('./images/final_plot.png',final_plot)

    return render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)

@app.route("/delete")
def delete():
    target = shutil.rmtree('images')
    #print(target)
    #if os.path.isdir(target):
    	#os.rmdir('./images')
    shutil.rmtree('crop_file')
    directory = "images"
    parent_dir = "./"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    directory = "crop_file"
    parent_dir = "./"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    print('deleted yo')
    return render_template("delete.html")

            

if __name__ == "__main__":
    app.run(debug=True)

