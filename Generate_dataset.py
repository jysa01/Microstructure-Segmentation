import os
import numpy as np
import rawpy
from PIL import Image
from skimage import io
from skimage.transform import resize
import warnings
import random
warnings.filterwarnings("ignore")

def folder_not_empty(dl,gt):
    if not dl or not gt:
        return False
    else:
        return True
        
def check_input_size(dl,gt,required_size):   
    num_images = len(dl)
    for index in range(0,num_images) :
        print('Verifying Image ', dl[index],' and ',gt[index])
        data = Image.open(os.path.join(main_dir,dl[index]))
        data_gt = Image.open(os.path.join(main_gt_dir,gt[index]))
        count = 0
        if data.size[0] < required_size or data.size[1]< required_size or data_gt.size[0]<required_size or data_gt.size[1]<required_size:
            count+=1
    if count>0:
        return False
    else:
        return True

def generate_unique_subimages(index,data,data_gt,required_size) :
    print("generate_unique_subimages")
    unique_coordinates = {}
    count = 0
    for i in range(0,2000) :
        x = random.randrange(0,data.size[0]-required_size)
        y = random.randrange(0,data.size[1]-required_size)
        while (x,y) in unique_coordinates:
            x = random.randrange(0,data.size[0]-required_size)
            y = random.randrange(0,data.size[1]-required_size)
        unique_coordinates[(x,y)] = 1
        count+=1
        tile_data = data.crop((x,y,x+required_size,y+required_size))
        tile_data.save(os.path.join(data_dir,str(index)+'_'+str(count)+'.tif'))
        tile_gt = data_gt.crop((x,y,x+required_size,y+required_size))
        tile_gt.save(os.path.join(data_gt_dir,str(index)+'_'+str(count)+'.tif'))
                    
def subimage_count():
    dl = os.listdir(data_dir)
    gt = os.listdir(data_gt_dir)
    num_images = len(dl)
    num_images_gt = len(gt)
    if num_images!=4000 or num_images_gt!=4000:
        raise Exception('Error in subimage count')
    else:
        return num_images,num_images_gt

def compare_subimage_size(required_size) :
    dl = os.listdir(data_dir)
    gt = os.listdir(data_gt_dir)
    num_images = len(dl)
    for index in range(0,num_images) :
        #print('Verifying Image ', dl[index],' and ',gt[index])
        data = Image.open(os.path.join(data_dir,dl[index]))
        data_gt = Image.open(os.path.join(data_gt_dir,gt[index]))
        if data.size[0] != required_size or data.size[1]!= required_size:
            raise Exception('Failed to create sub images of required size')
        else: 
            if data.size[0] != data_gt.size[0] or data.size[1]!= data_gt.size[1]:
                raise Exception('Dimensions mismatch between train and ground truth images')
    return True

def generate_sub_images(main_dir,main_gt_dir):
    #  Check if the files exist
    dl = os.listdir(main_dir)
    gt = os.listdir(main_gt_dir)
    result = folder_not_empty(dl,gt)
    if result :
        num_images = len(dl)
        required_size = 256
        if check_input_size(dl,gt,required_size) != False:
            for index in range(0,num_images) :
                print('Generating sub images for Image ', dl[index],' and ',gt[index])
                data = Image.open(os.path.join(main_dir,dl[index]))
                data_gt = Image.open(os.path.join(main_gt_dir,gt[index]))
                generate_unique_subimages(index,data,data_gt,required_size)
            num_train,num_ground_truth=subimage_count()
            if (num_train == num_ground_truth) :
                compare_subimage_size(required_size)

data_dir = 'Dataset/Data/'
data_gt_dir = 'Dataset/Data_GT/'
main_dir = 'Dataset/Main/Method1/'
main_gt_dir = 'Dataset/Main_GT/Method1/'
# generate_sub_images(main_dir,main_gt_dir)