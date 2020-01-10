import os
import numpy as np
import rawpy
from PIL import Image
from skimage import io
from skimage.transform import resize
import warnings
import random
import unittest
import Generate_dataset
warnings.filterwarnings("ignore")

class Test_generate_dataset(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'Dataset/Data/'
        self.data_gt_dir = 'Dataset/Data_GT/'
        self.main_dir = 'Dataset/Main/Method1/'
        self.main_gt_dir = 'Dataset/Main_GT/Method1/'
        self.dl = os.listdir(self.main_dir)
        self.gt = os.listdir(self.main_gt_dir)
        self.required_size = 256
        self.num_images=4000
        self.data = os.listdir(self.data_dir)
        self.data_gt = os.listdir(self.data_gt_dir)
         
    def tearDown(self):
        pass
        
# 1
    def test_file_exists(self):
        self.assertEqual(Generate_dataset.folder_not_empty(self.dl,self.gt),True)

# 2
    def test_input_size(self):
        self.assertEqual(Generate_dataset.check_input_size(self.dl,self.gt,self.required_size),True)

# 3,5
    def test_subimage_count(self):
        num1,num2 = Generate_dataset.subimage_count()
        self.assertEqual((num1,num2),(4000,4000))

# 4,6
    def test_compare_subimage_size(self):
        self.assertEqual(Generate_dataset.compare_subimage_size(self.required_size),True)

# 7
# Implemented in generate_dataset.py

# 8
# Manual

# 9
# Manual


if __name__ == '__main__' :
    unittest.main()
