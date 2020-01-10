import Testmodel_for_user_input
import unittest
from PIL import Image
import datetime
import os

class Test_trainandtestmodel(unittest.TestCase):

    def test_bestmodel_location(self):
        self.best_model_dir = 'models/'
        self.dl = os.listdir(self.best_model_dir)
        self.assertIsNotNone(self.dl)
       
    def test_userinput_dimension(self):
        self.required_size = 256
        self.user_input_dir = 'Dataset/UserInput/'
        self.dl = os.listdir(self.user_input_dir)
        self.num_images = len(self.dl)
        self.img = Image.open(os.path.join(self.user_input_dir,self.dl[0]))
        self.assertTrue(Testmodel_for_user_input.check_input_size(self.img,self.required_size))
        
    def test_data_with_bestmodel(self):
        self.name = 'unet_bn'
        self.assertTrue(Testmodel_for_user_input.test_model_for_user_input(self.name))

    def test_runtime(self):
        self.name = 'unet_bn'
        starttime = datetime.datetime.now()
        self.assertTrue(Testmodel_for_user_input.test_model_for_user_input(self.name))
        endtime= datetime.datetime.now()
        self.assertLessEqual((endtime-starttime).total_seconds(),5)

    def test_modeloutput_location(self):
        self.output_dir = 'TestOutput/'
        self.dl = os.listdir(self.output_dir)
        self.assertIsNotNone(self.dl)
        
if __name__ == '__main__':
    unittest.main()
