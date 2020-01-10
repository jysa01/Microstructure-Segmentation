import os
from skimage import io
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import transforms
# from torch.autograd import Variable
# import torch.nn.functional as F

def get_tif(directory_name):
    correct_files = []
    for file in os.listdir(directory_name):
        if file.endswith(".tif"):
            correct_files.append(file)
    return correct_files

class my_datasetLoader(Dataset):

    def __init__(self, x_dir, y_dir, transform=None):

        self.x_dir =  x_dir
        # self.x_dir_files = sorted(os.listdir(x_dir))[1:]
        self.x_dir_files = get_tif(x_dir)

        self.y_dir = y_dir
        # self.y_dir_files = sorted(os.listdir(y_dir))[1:]
        self.y_dir_files = get_tif(y_dir)

        self.transform = transform

    def __len__(self):
        return len(self.x_dir_files)

    def __getitem__(self, idx):
        inp_img_name = os.path.join(self.x_dir, self.x_dir_files[idx])
        out_img_name = os.path.join(self.y_dir, self.y_dir_files[idx])

        in_image = io.imread(inp_img_name)
        out_image = io.imread(out_img_name)

        if self.transform:
            in_image = self.transform(in_image)
            out_image = self.transform(out_image)

        return [in_image, out_image]
