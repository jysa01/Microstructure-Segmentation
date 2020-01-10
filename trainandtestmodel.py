import os
import torch
import torch.nn as nn
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
# from torchvision.transforms.transforms import ToPILImage as trans
from torch.autograd import Variable
import torch.nn.functional as F
from skimage.measure import compare_ssim
from unetmodel import *
from datasetLoader import my_datasetLoader
trans = transforms.ToPILImage()


path = ''

# n = 1234
# np.random.seed(n)
# torch.cuda.manual_seed_all(n)
# torch.manual_seed(n)

def weights_init(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight.data, mean=0.0, std=1e-3)
        m.bias.data.fill_(0.0)
        
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data) 
        m.bias.data.fill_(0.0)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# forwardTrans = transforms.Compose([ transforms.Normalize(mean = [ 0.5, 0.5, 0.5],
#                                                          std = [ 0.5, 0.5, 0.5]),
#                                     transforms.ToTensor()
#                                     ])

# invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.5, 1/0.5, 1/0.5 ]),
#                                 transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
#                                                      std = [ 1., 1., 1. ]),
#                                ])

#sitd_dataset = SeeingIntTheDarkDataset(path+'dataset/Sony/short_temp_down/', path+'dataset/Sony/long_temp_down/', transforms.ToTensor())
my_dataset = my_datasetLoader(path+'Dataset/Data/', path+'Dataset/Data_GT/', transforms.ToTensor())
print('Input Image Size:')
print(my_dataset[0][0].size())
inImageSize = my_dataset[0][0].size()
inImage_xdim = int(inImageSize[1])
inImage_ydim = int(inImageSize[2])

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    print('Initial GPU:',torch.cuda.current_device())
   
    torch.cuda.set_device(1)
    print('Selected GPU:', torch.cuda.current_device())

print('Using device: %s'%device)


### final params
num_training= 3500
num_validation = 350
num_test = 150

num_epochs = 100
learning_rate = 1e-3
learning_rate_decay = 0.99
reg = 0.005
batch_size = 10
data_aug=False

#### dev params
#num_training= 20
#num_validation = 7
#num_test = 7


mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(my_dataset, mask)

mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(my_dataset, mask)

mask = list(range(num_training + num_validation, num_training + num_validation + num_test))
test_dataset = torch.utils.data.Subset(my_dataset, mask)

data_aug_transforms = []
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
if data_aug == True:

    transformsList = [
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(.3,.7)),
        # transforms.ColorJitter(
        #         brightness=float(0.1*np.random.rand(1)),
        #         contrast=float(0.1*np.random.rand(1)),
        #         saturation=float(0.1*np.random.rand(1)),
        #         hue=float(0.1*np.random.rand(1))),
        #transforms.RandomGrayscale(p=0.1)   
                    ]

    data_aug_transforms = transformsList
    data_aug_transforms = transforms.RandomApply(transformsList, p=0.5)
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
# norm_transform = transforms.Compose([data_aug_transforms]+[transforms.ToTensor()])
# test_transform = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

def check_input_size(input_image,required_size):   
    x_dim = int(input_image.size()[2])
    y_dim = int(input_image.size()[3])
    if x_dim != required_size or y_dim != required_size :
        return False
    else:
        return True

def train_and_validateModel(name):

    # Initialize the model for this run
    if name == 'unet':
        model = unet()
    elif name == 'unet_bn':
        model = unet_bn()
    elif name == 'unet_d':
        model = unet_d()
    else:
        print('Method does not exist')
        return None

    model.apply(weights_init)

    # Print the model we just instantiated
    print(model)

    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

    # Train the model
    lr = learning_rate
    Loss = []                           #to save all the model losses
    valMSE = []
    valSSIM = []
    total_step = len(train_loader)

    for epoch in range(num_epochs):
        for i, (in_images, exp_images) in enumerate(train_loader):
            # Move tensors to the configured device
            in_images = in_images.type(torch.FloatTensor).to(device)
            exp_images = exp_images.type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(in_images)
            loss = criterion(outputs, exp_images)
            Loss.append(loss)               #save the loss so we can get accuracies later

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 20 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
        # Code to update the lr
        lr = lr * learning_rate_decay
        update_lr(optimizer, lr)

        with torch.no_grad():

            overallSSIM = 0
            MSE = 0
            for in_images, exp_images in val_loader:
                in_images = in_images.to(device)
                exp_images = exp_images.to(device)
                outputs = model(in_images)
                MSE += torch.sum((outputs - exp_images) ** 2)
                #outputs = outputs.cpu()
                outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy() 
                exp_images_np = exp_images.permute(0,2,3,1).cpu().numpy()

                SSIM = 0
                for i in range(len(outputs_np)):
                    SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)
                overallSSIM += SSIM
            
            total = len(val_dataset)
            valSSIM.append(overallSSIM/total)

            current_MSE = MSE/total
            valMSE.append(current_MSE)
            if current_MSE <= np.amin(valMSE):
                torch.save(model.state_dict(),path+'models/ESmodel'+str(epoch+1)+'.ckpt')

            print('Avg Validation MSE on all the {} Val images is: {} '.format(total, current_MSE))
            print('Avg Validation SSIM on all the {} Val images is: {} '.format(total, overallSSIM/total))

    # Training loss and Val MSE curves
    plt.plot(valMSE)
    title='AvgValMSE_vs_Epochs'
    plt.ylabel('Avg Validation MSE')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/_'+name+title+'.png')
    plt.show()
    plt.close()

    plt.plot(valSSIM)
    title='AvgValSSIM_vs_Epochs'
    plt.ylabel('Avg Validation SSIM')
    plt.xlabel('Epochs')
    plt.title(title)
    plt.savefig(path+'plots/_'+name+title+'.png')
    plt.show()
    plt.close()

    plt.plot(Loss)
    title='Loss_vs_Iterations'
    plt.ylabel('Loss')
    plt.xlabel('Iterations')
    plt.title(title)
    plt.savefig(path+'plots/_'+name+title+'.png')
    plt.show()
    plt.close()

    best_id = np.argmin(valMSE)
    return best_id, model




def test_model(best_id, model):
    
    print('Testing Unet Implementation')
    # Initialize the model for this run
    if name == 'unet':
        bestESmodel = unet()
    elif name == 'unet_bn':
        bestESmodel = unet_bn()
    elif name == 'unet_d':
        bestESmodel = unet_d()
    else:
        print('Method does not exist')
        return None

    bestESmodel.load_state_dict(torch.load(path+'models/ESmodel'+str(best_id+1)+'.ckpt'))
    #bestESmodel=torch.load(path+'models/ESmodel'+str(best_id+1)+'.ckpt')
    bestESmodel = bestESmodel.to(device)
    required_size = 256
    # last_model.eval()
    bestESmodel.eval()
    for input_image, _ in test_loader:
        # compare input dimensions to required dimension
        if check_input_size(input_image,required_size) == False:
            raise Exception('Invalid Input size')
        
    with torch.no_grad():

        overallSSIM = 0
        MSE = 0
        count = 0 
        for in_images, exp_images in test_loader:
            
            in_images = in_images.to(device)
            exp_images = exp_images.to(device)
            outputs = bestESmodel(in_images)

            MSE += torch.sum((outputs - exp_images) ** 2)

            # Visualize the output of the best model against ground truth
            in_images_py = in_images.cpu()
            outputs_py = outputs.cpu()
            exp_images_py = exp_images.cpu()
            
            reqd_size = int(in_images.size()[0])

            for i in range(reqd_size):

                img = in_images_py[i].numpy()
                nonZero = np.count_nonzero(img)
                count += 1 
                f, axarr = plt.subplots(1,3)
                title='Input ('+str(round((nonZero*100)/(inImage_xdim*inImage_ydim*3) , 2))+'% Non Zero) vs Model Output vs Ground truth'
                plt.suptitle(title)
                axarr[0].imshow(trans(in_images_py[i]),cmap="gray")
                axarr[1].imshow(trans(outputs_py[i]),cmap="gray")
                axarr[2].imshow(trans(exp_images_py[i]),cmap="gray")
                #edit
                #implot=plt.imshow(trans(outputs_py[i]))
                
                plt.savefig(path+'model_output/'+name+'_%d.tif'%(count))
                plt.close()

                if count % 10 == 0:
                    print('Saving image_%d.tif'%(count))

            outputs_np = outputs.permute(0, 2, 3, 1).cpu().numpy()
            exp_images_np = exp_images.permute(0, 2, 3, 1).cpu().numpy()

            SSIM = 0
            for i in range(len(outputs_np)):
                SSIM += compare_ssim(exp_images_np[i], outputs_np[i], multichannel=True)

            overallSSIM += SSIM

        total = len(test_dataset)
        print('Avg Test MSE of the best ES network on all the {} test images: {} '.format(total, MSE/total))
        print('Avg Test SSIM of the best ES network on all the {} test images: {} '.format(total, overallSSIM/total))
        print("Best Epoch with lowest Avg Validation MSE: ", best_id+1)

    torch.save(bestESmodel.state_dict(), path+'models/bestESModel_'+name+'.ckpt')
    return True
    



###############################################################################################################################################
# parameters to select different models ==> Just change here. 
# name = 'unet'
name = 'unet_bn'
# name = 'unet_d'
best_id = train_and_validateModel(name)
test_model(best_id, name)

