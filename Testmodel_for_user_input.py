from unetmodel import *
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
trans = transforms.ToPILImage()

user_input_dir = 'Dataset/UserInput/'
path = ''
#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    print('Initial GPU:',torch.cuda.current_device())
   
    torch.cuda.set_device(1)
    print('Selected GPU:', torch.cuda.current_device())

print('Using device: %s'%device)

def check_input_size(input_image,required_size):   
    x_dim,y_dim = input_image.size
    # y_dim = int(input_image.size()[3])
    if x_dim != required_size or y_dim != required_size :
        return False
    else:
        return True

def test_model_for_user_input(model):
    
    # print('Testing Unet Implementation for user input')
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

    bestESmodel.load_state_dict(torch.load(path+'models/bestESModel_'+name+'.ckpt'))
    bestESmodel = bestESmodel.to(device)
    bestESmodel.eval()

    required_size = 256
    dl = os.listdir(user_input_dir)
    num_images = len(dl)
    for index in range(0,num_images):
        img = Image.open(os.path.join(user_input_dir,dl[index]))
        # compare input dimensions to required dimension
        if check_input_size(img,required_size) == False:
            raise Exception('Invalid Input size')
        else:
            img = transforms.ToTensor()(img)
            img = img.to(device)
            #print(img.shape)
            img = img.unsqueeze(0)
            #print(img.shape)
            output = bestESmodel(img)
            output = output.cpu()
            plt.imshow(trans(output[0]),cmap="gray")
            plt.savefig(path+'TestOutput/'+name+'_%d.tif'%(index))
            # plt.show()
            plt.close()
    return True
    
###############################################################################################################################################
# parameters to select different models ==> Just change here. 
# name = 'unet'
name = 'unet_bn'
# name = 'unet_d'

test_model_for_user_input(name)
