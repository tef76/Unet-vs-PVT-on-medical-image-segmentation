from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2
import PIL
import skimage.io as io
import numpy as np

"""
    Read the dataSet

    Parameters
    ----------
    transforms :
        pytorch transformation.
    transforms2 : 
        pytorch transformation.
    path : 
        path of datas

"""
class SegmentationDataset(Dataset):
    
	def __init__(self, transforms, transforms2, path):   
		self.imagePaths = path
		self.maskPaths = path
		self.transform = transforms
		self.transform2 = transforms2

	def __len__(self):
		return 10
	def __getitem__(self, idx):
		r = '0000000' + str(idx + 1)
		image = io.imread(self.imagePaths + "\patient" + r[len(r) - 4:] +"\patient" + r[len(r) - 4:]+ "_4CH_ED.mhd", plugin='simpleitk')
		image = PIL.Image.fromarray(image[0]).convert("RGB")
		image = np.array(image)
		
		mask = io.imread(self.maskPaths + "\patient" + r[len(r) - 4:] +"\patient" + r[len(r) - 4:]+ "_4CH_ED_gt.mhd", plugin='simpleitk')
		mask = PIL.Image.fromarray(mask[0])
		mask = np.array(mask)
  
		image = self.transform(image).float()
		mask = self.transform2(mask).type(torch.LongTensor)

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		return (image, mask)


"""
    Train the model

    Parameters
    ----------
    model :
        model.
    datas : 
        training datas.
    optimizer : 
        Optimizer function.
    loss_f:
        loss function.

"""
def train(num_epochs, model, datas, optimizer, loss_f):

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(datas):

            optimizer.zero_grad()
            outputs = model(images)
        
            labels = torch.permute(labels,(1, 0, 2, 3))
            labels = torch.squeeze(labels)
      
            loss = loss_f(outputs, labels)  
            loss.retain_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
        print("Epoch:" + str(epoch) + " Loss: " + str(running_loss))

    return model