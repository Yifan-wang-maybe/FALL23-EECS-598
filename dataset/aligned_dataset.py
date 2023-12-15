import os
from dataset.base_dataset import BaseDataset, get_params, get_transform, get_transform_demo, augement_transform
from dataset.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import albumentations as ALBU


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    ### Yifan ###
    def __init__(self, opt, eval=0):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.test_phase = 0
        if eval == 0:
            phase = 'train'
        elif eval == 1:
            phase = 'val'
            self.test_phase = 1
        elif eval == 2:
            phase = 'test'
            self.test_phase = 1

        self.dir_AB = os.path.join(opt.dataroot, phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        
        self.dir_AB_demo = os.path.join(opt.dataroot_demo, phase)  # get the image directory
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths

        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB_path_demo = self.dir_AB_demo
        
        AB = Image.open(AB_path)
        label = AB_path[-20]
        AB = np.array(AB)  ### [0.65535]
        AB = Image.fromarray(np.uint8(AB))

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
          


        demo = np.load(AB_path_demo + '/' + AB_path[-20:-4] + '.npy')
        demo_A = demo[:16]
        demo_B = demo[16:]

        ### Diameter ###
        diameter_B = np.zeros(1).astype(np.float32)
        diameter_B[0] = demo_B[2] ## Yifan_2 ##
        if diameter_B[0] > 100:
            diameter_B[0] = demo_A[2]

        diameter_A = np.zeros(1).astype(np.float32)
        diameter_A[0] = demo_A[2] ## Yifan_2 ##


        
        demo_64_A = np.ones((w2,h),dtype=np.uint8)
        demo_A[2] = demo_A[2]*5
        if demo_A[2]>250:
            demo_A[2] = 250       
        demo_64_A[:,:] = demo_64_A[:,:] * demo_A[2]
        

        demo_64_B = np.ones((w2,h),dtype=np.uint8)
        demo_B[2] = demo_B[2]*5
        if demo_B[2] > 250:
            demo_B[2] = 250
        demo_64_B[:,:] = demo_64_B[:,:] * demo_B[2]

        
        ALMC_64_A = np.ones((w2,h,3),dtype=np.uint8)

        ### SCT_PRE_ATT ###
        demo_A[0] = demo_A[0]*25
        if demo_A[0]>250:
            demo_A[0] = 250   
        
        ### SCT_EPI_LOC ###
        demo_A[1] = demo_A[1]*25
        if demo_A[1]>250:
            demo_A[1] = 250 

        ### SCT_MARGINS ###
        demo_A[4] = demo_A[4]*25
        if demo_A[4]>250:
            demo_A[4] = 250 

        ALMC_64_A[:,:,0] = ALMC_64_A[:,:,0] * demo_A[0]
        ALMC_64_A[:,:,1] = ALMC_64_A[:,:,1] * demo_A[1]
        ALMC_64_A[:,:,2] = ALMC_64_A[:,:,2] * demo_A[0]
        

        

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))    
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))    
        demo_transform = get_transform_demo(self.opt, transform_params, grayscale=(self.input_nc == 1))
        

        # A[255,0]
        A = A_transform(A)
        B = B_transform(B)  
        if self.test_phase == 0: 
            Augement_transform = augement_transform()     
            AAAugement_transformed = Augement_transform(image = np.array(A), mask = np.array(B))       
            A = AAAugement_transformed['image'] 
            B = AAAugement_transformed['mask']
        
        
        '''
        ### Change input condition ###
        demo_64_A = np.concatenate((np.expand_dims(demo_64_A,2), ALMC_64_A), axis=2)
        
        print('ccccccccccccccccc',np.shape(demo_64_A))
        demo_64_A = demo_transform(demo_64_A).type(torch.FloatTensor)
        print('dddddddddddddddddd')
        demo_64_B = demo_transform(demo_64_B).type(torch.FloatTensor)   
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'demo_A' : demo_64_A, 'demo_B' : demo_64_B, 'diameter_A' : diameter_A,'diameter_B' : diameter_B, 'label':label}
        '''
        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'label':label}

        
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
