import os, csv, glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

class FlowerDataset(Dataset):
    def __init__(self, data_dir, trainValidTest, transform=None):
        
        self.data_dir = data_dir
        self.trainValidTest = trainValidTest
        
        # count folders in data_dir
        self.num_classes = sum(os.path.isdir(i) for i in os.listdir(data_dir))

        # read csv file, create codename_list. Remember it is a list of strings!
        with open(data_dir + 'flower_list.csv', 'rb') as csvfile:
            data = csv.reader(csvfile)
            
            # get the list of corresponding codename list
            codename_list = [row[0] for row in data]
            
            # ignore the first element which is the title
            self.codename_list = codename_list[1:]
                        
        # get the list of all jpg filenames in data_dir    
        image_filenames = glob.glob(data_dir + '/**/*.jpg')
                
        # another way to do it is to use the following, 
        # but it does not separate train, valid, test for us. 
        # see https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L57
        #self.imagefolder = datasets.ImageFolder(data_dir,transform=transform)
        
        if self.trainValidTest=='train':        
            # create train list: all jpg filenames which is less than 17.
            self.data_list = [name for name in image_filenames 
                              if int(os.path.basename(name).split('.')[0]) <= 17]
            
        elif self.trainValidTest=='valid':
            # create valid list: all jpg filenames which is equal to 18.
            self.data_list = [name for name in image_filenames 
                              if int(os.path.basename(name).split('.')[0]) == 18] 
            
        elif self.trainValidTest=='test':
        # create test list: all jpg filenames which is larger than 19. 
        # ^ need to change this, leave it for now
            self.data_list = [name for name in image_filenames 
                              if int(os.path.basename(name).split('.')[0]) >= 19] 
        
        self.num_samples = len(self.data_list)
        print('Total number of %s samples: %i' % (self.trainValidTest, self.num_samples))

        #assert(len(self.train_list) + len(self.valid_list) + len(self.test_list) == self.num_samples)
        
        self.transform = transform
        
    def __len__(self):
        #All subclasses should override __len__, that provides the size of the dataset        
        return self.num_samples
    
    def __getitem__(self, idx):
        # load image
        image_name = self.data_list[idx]
        image = Image.open(image_name)
        
        # load label
        class_label = image_name.split('/')[-2]
        label = self.codename_list.index(class_label)
                
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample
