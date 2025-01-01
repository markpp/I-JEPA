import numpy as np
from glob import glob
import zipfile
import io
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image

def unzip(zip_dir, output_dir):
    '''Extracts files in correct directories (train/val/test) based on zip file naming.'''
    
    zip_files = glob(os.path.join(zip_dir, '*.zip'))

    print(f"found {len(zip_files)} zip files: {zip_files}")
    
    for zip_file in zip_files:
        if 'train' in zip_file:
            split_folder = 'train'
        elif 'val' in zip_file:
            split_folder = 'val'
        elif 'test' in zip_file:
            split_folder = 'test'
        else:
            print(f'Zip file {zip_file} doesnt contain either train, val, or test so it is ignored')
            continue
        
        extract_dir = os.path.join(output_dir, split_folder)
        print(f'Extracting {zip_file} to {extract_dir}')
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)


class FrameFolderDataset(torch.utils.data.Dataset):
    '''
    Expects data divided into class folders
    '''
    def __init__(self, dataset_path, crop_size=224, num_classes=None, transforms=None, stage='train'):                
        self.dataset_path = dataset_path
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.transform = transforms
        self.stage = stage

        search_key = os.path.join(dataset_path, '*/*.jpg')
        print(f"Searching for images using search key {search_key}")
        self.img_files = sorted(glob(search_key))
        print('Found {} images'.format(len(self.img_files)))

    def __getitem__(self, idx):
       
        #Collect img and annotation at idx
        img_path = self.img_files[idx]

        if not os.path.isfile(img_path):
            print("File not found: {}".format(img_path))
            return None
        
        #load image
        img = cv2.imread(img_path)
        #img_h, img_w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Apply transforms
        if self.transform and self.stage=='train':
            img = self.transform(self.crop_size, self.crop_size)(image=img)['image']
        else:
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            img = img.astype(np.float32)/255.0

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).float()

        if self.num_classes is not None:
            # get name of class from folder name
            label = 0 if 'dont' in os.path.basename(os.path.dirname(img_path)) else 1
            #target = torch.as_tensor(label, dtype=torch.int64)
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            target[label] = 1.0
            return img, target
        else:
            return img
       
    
    def __len__(self):
        return len(self.img_files)

class MultiZipDataset(torch.utils.data.Dataset):
    '''
    Expects multiple zip files where images are divided into class folders.
    '''
    def __init__(self, zip_files, crop_size=224, num_classes=None, transforms=None, stage='train'):
        self.zip_files = zip_files  # List of zip file paths
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.transform = transforms
        self.stage = stage

        # Dictionary to hold open zip files
        self.zip_file_objs = []
        self.img_files = []
        self.label_distribution = []

        # Populate img_files with images from all zip files and count the label distribution (if num_classes is not None)
        for zip_file in self.zip_files:
            zip_obj = zipfile.ZipFile(zip_file, 'r')
            self.zip_file_objs.append(zip_obj)
            img_files_in_zip = [f for f in zip_obj.namelist() if f.endswith('.jpg')]

            if num_classes is not None:
                for img_path in img_files_in_zip:
                    if 'dont' in img_path:
                        self.label_distribution.append('dont')
                    elif 'spray' in img_path:
                        self.label_distribution.append('spray')

            # Store image paths along with the corresponding zip file index
            self.img_files.extend([(zip_file, img_path) for img_path in img_files_in_zip])

        print(f"Found {len(self.img_files)} images across {len(self.zip_files)} zip files.")

        if len(self.label_distribution) > 0:
            print(f"Label distribution: {np.unique(self.label_distribution, return_counts=True)}")

    def __getitem__(self, idx):
        # Get the zip file and image path
        zip_file, img_path_in_zip = self.img_files[idx]

        # Find the corresponding zip object based on the zip file
        zip_obj = next(z for z in self.zip_file_objs if z.filename == zip_file)

        if 1:
            # Open the image in memory directly from the zip
            with zip_obj.open(img_path_in_zip) as file:
                img_data = file.read()

            # Convert the image data to a numpy array for OpenCV
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode image from bytes

            if img is None:
                print(f"Could not decode image: {img_path_in_zip}")
                return None

            # Convert color space from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            try:
                # Try to open the image in memory directly from the zip
                with zip_obj.open(img_path_in_zip) as file:
                    img_data = file.read()

                # Use Pillow to open the image from bytes
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                img = np.array(img)

            except zipfile.BadZipFile as e:
                print(f"BadZipFile Error: {e} - Skipping {img_path_in_zip}")
                return None  # You can also choose to return a placeholder image or label
            
            except Exception as e:
                print(f"Error opening {img_path_in_zip}: {e}")
                return None  # You can also choose to return a placeholder image or label


        # Apply transforms if provided
        if self.transform and self.stage == 'train':
            img = self.transform(self.crop_size, self.crop_size)(image=img)['image']
        else:
            img = cv2.resize(img, (self.crop_size, self.crop_size))
            img = img.astype(np.float32) / 255.0

        # Convert to PyTorch tensor
        img = img.transpose((2, 0, 1))  # HWC to CHW
        img = torch.from_numpy(img).float()

        # If classification, infer label from folder name
        if self.num_classes is not None:
            label = 0 if 'dont' in img_path_in_zip else 1
            #target = torch.as_tensor(label, dtype=torch.int64)
            target = torch.zeros(self.num_classes, dtype=torch.float32)
            target[label] = 1.0
            return img, target
        else:
            return img

    def __len__(self):
        return len(self.img_files)

    def __del__(self):
        # Close all opened zip files when the dataset is deleted
        for zip_obj in self.zip_file_objs:
            zip_obj.close()


'''Dummy Dataset'''
class IJEPASupervisedDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 stage='train',
                 ):
        super().__init__()
        img1 =torch.randn(3, 224, 224)
        self.data = img1.repeat(100, 1, 1, 1)
        label = torch.tensor([0., 0., 0., 1., 0.])
        self.label = label.repeat(100, 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

'''Dummy Dataset'''
class IJEPADataset(Dataset):
    def __init__(self,
                 dataset_path,
                 stage='train',
                 ):
        super().__init__()
        img1 =torch.randn(3, 224, 224)
        self.data = img1.repeat(100, 1, 1, 1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
if __name__ == '__main__':
    # Usage example:
    zip_dir = '/home/markpp/Datasets/WeedJet/blobstorage/datasets/supervised/ipm'
    zip_files = glob(os.path.join(zip_dir, '*.zip'))

    task = 'classification'
    if task == 'classification':
        dataset = MultiZipDataset(zip_files=zip_files, crop_size=224, num_classes='classification')
    else:
        dataset = MultiZipDataset(zip_files=zip_files, crop_size=224, num_classes=None)

    for i in range(3):
        if task == 'classification':
            img, target = dataset[i]
            print(target.item())
        else:
            img = dataset[i]

        # display image
        img = img.numpy().transpose(1, 2, 0)*255
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        cv2.waitKey(0)