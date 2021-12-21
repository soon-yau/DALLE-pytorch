from pathlib import Path
from random import randint, choice, uniform

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import pandas as pd
from einops import rearrange
import numpy as np

from dalle_pytorch.pose_utils import keypoints_to_image, keypoints_to_heatmap, RotateScale, Crop, ToTensor, ConcatSamples

class PoseDatasetPickle(Dataset):
    def __init__(self,
                 folder,
                 pickle_file,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 threshold=0.25,
                 pose_format='image', # 'image' or 'keypoint' or 'heatmap'
                 output_shape=(128, 128),
                 merge_images=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.merge_images = merge_images
        self.pose_format = pose_format
        self.shuffle = shuffle
        df = pd.read_pickle(pickle_file)
        self.df = df[df.pose_score > threshold]
        self.root_dir = Path(folder)

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.concat = ConcatSamples()
        self.image_keypoint_transform = T.Compose([
            #RotateScale((-10,10),(1.0,1.1)),
            #Crop((0.0, 0.15)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        sample = self.df.iloc[ind]
        image_file = self.root_dir / sample.image
        descriptions = sample.text.copy()
        keypoints = sample.keypoints.copy()
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        image = PIL.Image.open(image_file)
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = np.array(image)


        if self.merge_images:
            ind = randint(0, self.__len__() - 1)
            sample = self.df.iloc[ind]
            image_file = self.root_dir / sample.image
            descriptions_2 = list(sample.text.copy())
            description_2 = choice(descriptions_2)
            keypoints_2 = sample.keypoints.copy()
            image_2 = PIL.Image.open(image_file)
            image_2 = image_2.convert('RGB') if image_2.mode != 'RGB' else image_2
            image_2 = np.array(image_2)

            two_images = np.vstack((np.expand_dims(image, 0), 
                                    np.expand_dims(image_2, 0)))
            two_keypoints = np.vstack((keypoints, keypoints_2))
            merged = self.concat({'image':two_images, 'keypoints':two_keypoints})
            image, keypoints = merged['image'], merged['keypoints']
            description = description + '. ' +  description_2

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        
        try:
            # augmentation, to do, multiple keypoints
            augmented = self.image_keypoint_transform({'image':image, 'keypoints':keypoints})
            image_tensor, keypoints = augmented['image'], augmented['keypoints']

            if self.pose_format == 'keypoint':
                pose_tensor = keypoints
            else:
                raise(ValueError, f'f pose format of {self.pose_format}is undefined')
                
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success

        return tokenized_text, image_tensor, pose_tensor

class PoseDataset(Dataset):
    def __init__(self,
                 folder,
                 pickle_file,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False,
                 threshold=0.25,
                 pose_format='image', # 'image' or 'keypoint' or 'heatmap'
                 output_shape=(128, 128)
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.pose_format = pose_format
        self.shuffle = shuffle
        df = pd.read_pickle(pickle_file)
        df = df[df.pose_score > threshold]
        
        # normalize keypoints to [0, 1] and flatten to (75,)
        keypoints = list(df.keypoints)
        for i in range(len(keypoints)):
            keypoints[i][:,:,:2] = keypoints[i][:,:,:2]/256
            #keypoints[i] = rearrange(keypoints[i], 'a b c -> a (b c)')       

        root_dir = Path(folder)
        image_dir = root_dir / "images/"
        text_dir = root_dir / "captions/"
        pose_dir = root_dir / "poses/"

        keys = list(df.image.map(lambda x: Path(x).stem))
        image_paths = list(df.image.map(lambda x: image_dir/x))
        #pose_paths = list(df.image.map(lambda x: pose_dir/x))
        text_paths = list(df.image.map(lambda x: text_dir/x.replace('.png','.txt')))
        
        #keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.image_files = dict(zip(keys, image_paths))
        self.text_files = dict(zip(keys, text_paths))
        self.keypoints =  dict(zip(keys, np.squeeze(keypoints)))
        #self.pose_files = dict(zip(keys, pose_paths))
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer

        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            #T.RandomResizedCrop(image_size,
            #                    scale=(self.resize_ratio, 1.),
            #                    ratio=(1., 1.)),
            T.ToTensor()
        ])

        self.image_keypoint_transform = T.Compose([
            #RotateScale((-10,10),(1.0,1.1)),
            Crop((0.0, 0.15)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]
        #pose_file = self.pose_files[key]
        kp = self.keypoints[key]
        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image = PIL.Image.open(image_file)
            image = image.convert('RGB') if image.mode != 'RGB' else image

            # augmentation
            augmented = self.image_keypoint_transform({'image':np.array(image), 'keypoints':kp})
            image_tensor, kp = augmented['image'], augmented['keypoints']

            if self.pose_format == 'image':
                # fix me
                pose_tensor = keypoints_to_image(kp)
                pose_tensor = self.image_transform(PIL.Image.fromarray(pose_tensor))
            elif self.pose_format == 'keypoint':
                pose_tensor = kp
            elif self.pose_format == 'heatmap':
                pose_tensor = keypoints_to_heatmap(kp, image_shape=(64,64), sigma=2.)
            else:
                raise(ValueError, f'f pose format of {self.pose_format}is undefined') 
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success

        return tokenized_text, image_tensor, pose_tensor

class MPIIDataset(Dataset):
    def __init__(self,
                 folder,
                 csv_file,                
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        df = pd.read_csv(csv_file)
        image_files = df['NAME'].tolist()
        texts = df['Activity'].tolist()

        self.image_files = {x.split('.jpg')[0]:Path(folder)/x for x in image_files}
        self.texts = {x.split('.jpg')[0]:text for x, text in zip(image_files, texts)}

        keys = (self.image_files.keys() & self.texts.keys())

        self.keys = list(keys)
        #self.text_files = {k: v for k, v in text_files.items() if k in keys}
        #self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        #text_file = self.text_files[key]
        image_file = self.image_files[key]
        text = self.texts[key]
        descriptions = text.split('\n')
        #descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor


class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]
        image_file = self.image_files[key]

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        #import pdb
        #pdb.set_trace()
        return tokenized_text, image_tensor
