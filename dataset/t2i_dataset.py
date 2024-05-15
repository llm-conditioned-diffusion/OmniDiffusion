import os
import json
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from .utils import *


def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        
        return image

    except Exception as e:
        # cases: image don't have getexif
        return image
    

class T2IDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        json_file=None,
        caption=None,
        size=1024,
        center_crop=False,
        random_flip=False,
        classifier_free_prob=0.0,
        img_path = None
    ):
        with open(json_file, 'r', encoding='utf-8') as file:
            self.df = json.load(file)
        print("data length = ", len(self.df))
        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.custom_crop = CustomCrop()
        self.classifier_free_prob = classifier_free_prob
        self.size = size 
        self.img_path = img_path
        self.caption = caption

        self.error_data = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        example = {}
        # TODO: modify obs
        while True:
            try:
                img_name = self.df[idx]['obs_path'].split('/')[-1]
                img_path = os.path.join(self.img_path, img_name)
                instance_image = correct_orientation(Image.open(img_path).convert('RGB') )
                text_caption = self.df[idx][self.caption]
                cropped_image, cropped_points = self.custom_crop.center_crop(instance_image, (self.size, self.size), face_boxes=None, body_boxes=None, return_xyxy=True)
                
                example["instance_images"] = self.image_transforms(cropped_image)
                example["instance_image_id"] = idx
                example["instance_pil_image"] = instance_image
                example["text_prompt"] = text_caption
                example["aes_score"] = 7.5
                example["cropped_points"] = cropped_points
                example["resized_body_boxes"] = None
                example["position"] = None
                example["original_size"] = (instance_image.height, instance_image.width) 
                example["crops_coords_top_left"] = (cropped_points[1], cropped_points[0])
                break

            except:
                item = self.df[idx]['obs_path'].split('/')[-1]
                if item not in self.error_data:
                    self.error_data.append(item)

                pass

        return example

    def get_error_data(self):
        return self.error_data