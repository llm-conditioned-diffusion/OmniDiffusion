import re
import torch 
import transformers

from typing import Dict
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataclasses import dataclass
from torchvision import transforms

from constants import *


def image_grid(imgs, rows, cols):
    
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))

    return grid


class CustomCrop:
    def __init__(self, crop_threshold=0.5, additional_shift_ratio=0.25):
        self.crop_threshold = crop_threshold
        self.additional_shift_ratio = additional_shift_ratio

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path)

    def is_within_threshold(self, center, center_crop, wh):
        return abs(center - center_crop) / wh < self.crop_threshold

    def is_fully_within(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])

        return abs(interArea - boxAArea) < 1e-5

    def resize_by_short_edge(self, image, ratio):
        width, height = image.size
        new_width = round(width * ratio)
        new_height = round(height * ratio)

        return image.resize((new_width, new_height))

    def resize_image_and_boxes(self, image, target_size, face_boxes=None, body_boxes=None):
        width, height = image.size
        if height > width:
            ratio = target_size / width
        else:
            ratio = target_size / height

        resized_image = self.resize_by_short_edge(image, ratio)
        
        # 如果没有提供face_boxes和body_boxes，只返回调整后的图像
        if face_boxes is None and body_boxes is None:
            return resized_image

        resized_face_boxes = [[x*ratio for x in box] for box in face_boxes] if face_boxes else []
        resized_body_boxes = [[box[0]*ratio, box[1]*ratio, box[2]*ratio, box[3]*ratio, box[4], box[5]] for box in body_boxes] if body_boxes else []

        return resized_image, resized_face_boxes, resized_body_boxes


    def center_crop(self, image, target_size, face_boxes=None, body_boxes=None, return_xyxy=False):
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        crop_width, crop_height = target_size

        x1 = center_x - crop_width // 2
        x2 = center_x + crop_width // 2
        y1 = center_y - crop_height // 2
        y2 = center_y + crop_height // 2
        center_crop_box = [x1, y1, x2, y2]

         # 对face_boxes进行处理
        if face_boxes:
            # 如果只有一个face box
            if len(face_boxes) == 1:
                box = face_boxes[0]
                if not self.is_fully_within(box, center_crop_box):
                    center_x, center_y = self.adjust_center(center_x, center_y, box, crop_width, crop_height, width, height)
        # 对body_boxes进行处理
        elif body_boxes:
            # 如果有多个body box，选择置信度最高的
            if len(body_boxes) > 1:
                body_boxes = [max(body_boxes, key=lambda x: x[4])]
            
            # 如果只有一个body box
            if len(body_boxes) == 1:
                box = body_boxes[0]
                if not self.is_fully_within(box, center_crop_box):
                    center_x, center_y = self.adjust_center(center_x, center_y, box, crop_width, crop_height, width, height)

        x1 = center_x - crop_width // 2
        x2 = center_x + crop_width // 2
        y1 = center_y - crop_height // 2
        y2 = center_y + crop_height // 2

        if not return_xyxy:
            return image.crop((x1, y1, x2, y2))
        else:
            return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)

    def adjust_center(self, center_x, center_y, box, crop_width, crop_height, width, height):
        if width > height:
            if box[0] < center_x - crop_width // 2:
                new_center_x = round(box[0] + crop_width // 2)
                if self.is_within_threshold(center_x, new_center_x, width):
                    center_x = new_center_x
            elif box[2] > center_x + crop_width // 2:
                new_center_x = round(box[2] - crop_width // 2)
                if self.is_within_threshold(center_x, new_center_x, width):
                    center_x = new_center_x
        else:
            if box[1] < center_y - crop_height // 2:
                new_center_y = round(box[1] + crop_height // 2)
                if self.is_within_threshold(center_y, new_center_y, height):
                    center_y = new_center_y
                    additional_shift = round(crop_height * self.additional_shift_ratio)
                    if center_y - additional_shift - crop_height // 2 < 0: 
                        center_y = crop_height // 2
                    else:
                        center_y -= additional_shift
            elif box[3] > center_y + crop_height // 2:
                new_center_y = round(box[3] - crop_height // 2)
                if self.is_within_threshold(center_y, new_center_y, height):
                    center_y = new_center_y

        return center_x, center_y


def collate_fn(examples):
    input_ids = None
    if "instance_prompt_ids" in examples[0]:
        input_ids = [example["instance_prompt_ids"] for example in examples]
    input_position = [example["position"] for example in examples]
    text_prompt = [example["text_prompt"] for example in examples]
    instance_image_id = [example["instance_image_id"] for example in examples]
    instance_pil_image = [example["instance_pil_image"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    original_size = [example["original_size"] for example in examples]
    aes_score = [example["aes_score"] for example in examples]
    crops_coords_top_left = [example["crops_coords_top_left"] for example in examples]
    cropped_points = [example["cropped_points"] for example in examples]
    resized_body_boxes = [example["resized_body_boxes"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    if input_ids is not None:
        input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "input_position": input_position,
        "text_prompt": text_prompt,
        "instance_image_id": instance_image_id, 
        "instance_pil_image": instance_pil_image,
        "original_size": original_size, 
        "crops_coords_top_left" : crops_coords_top_left,
        "aes_score" : aes_score, 
        "cropped_points" : cropped_points, 
        "resized_body_boxes" : resized_body_boxes, 
    }
    
    return batch


class CleanCaption(object):

    def evaluate(self, caption):
        if caption is None:
            return None

        # 使用正则表达式删除网址，包括全网址和没有http或www前缀的网址
        url_pattern = r"(https?:\/\/)?(www\.)?([-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))"
        caption = re.sub(url_pattern, '', caption)

        # 删除人名，假设人名跟在@字符后面
        name_pattern = r"@\w+"
        caption = re.sub(name_pattern, '', caption)

        # 删除HTML代码
        html_pattern = r"<[^>]*>"
        caption = re.sub(html_pattern, '', caption)

        # 删除无意义的语气词
        meaningless_words_pattern = r"\b(Mmmm...|Emmm...)\b"
        caption = re.sub(meaningless_words_pattern, '', caption)

        # 删除图书页码
        page_number_pattern = r"Page \d+-\d+"
        caption = re.sub(page_number_pattern, '', caption)
        
        # 删除图书页码
        page_number_pattern = r"Page \d+"
        caption = re.sub(page_number_pattern, '', caption)
    
        # 删除图书页码
        page_number_pattern = r"Chapter \d+"
        caption = re.sub(page_number_pattern, '', caption)

        # 删除网站名
        website_name_pattern = r"(Google Search|YouTube)"
        caption = re.sub(website_name_pattern, '', caption)
        
        ##删除特殊字符
        special_char_pattern = r'[●~♥]'
        caption = re.sub(special_char_pattern, '', caption)
        
        #删除特殊符号
        try:  
            pattern = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')  
        except re.error:  
            pattern = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
        caption = pattern.sub('', caption)

        #“#”+数字
        special_char_pattern = r'(#\d+|MLS®?#? \d+|#)'
        caption = re.sub(special_char_pattern, '', caption)
        
        #MLS®#+数字
        special_char_pattern = r'(#\d+|MLS®?#? \d+|#)'
        caption = re.sub(special_char_pattern, '', caption)

        return caption


@dataclass
class AligmentDataCollator(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        # print(instances)
        # if isinstance(instances[0], Dict):
        key_list = instances[0].keys()
        output_dict = {}
        for key in key_list:
            # Need to remove the batch dimension
            if key in ['input_ids', 'attention_mask', 'labels', 'input_ids_en', 'attention_mask_en', 'labels_en', 'input_ids_zh', 'attention_mask_zh', 'labels_zh']:
                output_value = [instance[key][0] for instance in instances]
            else:
                output_value = [instance[key] for instance in instances]
            if "input_ids" in key:
                # print(output_value, instances)
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            elif "labels" in key:
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=-100)
            elif "attention_mask" in key:
                output_value = torch.nn.utils.rnn.pad_sequence(output_value, batch_first=True, padding_value=0)
            elif key == 'input_images':
                output_value = [v.to(PRECISION) for v in output_value]
            elif key == 'output_image':
                output_value = torch.concat(output_value).to(PRECISION)
            elif key == 'output_image_feature':
                output_value = torch.concat(output_value)
            elif key == 'sd_image_clip':
                output_value = torch.concat(output_value).to(PRECISION)
            output_dict[key] = output_value
        return output_dict