from typing import Callable, Optional
from torchvision.datasets.vision import VisionDataset
# importing necessary packages
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import torch, os
from torchvision import transforms
from torch.utils import data
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torchsummary import summary


# _coarse_labels = {
#     1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4,
#     11: 4, 12: 4, 13: 4, 14: 5, 15: 5, 16: 5, 17: 6, 18: 7, 19: 7,
#     20: 8, 21: 9, 22: 10, 23: 11, 24: 11, 25: 11, 26: 12, 27: 12,
#     28: 13, 29: 14, 30: 14, 31: 15, 32: 15, 33: 15, 34: 16, 35: 17,
#     36: 18, 37: 19, 38: 19, 39: 19, 40: 19, 41: 19, 42: 19, 43: 19,
#     44: 20, 45: 21, 46: 22, 47: 23, 48: 23, 49: 24, 50: 25, 51: 25,
#     52: 25, 53: 25, 54: 26, 55: 26, 56: 26, 57: 26, 58: 27, 59: 28,
#     60: 28, 61: 28, 62: 28, 63: 28, 64: 28, 65: 28, 66: 28, 67: 29,
#     68: 29, 69: 29, 70: 29, 71: 30, 72: 30, 73: 31, 74: 31, 75: 31,
#     76: 32, 77: 33, 78: 33, 79: 34, 80: 34, 81: 34, 82: 34, 83: 34,
#     84: 35, 85: 36, 86: 37, 87: 38, 88: 39, 89: 40, 90: 40, 91: 41,
#     92: 42, 93: 43, 94: 44, 95: 45, 96: 45, 97: 45, 98: 45, 99: 46,
#     100: 47, 101: 47, 102: 48, 103: 48, 104: 49, 105: 50, 106: 51,
#     107: 52, 108: 52, 109: 53, 110: 54, 111: 55, 112: 55, 113: 56,
#     114: 56, 115: 56, 116: 56, 117: 56, 118: 56, 119: 56, 120: 56,
#     121: 56, 122: 56, 123: 56, 124: 56, 125: 56, 126: 56, 127: 56,
#     128: 56, 129: 56, 130: 56, 131: 56, 132: 56, 133: 56, 134: 57,
#     135: 58, 136: 58, 137: 58, 138: 58, 139: 59, 140: 59, 141: 60,
#     142: 60, 143: 60, 144: 60, 145: 60, 146: 60, 147: 60, 148: 61,
#     149: 62, 150: 62, 151: 63, 152: 63, 153: 63, 154: 63, 155: 63,
#     156: 63, 157: 63, 158: 64, 159: 64, 160: 64, 161: 64, 162: 64,
#     163: 64, 164: 64, 165: 64, 166: 64, 167: 64, 168: 64, 169: 64,
#     170: 64, 171: 64, 172: 64, 173: 64, 174: 64, 175: 64, 176: 64,
#     177: 64, 178: 64, 179: 64, 180: 64, 181: 64, 182: 64, 183: 65,
#     184: 65, 185: 66, 186: 66, 187: 67, 188: 67, 189: 67, 190: 67,
#     191: 67, 192: 67, 193: 68, 194: 68, 195: 68, 196: 68, 197: 68,
#     198: 68, 199: 68, 200: 69
# }

# _coarse_labels = {
#     1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3,
#     11: 3, 12: 3, 13: 3, 14: 4, 15: 4, 16: 4, 17: 4, 18: 2, 19: 2,
#     20: 3, 21: 3, 22: 3, 23: 2, 24: 2, 25: 2, 26: 3, 27: 3, 28: 10,
#     29: 10, 30: 10, 31: 5, 32: 5, 33: 5, 34: 14, 35: 14, 36: 2, 37: 7,
#     38: 7, 39: 7, 40: 7, 41: 7, 42: 7, 43: 7, 44: 2, 45: 2, 46: 6,
#     47: 14, 48: 14, 49: 3, 50: 2, 51: 2, 52: 2, 53: 2, 54: 14, 55: 14,
#     56: 14, 57: 14, 58: 8, 59: 8, 60: 8, 61: 8, 62: 8, 63: 8, 64: 8,
#     65: 8, 66: 8, 67: 9, 68: 9, 69: 9, 70: 9, 71: 8, 72: 8, 73: 10,
#     74: 10, 75: 10, 76: 4, 77: 7, 78: 7, 79: 5, 80: 5, 81: 5, 82: 5,
#     83: 5, 84: 8, 85: 11, 86: 6, 87: 6, 88: 11, 89: 6, 90: 6, 91: 15,
#     92: 3, 93: 10, 94: 4, 95: 3, 96: 3, 97: 3, 98: 3, 99: 15, 100: 6,
#     101: 6, 102: 7, 103: 7, 104: 15, 105: 15, 106: 2, 107: 10, 108: 10, 109: 12,
#     110: 15, 111: 15, 112: 15, 113: 4, 114: 4, 115: 4, 116: 4, 117: 4, 118: 4,
#     119: 4, 120: 4, 121: 4, 122: 4, 123: 4, 124: 4, 125: 4, 126: 4, 127: 4,
#     128: 4, 129: 4, 130: 4, 131: 4, 132: 4, 133: 4, 134: 11, 135: 11, 136: 11,
#     137: 11, 138: 11, 139: 12, 140: 12, 141: 8, 142: 8, 143: 8, 144: 8, 145: 8,
#     146: 8, 147: 8, 148: 3, 149: 15, 150: 15, 151: 12, 152: 12, 153: 12, 154: 12,
#     155: 12, 156: 12, 157: 12, 158: 12, 159: 12, 160: 12, 161: 12, 162: 12, 163: 12,
#     164: 12, 165: 12, 166: 12, 167: 12, 168: 12, 169: 12, 170: 12, 171: 12, 172: 12,
#     173: 12, 174: 12, 175: 12, 176: 12, 177: 12, 178: 12, 179: 12, 180: 12, 181: 12,
#     182: 12, 183: 12, 184: 12, 185: 11, 186: 11, 187: 13, 188: 13, 189: 13, 190: 13,
#     191: 13, 192: 13, 193: 13, 194: 13, 195: 13, 196: 13, 197: 13, 198: 13, 199: 13,
#     200: 12
# }


_coarse_labels = {1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:2, 8:2, 9:3, 10:3,
11:3, 12:3, 13:3, 14:4, 15:4, 16:4, 17:3, 18:4, 19:4, 20:4,
21:3, 22:3, 23:4, 24:4, 25:4, 26:5, 27:5, 28:3, 29:6, 30:6,
31:7, 32:7, 33:7, 34:8, 35:8, 36:9, 37:10, 38:10, 39:10, 40:10,
41:10, 42:10, 43:10, 44:9, 45:9, 46:8, 47:8, 48:8, 49:3, 50:11,
51:11, 52:11, 53:11, 54:12, 55:12, 56:12, 57:12, 58:13, 59:14, 60:14,
61:14, 62:14, 63:14, 64:14, 65:14, 66:14, 67:15, 68:15, 69:15, 70:3,
71:16, 72:16, 73:16, 74:16, 75:16, 76:3, 77:17, 78:17, 79:18, 80:18,
81:18, 82:18, 83:18, 84:13, 85:3, 86:13, 87:19, 88:19, 89:19, 90:19,
91:19, 92:19, 93:19, 94:19, 95:20, 96:20, 97:20, 98:20, 99:20, 100:21,
101:21, 102:22, 103:22, 104:22, 105:22, 106:22, 107:22, 108:22, 109:22, 110:22,
111:23, 112:23, 113:24, 114:24, 115:24, 116:24, 117:24, 118:24, 119:24, 120:24,
121:24, 122:24, 123:24, 124:24, 125:24, 126:24, 127:24, 128:24, 129:24, 130:24,
131:24, 132:24, 133:24, 134:25, 135:26, 136:26, 137:26, 138:26, 139:25, 140:25,
141:27, 142:27, 143:27, 144:27, 145:27, 146:27, 147:27, 148:27, 149:28, 150:28,
151:29, 152:29, 153:29, 154:29, 155:29, 156:29, 157:29, 158:30, 159:30, 160:30,
161:30, 162:30, 163:30, 164:30, 165:30, 166:30, 167:30, 168:30, 169:30, 170:30,
171:30, 172:30, 173:30, 174:30, 175:30, 176:30, 177:30, 178:30, 179:30, 180:30,
181:30, 182:30, 183:31, 184:31, 185:32, 186:32, 187:33, 188:33, 189:33, 190:33,191:33, 192:33, 193:34, 194:34, 195:34, 196:34, 197:34, 198:34, 199:34, 200:34}

# 20->4
# 50->11
# 200->34

# 1234,10,5,14,7,6
# train test split 
class CUB():
    def __init__(self, root, dataset_type='train', train_ratio=1, valid_seed=123, transform=None, target_transform=None,download: bool = False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        df_img = pd.read_csv(os.path.join(root, 'images.txt'), sep=' ', header=None, names=['ID', 'Image'], index_col=0)
        df_label = pd.read_csv(os.path.join(root, 'image_class_labels.txt'), sep=' ', header=None, names=['ID', 'Label'], index_col=0)
        df_split = pd.read_csv(os.path.join(root, 'train_test_split.txt'), sep=' ', header=None, names=['ID', 'Train'], index_col=0)

        # Read image_attribute labels
        image_attributes = {}
        with open(os.path.join(root,'image_attribute_labels.txt')) as f:  # Replace "image_attributes.txt" with the actual filename
            for line in f:
                parts = line.strip().split()
                image_no = int(parts[0])
                attribute_no = int(parts[1])
                binary_value = int(parts[2])
                if image_no not in image_attributes:
                    image_attributes[image_no] = []
                image_attributes[image_no].append(binary_value)
                
        self.attri = list(image_attributes.values())
        # Convert image_attributes to DataFrame
        
        # data = [i for i in image_attributes.values()]
        # my_tensor = torch.tensor(data)

        # Create the DataFrame with 'ID' as the index and 'Attributes' as the column
        df_attributes = pd.DataFrame(image_attributes, columns=['ID', 'Attributes']).set_index('ID')

        #df_attributes = pd.DataFrame.from_dict(image_attributes, orient='index', columns=['Attributes'])
        df = pd.concat([df_img, df_label, df_split, df_attributes], axis=1)

        # relabel
        
        ###################################################################333
        self.selected_classes = [i for i in range(1,21)]
        ###################################################333

        df = df[df['Label'].isin(self.selected_classes)]
        df['Label'] = df['Label'] 
        
        # split data
        if dataset_type == 'test':
            df = df[df['Train'] == 0]
        elif dataset_type == 'train' or dataset_type == 'valid':
            df = df[df['Train'] == 1]
            # random split train, valid
            if train_ratio != 1:
                np.random.seed(valid_seed)
                indices = list(range(len(df)))
                np.random.shuffle(indices)
                split_idx = int(len(indices) * train_ratio) + 1
            elif dataset_type == 'valid':
                raise ValueError('train_ratio should be less than 1!')
            if dataset_type == 'train':
                df = df.iloc[indices[:split_idx]]
            else:       # dataset_type == 'valid'
                df = df.iloc[indices[split_idx:]]
        else:
            raise ValueError('Unsupported dataset_type!')
        
        self.img_name_list = df['Image'].tolist()
        self.label_list = df['Label'].tolist()
        self.coarse_mapping = _coarse_labels
        self.image_attributes = df['Attributes'].tolist()
        # Convert greyscale images to RGB mode
        self._convert2rgb()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, 'images', self.img_name_list[idx])
        image = Image.open(img_path)  #direct here /covert('RGB')
        target = self.label_list[idx]
        #print(target)
        #coarse_label = self.coarse_mapping.get(target, 1)

        # #for graffit
        coarse_label = self.coarse_mapping.get(target, 1)-1
        
        # Get image number from image filename
        # image_no = int(os.path.splitext(self.image_paths[idx])[0])

        # Get binary vector for image attributes
        # binary_vectors = self.image_attributes.tolist()  # Convert the column to a list of binary vectors
        # Convert the list of binary vectors into a torch tensor
        binary_tensor = torch.tensor(self.attri[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target1 = self.target_transform(target)
        #return image, coarse_label , target, binary_tensor # image -> coarse -> fine -> binary
        return image, target, target, binary_tensor # image -> fine -> fine -> binary


    # def _convert2rgb(self):
    #     for i, img_name in enumerate(self.img_name_list):
    #         img_path = os.path.join(self.root, 'images', img_name)
    #         image = Image.open(img_path)
    #         color_mode = image.mode
    #         if color_mode != 'RGB':
    #             # image = image.convert('RGB')
    #             # image.save(img_path.replace('.jpg', '_rgb.jpg'))
    #             self.img_name_list[i] = img_name.replace('.jpg', '_rgb.jpg')

    def _convert2rgb(self):
        for i, img_name in enumerate(self.img_name_list):
            orig_path = os.path.join(self.root, 'images', img_name)
            image = Image.open(orig_path)
            if image.mode != 'RGB':
                rgb = image.convert('RGB')
                new_name = img_name.replace('.jpg', '_rgb.jpg')
                rgb.save(os.path.join(self.root, 'images', new_name))
                self.img_name_list[i] = new_name