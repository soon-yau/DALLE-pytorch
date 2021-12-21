import cv2
import math
import numpy as np
import torch
from torchvision import transforms as T
from copy import deepcopy
import torch

def keypoints_to_image(keypoints, # list of x,y,confidence
                       threshold= 0.2, 
                       fraction=True, 
                       image_shape=(256, 256)):
    segments = [
                [0,1,(255,0,0)], 
                [1,2,(255,165,0)],
                [2,3,(218,165,32)],
                [3,4,(255,255,0)],
                [1,5, (0,255,0)],
                [5,6,(144,238,133)],
                [6,7,(144,238,133)],
                [1,8,(255,0,0)],
                [8,9,(124,252,0)],
                [9,10,(144,238,144)],
                [10,11,(135,206,235)],
                [8,12,(30,144,255)],
                [12,13,(128,0,128)],
                [13,14,(128,0,128)],
                [0, 15, (255,0,255)],
                [15, 17, (255,0,255)],
                [0, 16, (75,0,130)],
                [16, 18, (75,0,130),]
    ]
    def get_kp(kp):
        x, y = kp[0:2]
        if fraction:
            x = x * width
            y = y * height
        coords = tuple((int(x), int(y)))
        return coords
    height, width = image_shape[:2]
    img = np.zeros((height, width, 3), np.uint8)
    for person in keypoints:
        for kp1, kp2, color in segments:
            kp1 = person[kp1]
            kp2 = person[kp2]
            if kp1[-1]>=threshold and kp2[-1]>=threshold:
                cv2.line(img, get_kp(kp1), get_kp(kp2), color, 2)

    img = T.ToTensor()(img/255.).to(keypoints.device)
    return img



def keypoints_to_heatmap(keypoints,
                         threshold = 0.2, 
                         fraction=False, 
                         image_shape=(256, 256),
                         sigma=4.):
    
    height, width = image_shape[:2]
    heatmap = np.zeros((len(keypoints), height, width), np.float32)
    
    for i, kp in enumerate(keypoints):
        if kp[-1] <= threshold:
            continue
        center_x, center_y = kp[0] * height, kp[1] * width
        if fraction:
            center_x = int(center_x * width)
            center_y = int(center_y * height)

        th = 1.6052
        delta = math.sqrt(th * 2)
        x0 = int(max(0, center_x - delta * sigma))
        y0 = int(max(0, center_y - delta * sigma))

        x1 = int(min(width, center_x + delta * sigma))
        y1 = int(min(height, center_y + delta * sigma))

        # gaussian filter
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = (x - center_x) ** 2 + (y - center_y) ** 2
                exp = d / 2.0 / sigma / sigma
                if exp > th:
                    continue
                heatmap[i][y][x] = max(heatmap[i][y][x], math.exp(-exp))
                heatmap[i][y][x] = min(heatmap[i][y][x], 1.0)
    return heatmap

def heatmap_to_image(heatmaps):
    x = heatmaps.sum(axis=0)
    x/=x.max()
    return torch.unsqueeze(x, axis=0).repeat(3,1,1)

def heatmap_to_skeleton(heatmaps):

    keypoints = []
    for heatmap in heatmaps:
        coords = list(np.squeeze((heatmap==torch.max(heatmap)).nonzero().detach().cpu().numpy()))[::-1]
        if len(coords)==2:
            coords.append(1.0)
            keypoints.append(coords)
        else:
            keypoints.append([0 ,0, 0.])
    skeleton_img = keypoints_to_image(keypoints, fraction=False)

    #heatmap_img = heatmap_to_image(heatmaps)
    #mix_img = 0.3*heatmap_img + 0.7*skeleton_img
    #return mix_img
    return skeleton_img.to(heatmaps.device)

class PoseVisualizer:
    def __init__(self, pose_format):
        self.pose_format = pose_format
        if self.pose_format == 'image':
            self.fn = lambda x: x
        elif self.pose_format == 'heatmap':
            self.fn = lambda x: heatmap_to_skeleton(x[0])
        elif self.pose_format == 'keypoint':
            self.fn = lambda x : keypoints_to_image(x[0])
        else:
            raise(ValueError)

    def convert(self, x):
        return self.fn(x)


class RotateScale(object):
    def __init__(self, angle_degree=(0., 0.), scale=(1,1)):
        self.angle_degree = angle_degree
        self.scale = scale
    
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # get random degree and scale
        angle = np.random.uniform(self.angle_degree[0], self.angle_degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        # rotate image
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
        # rotate keypoint
        
        kp_ = deepcopy(keypoints)
        kp_[:,2] = 1.
        center = (0.5, 0.5)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        
        new_kp = np.dot(kp_, rotate_matrix.transpose())
        new_kp = np.concatenate((new_kp, np.expand_dims(keypoints[:,2].transpose(), 1)), axis=1)
        
        return {'image':rotated_image, 'keypoints':new_kp.astype(np.float32)}

class Crop(object):
    
    def __init__(self, margins=(0.05, 1.)):
        self.margin = margins
        
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        kps = keypoints.copy()
        height, width = image.shape[:2]
        
        left_x, top_y, right_x = np.random.uniform(self.margin[0], self.margin[1], size=3)
        right_x = 1 - right_x
        
        crop_h = crop_w = right_x - left_x
        if top_y + crop_h > 1:
            crop_h = crop_w = 1 - top_y

        right_x = left_x + crop_w
        bottom_y = top_y + crop_h

        # crop keypoints
        kps[:,0] = (kps[:,0] - left_x)/crop_w
        kps[:,1] = (kps[:,1] - top_y)/crop_h

        x_indices = np.where(np.logical_and(kps[:,0]<0, kps[:,0]>1.))[0]
        y_indices = np.where(np.logical_and(kps[:,1]<0, kps[:,1]>1.))[0]
        kps[list(set(y_indices) | set(x_indices))] = [0., 0., 0.]
        
        # crop images
        left_x = int(left_x * width)
        top_y = int(top_y * height)
        right_x = left_x + int(width*crop_w)
        bottom_y = top_y + int(height*crop_h)
        crop_image = image[top_y:bottom_y, 
                           left_x:right_x, :]
        crop_image = cv2.resize(crop_image, (width, height), interpolation=cv2.INTER_AREA)

        return {'image':crop_image, 'keypoints':kps}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image, keypoints = sample['image'], sample['keypoints']
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32)/255.
        return {'image':torch.from_numpy(image), \
                'keypoints': torch.from_numpy(keypoints)}

class ConcatSamples(object):    
    def __call__(self, sample):
        images, keypoints = sample['image'], sample['keypoints']
        kps = keypoints.copy()
        h, w, _ = images[0].shape
        left_half = images[0][:,int(0.25*h):int(0.75*h),:]
        right_half = images[1][:,int(0.25*h):int(0.75*h),:]
        combined_image = np.hstack((left_half, right_half))

        kps[0] = [[max(x-0.25, 0), y, c] for x, y, c in kps[0]]
        kps[1] = [[min(x+0.25, 1), y, c] for x, y, c in kps[1]]

        return {'image':combined_image, 'keypoints':kps}
