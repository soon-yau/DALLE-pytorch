import cv2
import math
import numpy as np
import torch
from torchvision import transforms as T

def keypoints_to_image(kp, # list of x,y,confidence
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
    for kp1, kp2, color in segments:
        kp1 = kp[kp1]
        kp2 = kp[kp2]
        if kp1[-1]>=threshold and kp2[-1]>=threshold:
            cv2.line(img, get_kp(kp1), get_kp(kp2), color, 2)

    img = T.ToTensor()(img/255.).to(kp.device)
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