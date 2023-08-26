import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class GolfDB(Dataset):
    def __init__(self, video_path, vid_dir, seq_length, transform=None, train=True):
        self.video_path = video_path
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images, labels = [], []
        cap = cv2.VideoCapture(self.video_path)

        i = 0
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ret, img = cap.read()
            if not ret:
                continue
            dim = (200, 200)
            img = cv2.resize(img, dim)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            i += 1
        cap.release()

        sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}
    





       

