from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np

import argparse

def eval(model, seq_length, n_cpu, video_path, disp):
    dataset = GolfDB(
                     video_path=video_path,
                     vid_dir='C:/Users/vital/Downloads/train_dataset_Синтез/positions_of_the_golf_swing_train/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        if images == []:
            continue
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cpu())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        preds = np.zeros(8)
        for k in range(8):
            preds[k] = np.argsort(probs[:, k])[-1]
        preds = preds.astype(int)
        if disp:
            print(preds)
        return preds

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if not args.path:
        print("Не введен путь до видео!!!")
        exit(1)
    else:
        print(args.path)

    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    save_dict = torch.load('models/swingnet_1800.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict(save_dict['model_state_dict'])
    model.cpu()
    model.eval()
    result = eval(model, seq_length, n_cpu, args.path,False)
    print(*result)

