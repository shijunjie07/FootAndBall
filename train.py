import tqdm
import argparse
import pickle
import numpy as np
import os
import time

import torch
import torch.optim as optim

from network import footandball
from network.ssd_loss import SSDLoss
from data.soccernet_dataset import SoccerNetTrackingDataset, collate_fn
from torchvision import transforms
from torch.utils.data import DataLoader

MODEL_FOLDER = 'models'

def train_model(model, optimizer, scheduler, num_epochs, dataloaders, device, model_name):
    alpha_l_player = 0.01
    alpha_c_player = 1.
    alpha_c_ball = 5.

    total = alpha_l_player + alpha_c_player + alpha_c_ball
    alpha_l_player /= total
    alpha_c_player /= total
    alpha_c_ball /= total

    criterion = SSDLoss(neg_pos_ratio=3)

    is_validation_set = 'val' in dataloaders
    phases = ['train', 'val'] if is_validation_set else ['train']

    training_stats = {'train': [], 'val': []}

    print('Training...')
    for epoch in tqdm.tqdm(range(num_epochs)):
        for phase in phases:
            model.train() if phase == 'train' else model.eval()

            batch_stats = {'loss': [], 'loss_ball_c': [], 'loss_player_c': [], 'loss_player_l': []}

            for images, targets in dataloaders[phase]:
                images = [img.to(device) for img in images]
                images = torch.stack(images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                h, w = images.shape[-2], images.shape[-1]
                gt_maps = model.groundtruth_maps(
                    [t['boxes'].cpu().numpy() for t in targets],
                    [t['labels'].cpu().numpy() for t in targets],
                    (h, w)
                )
                gt_maps = [e.to(device) for e in gt_maps]

                with torch.set_grad_enabled(phase == 'train'):
                    predictions = model(images)
                    loss_l_player, loss_c_player, loss_c_ball = criterion(predictions, gt_maps)

                    loss = alpha_l_player * loss_l_player + alpha_c_player * loss_c_player + alpha_c_ball * loss_c_ball

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                batch_stats['loss'].append(loss.item())
                batch_stats['loss_ball_c'].append(loss_c_ball.item())
                batch_stats['loss_player_c'].append(loss_c_player.item())
                batch_stats['loss_player_l'].append(loss_l_player.item())

            avg_batch_stats = {k: np.mean(v) for k, v in batch_stats.items()}
            training_stats[phase].append(avg_batch_stats)

            print(f"{phase} Avg. loss total / ball conf. / player conf. / player loc.: "
                  f"{avg_batch_stats['loss']:.4f} / {avg_batch_stats['loss_ball_c']:.4f} / "
                  f"{avg_batch_stats['loss_player_c']:.4f} / {avg_batch_stats['loss_player_l']:.4f}")

        scheduler.step()
        print('')

    model_filepath = os.path.join(MODEL_FOLDER, model_name + '_final.pth')
    torch.save(model.state_dict(), model_filepath)

    with open(f'training_stats_{model_name}.pickle', 'wb') as handle:
        pickle.dump(training_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return training_stats

def train():
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = SoccerNetTrackingDataset(
        images_dir='F:/soccer_tracking/sports_datasets/sntracking/train/images',
        labels_dir='F:/soccer_tracking/sports_datasets/sntracking/train/labels',
        transform=transform
    )
    val_dataset = SoccerNetTrackingDataset(
        images_dir='F:/soccer_tracking/sports_datasets/sntracking/valid/images',
        labels_dir='F:/soccer_tracking/sports_datasets/sntracking/valid/labels',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    dataloaders = {'train': train_loader, 'val': val_loader}

    model = footandball.model_factory('fb1', phase='train')
    model.print_summary(show_architecture=True)
    model = model.to(device)

    model_name = 'footandball_custom_' + time.strftime("%Y%m%d_%H%M")
    print('Model name:', model_name)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[22], gamma=0.1)

    train_model(model, optimizer, scheduler, num_epochs=30, dataloaders=dataloaders, device=device, model_name=model_name)

if __name__ == '__main__':
    train()
