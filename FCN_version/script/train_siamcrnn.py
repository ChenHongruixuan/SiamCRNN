import sys
sys.path.append('')
import argparse
import os
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import util_func.lovasz_loss as L

from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.make_data_loader import OSCDDatset3Bands, make_data_loader, OSCDDatset13Bands
from util_func.metrics import Evaluator
from deep_networks.SiamCRNN import SiamCRNN



class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.train_data_loader = make_data_loader(args)
        print(args.model_type + ' is running')
        self.evaluator = Evaluator(num_class=2)

        self.deep_model = SiamCRNN(in_dim_1=3, in_dim_2=3)
        self.deep_model = self.deep_model.cuda()

        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        self.deep_model.train()
        class_weight = torch.FloatTensor([1, 10]).cuda()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            self.optim.zero_grad()

            pre_img, post_img, bcd_labels, _ = data

            pre_img = pre_img.cuda().float()
            post_img = post_img.cuda().float()
            bcd_labels = bcd_labels.cuda().long()
            # input_data = torch.cat([pre_img, post_img], dim=1)

            # bcd_output = self.deep_model(input_data)
            bcd_output = self.deep_model(pre_img, post_img)

            bcd_loss = F.cross_entropy(bcd_output, bcd_labels, weight=class_weight, ignore_index=255)
            lovasz_loss = L.lovasz_softmax(F.softmax(bcd_output, dim=1), bcd_labels, ignore=255)

            main_loss = bcd_loss + 0.75 * lovasz_loss
            main_loss.backward()

            self.optim.step()

            if (itera + 1) % 10 == 0:
                print(
                    f'iter is {itera + 1},  change detection loss is {bcd_loss}'
                )
                if (itera + 1) % 200 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation()
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))

                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    self.deep_model.train()

        print('The accuracy of the best round is ', best_round)

    def validation(self):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset_path = ''
        with open('', "r") as f:
            # data_name_list = f.read()
            data_name_list = [data_name.strip() for data_name in f]
        data_name_list = data_name_list
        dataset = OSCDDatset3Bands(dataset_path=dataset_path, data_list=data_name_list, crop_size=512,
                                   max_iters=None, type='test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=8, drop_last=False)
        torch.cuda.empty_cache()

        for itera, data in enumerate(val_data_loader):
            pre_img, post_img, bcd_labels, data_name = data

            pre_img = pre_img.cuda().float()
            post_img = post_img.cuda().float()
            bcd_labels = bcd_labels.cuda().long()
            # input_data = torch.cat([pre_img, post_img], dim=1)

            # bcd_output = self.deep_model(input_data)
            bcd_output = self.deep_model(pre_img, post_img)
            bcd_output = bcd_output.data.cpu().numpy()
            bcd_output = np.argmax(bcd_output, axis=1)

            bcd_img = bcd_output[0].copy()
            bcd_img[bcd_img == 1] = 255

            # imageio.imwrite('./' + data_name[0] + '.png', bcd_img)

            bcd_labels = bcd_labels.cpu().numpy()
            self.evaluator.add_batch(bcd_labels, bcd_output)

        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


def main():
    parser = argparse.ArgumentParser(description="Training on OEM_OSM dataset")
    parser.add_argument('--dataset', type=str, default='OSCD_3Bands')
    parser.add_argument('--dataset_path', type=str,
                        default='')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_data_list_path', type=str,
                        default='')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--max_iters', type=int, default=100000)
    parser.add_argument('--model_type', type=str, default='SiamCRNN')
    parser.add_argument('--model_param_path', type=str, default='/content/saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.data_name_list = data_name_list

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
