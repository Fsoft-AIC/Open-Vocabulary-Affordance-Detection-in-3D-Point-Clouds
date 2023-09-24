import numpy as np
import torch
import torch.nn as nn
import os
from os.path import join as opj
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from utils.eval import evaluation
from time import time


class Trainer(object):
    def __init__(self, cfg, running):
        super().__init__()
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.writer = SummaryWriter(self.work_dir)
        self.logger = running['logger']
        self.model = running["model"]
        self.dataset_dict = running["dataset_dict"]
        self.loader_dict = running["loader_dict"]
        self.train_loader = self.loader_dict.get("train_loader", None)
        self.val_loader = self.loader_dict.get("val_loader", None)
        self.loss = running["loss"]
        self.optim_dict = running["optim_dict"]
        self.optimizer = self.optim_dict.get("optimizer", None)
        self.scheduler = self.optim_dict.get("scheduler", None)
        self.epoch = 0
        self.best_val_mIoU = 0.0
        self.bn_momentum = self.cfg.training_cfg.get('bn_momentum', None)
        self.train_affordance = cfg.training_cfg.train_affordance
        self.val_affordance = cfg.training_cfg.val_affordance
        return

    def train(self):
        train_loss = 0.0
        count = 0.0
        self.model.train()
        num_batches = len(self.train_loader)
        start = time()
        self.logger.cprint("Epoch(%d) begin training........" % self.epoch)
        for data, _, label, _, _ in tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.9):
            data, label = data.float().cuda(), label.float().cuda()
            
            data = data.permute(0, 2, 1)
            label = torch.squeeze(label).long().contiguous()
            batch_size = data.size()[0]
            num_point = data.size()[2]
            self.optimizer.zero_grad()
            afford_pred = self.model(data, self.train_affordance)

            afford_pred = afford_pred.contiguous()
            loss = self.loss(afford_pred, label)

            loss.backward()
            self.optimizer.step()

            count += batch_size * num_point
            train_loss += loss.item()
        self.scheduler.step()
        if self.bn_momentum != None:
            self.model.apply(lambda x: self.bn_momentum(x, self.epoch))
        epoch_time = time() - start
        outstr = 'Train(%d), loss: %.6f, time: %d s' % (
            self.epoch, train_loss*1.0/num_batches, epoch_time//1)
        self.writer.add_scalar('Loss', train_loss*1.0/num_batches, self.epoch)
        self.logger.cprint(outstr)
        self.epoch += 1

    def val(self):
        self.logger.cprint('Epoch(%d) begin validating......' % (self.epoch-1))
        mIoU = evaluation(self.logger, self.model,
                         self.val_loader, self.val_affordance)

        if mIoU >= self.best_val_mIoU:
            self.best_val_mIoU = mIoU
            self.logger.cprint('Saving model......')
            self.logger.cprint('Best mIoU: %f' % self.best_val_mIoU)
            torch.save(self.model.state_dict(),
                   opj(self.work_dir, 'best_model.t7'))

        torch.save(self.model.state_dict(),
                      opj(self.work_dir, 'current_model.t7'))

    def run(self):
        EPOCH = self.cfg.training_cfg.epoch
        workflow = self.cfg.training_cfg.workflow
        while self.epoch < EPOCH:
            for key, running_epoch in workflow.items():
                epoch_runner = getattr(self, key)
                for e in range(running_epoch):
                    epoch_runner()
