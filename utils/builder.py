import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LambdaLR, MultiStepLR
from dataset import *
from models import *
import loss
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam


model_pool = {
    'openad_pn2': OpenAD_PN2,
    'openad_dgcnn': OpenAD_DGCNN,
}

optim_pool = {
    'sgd': SGD,
    'adam': Adam
}

init_pool = {
    'pn2_init': weights_init
}

scheduler_pool = {
    'step': StepLR,
    'cos': CosineAnnealingLR,
    'lr_lambda': LambdaLR,
    'multi_step': MultiStepLR
}


def build_model(cfg):
    if hasattr(cfg, 'model'):
        model_info = cfg.model
        weights_init = model_info.get('weights_init', None)
        model_name = model_info.type
        model_cls = model_pool[model_name]
        num_category = len(cfg.training_cfg.train_affordance)
        model = model_cls(model_info, num_category)
        if weights_init != None:
            init_fn = init_pool[weights_init]
            model.apply(init_fn)
        return model
    else:
        raise ValueError("Configuration does not have model config!")


def build_dataset(cfg):
    if hasattr(cfg, 'data'):
        data_info = cfg.data
        data_root = data_info.data_root
        afford_cat = data_info.category 
        if_partial = cfg.training_cfg.get('partial', False)
        # the training set
        train_set = AffordNetDataset(
            data_root, 'train', partial=if_partial)
        # the validation set
        val_set = AffordNetDataset(
            data_root, 'val', partial=if_partial)
        dataset_dict = dict(
            train_set=train_set,
            val_set=val_set
        )
        return dataset_dict
    else:
        raise ValueError("Configuration does not have data config!")


def build_loader(cfg, dataset_dict):
    train_set = dataset_dict["train_set"]
    val_set = dataset_dict["val_set"]
    batch_size_factor = 1
    # training loader
    train_loader = DataLoader(train_set, batch_size=cfg.training_cfg.batch_size // batch_size_factor,
                              shuffle=True, drop_last=True, num_workers=8)
    # validation loader
    val_loader = DataLoader(val_set, batch_size=cfg.training_cfg.batch_size // batch_size_factor,
                            shuffle=False, num_workers=8, drop_last=False)
    loader_dict = dict(
        train_loader=train_loader,
        val_loader=val_loader
    )
    return loader_dict


def build_loss(cfg):
    loss_fn = loss.EstimationLoss(cfg)
    return loss_fn


def build_optimizer(cfg, model):
    optim_info = cfg.optimizer
    optim_type = optim_info.type
    optim_info.pop("type")
    optim_cls = optim_pool[optim_type]
    optimizer = optim_cls(model.parameters(), **optim_info)
    scheduler_info = cfg.scheduler
    scheduler_name = scheduler_info.type
    scheduler_info.pop('type')
    scheduler_cls = scheduler_pool[scheduler_name]
    scheduler = scheduler_cls(optimizer, **scheduler_info)
    optim_dict = dict(
        scheduler=scheduler,
        optimizer=optimizer
    )
    return optim_dict
