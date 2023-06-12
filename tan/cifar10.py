'''
The code is built based on https://github.com/facebookresearch/tan/blob/main/cifar10.py
'''
#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

'''
Taining of a WideResNet on CIFAR-10 with DP-SGD
'''


import argparse
from math import ceil
import torch
import torchvision
import torchvision.transforms as transforms
from src.models.wideresnet import WideResNet
import torch.nn as nn
import torch.optim as optim
from src.opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented
from opacus.validators import ModuleValidator
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Subset
from opacus import GradSampleModule
# from EMA import EMA
from src.models.EMA_without_class import create_ema, update
import pickle
from src.models.augmented_grad_samplers import AugmentationMultiplicity
from math import ceil
from src.utils.utils import (init_distributed_mode,initialize_exp,bool_flag,accuracy,get_noise_from_bs,get_epochs_from_bs,print_params,)
import json
from src.models.prepare_models import prepare_data_cifar, prepare_augmult_cifar
from torch.nn.parallel import DistributedDataParallel as DDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

import warnings
import sys
import os

warnings.simplefilter("ignore")

import random
import numpy as np
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

def train(
    model,
    ema,
    train_loader,
    optimizer,
    epoch,
    max_nb_steps,
    device,
    privacy_engine,
    K,
    logger,
    losses,
    train_acc,
    epsilons,
    grad_sample_gradients_norms_per_epoch,
    lp_norms2_per_step,
    ft_norms2_per_step,    
    test_loader,
    is_main_worker,
    args,
    norms2_before_sigma,
    nb_steps,
    missing_keys,
):
    """
    Trains the model for one epoch. If it is the last epoch, it will stop at max_nb_steps iterations.
    If the model is being shadowed for EMA, we update the model at every step.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses_epoch, train_acc_epoch, grad_sample_norms = [], [], []
    lp_grad_sample_norms, ft_grad_sample_norms = [], []    
    nb_examples_epoch = 0
    max_physical_batch_size_with_augnentation = (args.max_physical_batch_size if K == 0 else args.max_physical_batch_size // K)
    with BatchMemoryManager(data_loader=train_loader,max_physical_batch_size=max_physical_batch_size_with_augnentation,optimizer=optimizer) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            nb_examples_epoch+=len(images)
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            target = target.to(device)
            assert K == args.transform
            l = len(images)
            ##Using Augmentation multiplicity
            if K:
                images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
                target = torch.repeat_interleave(target, repeats=K, dim=0)
                transform = transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),transforms.RandomHorizontalFlip(p=0.5),])
                images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)
                assert len(images) == args.transform * l

            # compute output
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)
            losses_epoch.append(loss.item())
            train_acc_epoch.append(acc)

            loss.backward()
            is_updated = not (optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch

            ## Logging gradient statistics on the main worker
            if is_main_worker:
                per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if (g.requires_grad and g.grad_sample is not None)]
                #per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.grad_sample is not None]
                per_sample_norms = (torch.stack(per_param_norms, dim=1).norm(2, dim=1).cpu().tolist())
                grad_sample_norms += per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s

                lp_per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for (n,g) in model.named_parameters() if (g.requires_grad and g.grad_sample is not None and n in missing_keys)]
                #per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.grad_sample is not None]
                lp_per_sample_norms = (torch.stack(lp_per_param_norms, dim=1).norm(2, dim=1).cpu().tolist()) if len(lp_per_param_norms)>0 else [0]*l
                lp_grad_sample_norms += lp_per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s                

                ft_per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for (n,g) in model.named_parameters() if (g.requires_grad and g.grad_sample is not None and n not in missing_keys)]
                #per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.grad_sample is not None]
                ft_per_sample_norms = (torch.stack(ft_per_param_norms, dim=1).norm(2, dim=1).cpu().tolist()) if len(ft_per_param_norms)>0 else [0]*l
                ft_grad_sample_norms += ft_per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s  

        
            optimizer.step()
            if is_updated:
                nb_steps += 1  
                if ema:
                    update(model, ema, nb_steps)
                if is_main_worker:
                    losses.append(np.mean(losses_epoch))
                    train_acc.append(np.mean(train_acc_epoch))
                    grad_sample_gradients_norms_per_epoch.append(np.mean(grad_sample_norms))
                    lp_norms2_per_step.append(np.mean(lp_grad_sample_norms))
                    ft_norms2_per_step.append(np.mean(ft_grad_sample_norms))
                    losses_epoch, train_acc_epoch = [],[]
                    if nb_steps % args.freq_log == 0:
                        print(f"epoch:{epoch},step:{nb_steps}")
                        m2 = max(np.mean(norms2_before_sigma)-1/args.batch_size,0)
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "nb_steps":nb_steps,
                                    "train_acc": np.mean(train_acc[-args.freq_log :]),
                                    "loss": np.mean(losses[-args.freq_log :]),
                                    "grad_sample_gradients_norms": np.mean(grad_sample_norms),
                                    "grad_sample_gradients_norms_lowerC": np.mean(np.array(grad_sample_norms)<args.max_per_sample_grad_norm),
                                    #"norms2_before_sigma":list(norms2_before_sigma),
                                    #"grad_sample_gradients_norms_hist":list(np.histogram(grad_sample_norms,bins=np.arange(100), density=True)[0]),
                                }
                            )
                        )
                        norms2_before_sigma=[]
                        grad_sample_norms = []
                        lp_grad_sample_norms = []
                        ft_grad_sample_norms = []                        
                    if nb_steps % args.freq_log_val == 0:
                        test_acc_ema, train_acc_ema = (
                            test(ema, test_loader, train_loader, device)
                            if ema
                            else test(model, test_loader, train_loader, device)
                        )
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "test_acc_ema": test_acc_ema,
                                    "train_acc_ema": train_acc_ema,
                                }
                            )
                        )
                nb_examples_epoch=0

                if nb_steps >= max_nb_steps:
                    break

        epsilon = privacy_engine.get_epsilon(args.delta)
        if is_main_worker:
            epsilons.append(epsilon)
        return nb_steps, norms2_before_sigma


def test(model, test_loader, train_loader, device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = 0
    train_top1_acc = 0
    test_cnt = 0
    train_cnt = 0

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            test_top1_acc +=acc*len(images)
            test_cnt +=len(images)            
            #test_top1_acc.append(acc)

    test_top1_avg = test_top1_acc/test_cnt

    with torch.no_grad():
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            # losses.append(loss.item())
            train_top1_acc +=acc*len(images)
            train_cnt +=len(images)   
    train_top1_avg = train_top1_acc/train_cnt
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")
    return (test_top1_avg, train_top1_avg)

def epoch_test(model, test_loader, device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = []
    test_acc = 0
    total_cnt =0

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            #test_top1_acc.append(acc)
            test_acc +=acc*len(images)
            total_cnt +=len(images)
            
    #test_top1_avg = np.mean(test_top1_acc)
    test_top1_avg = test_acc/total_cnt

    return test_top1_avg

def main():  ## for non poisson, divide bs by world size

    args = parse_args()
    if args.max_nb_steps is not None:
        args.epochs = int(np.ceil(args.max_nb_steps/int((50000/args.batch_size))))
    else:
        args.max_nb_steps = int(np.ceil(50000/args.batch_size))*args.epochs

    setup_seed(args.experiment)
    init_distributed_mode(args)
    logger = initialize_exp(args)
    model = WideResNet(args.WRN_depth,10,args.WRN_k,args.nb_groups,args.init,args.order1,args.order2)
    if args.load_pretrained:
        encoder_checkpoint = os.path.join(args.pretrained_path, 'stylegan-oriented', f'wrn{args.WRN_depth}{args.WRN_k}', 'encoder.pth')
        print(f"Loading from: {encoder_checkpoint}")
        ckpt = torch.load(encoder_checkpoint, map_location='cpu')

        msg = model.load_state_dict(ckpt, strict=False)
        print("Missing keys", msg.missing_keys)

        total_params = 0
        trainable_params = 0
        for (k,v) in model.named_parameters():
            total_params += v.numel()
            if k in msg.missing_keys:
                v.requires_grad = True
                trainable_params += v.numel()
            else:
                v.requires_grad = False
        print(f"Total parameters {total_params}. Trainable parameters {trainable_params}.")
                
        
    model.cuda()
    print_params(model)
    if args.multi_gpu:
        print("using multi GPU DPDDP")
        model = DPDDP(model)
    rank = args.global_rank
    is_main_worker = rank == 0
    weights = model.module.parameters() if args.multi_gpu else model.parameters()
    # Creating the datasets and dataloader for cifar10
    train_dataset,raw_train_loader, test_loader = prepare_data_cifar(args.data_root,args.batch_size,args.proportion)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(weights, lr=args.lp_lr, momentum=args.momentum)
    print(optimizer)
    # Creating the privacy engine
    privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)

    if args.sigma is None:
        from prv_accountant.dpsgd import find_noise_multiplier
        import math

        batch_size = args.batch_size
        sample_size = 50000

        sample_rate = batch_size / sample_size
        epsilon = args.epsilon
        epochs=args.epochs

        steps = args.max_nb_steps
        delta = args.delta

        noise_multiplier = find_noise_multiplier(
            sampling_probability=sample_rate,
            num_steps=steps,
            target_epsilon=epsilon,
            target_delta=delta,
            eps_error=0.01,
            mu_max=5000)

        print("Epsilon", epsilon)
        print("Steps", steps)
        print("sampling rate", sample_rate)
        print("Noise multiplier: ", noise_multiplier)
        args.sigma = noise_multiplier

    print(args)
    sys.stdout.flush()

    ##We use our PrivacyEngine Augmented to take into accoung the eventual augmentation multiplicity
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=raw_train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=args.poisson_sampling,
        K=args.transform
    )
    print("DP optimizer test",optimizer.expected_batch_size)
    name_list = []
    for (n, p) in model.named_parameters():
        name_list.append(n)
    print(name_list)
    #print(model.keys())
    ## Changes the grad samplers to work with augmentation multiplicity
    prepare_augmult_cifar(model,args.transform)
    ema = None
    # we create a shadow model
    print("shadowing de model with EMA")
    ema = create_ema(model)
    train_acc,test_acc,epsilons,losses,top1_accs,grad_sample_gradients_norms_per_step = (0, 0, [], [], [], [])
    lp_norms2_per_step, ft_norms2_per_step = [], []
    norms2_before_sigma = []

    E = args.epochs
    if is_main_worker: print(f"E:{E},sigma:{args.sigma}, BATCH_SIZE:{args.batch_size}, noise_multiplier:{args.sigma}, EPOCHS:{E}. Total Steps:{args.max_nb_steps}.")
    nb_steps = 0
    test_accs = []
    ema_test_accs = []
    steps_per_epoch = len(train_loader)
    if is_main_worker:print(f"steps_per_epoch:{steps_per_epoch}")
    missing_keys = ['bn1.weight', 'bn1.bias', 'fc.weight', 'fc.bias']
    module_missing_keys = ["_module."+k for k in missing_keys]
    print(module_missing_keys)
    test_acc = epoch_test(model, test_loader, rank)
    ema_test_acc = epoch_test(ema, test_loader, rank)
    print(f"Initialize accuracy test {test_acc}, emal model acc {ema_test_acc}.")    
    for epoch in range(E):

        if epoch == args.ft_epoch:
            print("Change to full finetuning!")
            model2 = WideResNet(args.WRN_depth,10,args.WRN_k,args.nb_groups,args.init,args.order1,args.order2)
            model2.cuda()
            model2.load_state_dict(model._module.state_dict())
            weights = model2.module.parameters() if args.multi_gpu else model2.parameters()
            optimizer2 = optim.SGD(weights, lr=args.lr, momentum=args.momentum)
            #optimizer2.load_state_dict(optimizer.original_optimizer.state_dict())
            print(optimizer2)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model2,
                optimizer=optimizer2,
                data_loader=raw_train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                poisson_sampling=args.poisson_sampling,
                K=args.transform
            )

        nb_steps, norms2_before_sigma = train(
            model,
            ema,
            train_loader,
            optimizer,
            epoch,
            args.max_nb_steps,
            rank,
            privacy_engine,
            args.transform,
            logger,
            losses,
            top1_accs,
            epsilons,
            grad_sample_gradients_norms_per_step,
            lp_norms2_per_step,
            ft_norms2_per_step,            
            test_loader,
            is_main_worker,
            args,
            norms2_before_sigma,
            nb_steps,
            module_missing_keys
        )
        test_acc = epoch_test(model, test_loader, rank)
        ema_test_acc = epoch_test(ema, test_loader, rank)

        test_accs.append(test_acc)
        ema_test_accs.append(ema_test_acc)
        print(f"Epoch {epoch}, model acc {test_acc}, ema model acc {ema_test_acc}.")
        sys.stdout.flush()

        if is_main_worker:
            print(f"epoch:{epoch}, Current loss:{losses[-1]:.2f},nb_steps:{nb_steps}, top1_acc of model (not ema){top1_accs[-1]:.2f},average gradient norm:{grad_sample_gradients_norms_per_step[-1]:.2f}")
            print(f"average lp gradient norm:{lp_norms2_per_step[-1]:.2f}, ")
            print(f"average ft gradient norm:{ft_norms2_per_step[-1]:.2f}")
    print("Total steps:", nb_steps)
    if is_main_worker:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        test_acc, train_acc = test(ema, test_loader, train_loader, rank)
        print(f"train_acc (EMA):{train_acc:.2f}, test_acc (EMA):{test_acc:.2f}, epsilon:{epsilons[-1]:.2f}")
        logger.info("__log:"+ json.dumps({
                    "final_train_acc_ema": train_acc,
                    "final_test_acc_ema": test_acc,
                    "final_epsilon": epsilons[-1],
                    "avergage_grad_sample_gradients_norms": np.mean(
                        grad_sample_gradients_norms_per_step)
                }))
        test_acc, train_acc = test(model, test_loader, train_loader, rank)
        print(f"final test acc of non ema model:{test_acc:.2f}, final train acc of non ema model:{train_acc:.2f}")
        logger.info("__log:"+ json.dumps({
                    "final_train_acc_non_ema": train_acc,
                    "final_test_acc_non_ema": test_acc,
                }))
        print("Test acc per epoch", test_accs)
        print("Ema test acc per epoch", ema_test_accs)
        print("Best test acc", max(test_accs))
        print("Beset ema test acc", max(ema_test_accs))



def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

    parser.add_argument("--batch_size",default=4096,type=int,help="Batch size for simulated training. It will automatically set the noise $\sigma$ s.t. $B/\sigma = B_{ref}/sigma_{ref}$")
    parser.add_argument("--max_physical_batch_size",default=512,type=int,help="max_physical_batch_size for BatchMemoryManager",)
    parser.add_argument("--WRN_depth",default=16,type=int)
    parser.add_argument("--WRN_k",default=4,type=int,help="k of resnet block",)

    parser.add_argument("--lr","--learning_rate",default=0.5,type=float,metavar="LR",help="initial learning rate",dest="lr",)

    parser.add_argument("--momentum",default=0,type=float,help="SGD momentum",)
    parser.add_argument("--experiment",default=0,type=int,help="seed for initializing training. ")
    parser.add_argument("-c","--max_per_sample_grad_norm",type=float,default=1,metavar="C",help="Clip per-sample gradients to this norm (default 1.0)",)
    parser.add_argument("--delta",type=float,default=1e-5,metavar="D",help="Target delta (default: 1e-5)",)
    parser.add_argument("--epsilon",type=float,default=1, help="epsilon",)
    parser.add_argument("--sigma",type=float,default=None, help="noise multiplier",)
    parser.add_argument("--epochs",type=int,default=67, help="noise multiplier",)
    parser.add_argument("--ft_epoch",type=int,default=10, help="noise multiplier",)
    parser.add_argument("--lp_lr",type=float,default=1, help="noise multiplier",)

    parser.add_argument("--max_nb_steps",type=int,default=None, help="noise multiplier",)        

    parser.add_argument("--nb_groups",type=int,default=16,help="number of groups for the group norms",)
    parser.add_argument("--data_root",type=str,default=None,help="Where CIFAR10 is/will be stored",)
    parser.add_argument("--dump_path",type=str,default=None,help="Where results will be stored",)    
    parser.add_argument("--transform",type=int,default=0,help="using augmentation multiplicity",)

    parser.add_argument("--freq_log", type=int, default=100, help="every each freq_log steps, we log",)

    parser.add_argument("--freq_log_val",type=int,default=200,help="every each freq_log steps, we log val and ema acc",)

    parser.add_argument("--poisson_sampling",type=bool_flag,default=True,help="using Poisson sampling",)


    parser.add_argument("--proportion",default=1,type=float,help="proportion of the training set to use for training",)

    parser.add_argument("--exp_name", type=str, default="bypass")

    parser.add_argument("--init", type=int, default=0)
    parser.add_argument("--order1", type=int, default=0)
    parser.add_argument("--order2", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)
    
    parser.add_argument(
        "--load_pretrained",
        action="store_true",
        help="augmentation factor",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="augmentation factor",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()