import os

import timm
import torch
import opacus
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from opacus.accountants.utils import get_noise_multiplier
import time
import random
import numpy as np
from opacus.utils.batch_memory_manager import wrap_data_loader
from opacus.utils.batch_memory_manager import BatchMemoryManager
import torchvision.models as models
from torchvision.datasets import CIFAR10, CIFAR100

#from args import args
import argparse
import sys
sys.path.append("./../../")
from wrn import WideResNet

start = time.time()
# utils
def get_features_wrn(f, img_loader):
    features = []
    for (img, target) in tqdm(img_loader):
        with torch.no_grad():
            img = img.cuda()
            features.append(f(img).detach().cpu())
    return torch.cat(features)    

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    #for data, target in train_loader:
    if not args.disable_dp:
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=args.max_phy_batch_size, optimizer=optimizer) as memory_safe_data_loader:
            for i, (data, target) in enumerate(memory_safe_data_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
    else:
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())        

    if not args.disable_dp:
        try:
            epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} (ε = {epsilon:.2f}, δ = {args.delta})")
        except:
            print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += data.shape[0]*criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(test_loss,correct,len(test_loader.dataset), 
                                                                                 100.0 * correct / len(test_loader.dataset)))
    return 100.0 * correct / len(test_loader.dataset)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training Linear Probing")
    parser.add_argument("--dataset", type=str, default="CIFAR10", help="dataset name")
    parser.add_argument("--arch", type=str, default="wrn", help="feature extractor")
    parser.add_argument("--lr", type=float, default=0.5, help="learing rate")
    parser.add_argument("--epochs", type=int, default=500, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=50000, help="batch size")
    parser.add_argument("--epsilon", type=float, default=0.2, help="privacy budget")
    parser.add_argument("--sigma", type=float, default=365.5, help="privacy budget")    
    parser.add_argument("--max_per_sample_grad_norm", type=float, default=1, help="max grad sample norm")
    parser.add_argument("--delta", type=float, default=1e-5, help="delta value")
    parser.add_argument("--secure_rng", type=bool, default=False)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11297)
    parser.add_argument("--disable-dp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--random_dataset", type=str, default="stylegan-oriented", help="feature extractor")
    parser.add_argument("--data_root", type=str, )
    parser.add_argument("--ckpt_root", type=str, )
    parser.add_argument("--feat_root", type=str, )

    args = parser.parse_args()

    if args.dataset == "CIFAR10":
        args.num_classes = 10
        args.max_phy_batch_size = 25000
        normalize = transforms.Compose([transforms.ToTensor()])
    else:
        args.num_classes = 100
        args.max_phy_batch_size = 2500
        normalize = transforms.Compose([transforms.ToTensor()])
    
    setup_seed(args.seed)

    train_ds = getattr(datasets, args.dataset)(args.data_root, transform=normalize, train=True, download=True)
    labels = torch.tensor(train_ds.targets)
    test_ds = getattr(datasets, args.dataset)(args.data_root, transform=normalize, train=False, download=True)
    labels_test = torch.tensor(test_ds.targets)

    extracted_path = os.path.join(args.feat_root, args.arch, str(args.dataset), str(args.random_dataset))

    if not os.path.exists(os.path.join(extracted_path, "features_train.npy")):
        if not os.path.exists(extracted_path):
            os.makedirs(extracted_path, exist_ok=True)
        extractor = WideResNet(16, 4, 16, 0, 0, 0)
        msg = extractor.load_state_dict(torch.load(os.path.join(args.ckpt_root, args.random_dataset, "wrn", "encoder.pth"), map_location="cpu"), strict=False)
        print("Missing keys", msg.missing_keys)
        extractor.eval()
        model = extractor.to(args.device)

        train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        
        features = get_features_wrn(extractor, train_dataloader)
        features_test = get_features_wrn(extractor, test_dataloader)
        features = features.detach().cpu().numpy()
        features_test = features_test.detach().cpu().numpy()
        np.save(os.path.join(extracted_path, "features_train.npy"), features)
        np.save(os.path.join(extracted_path, "features_test.npy"), features_test)
    else:
        features = np.load(os.path.join(extracted_path, "features_train.npy"))
        features_test = np.load(os.path.join(extracted_path, "features_test.npy"))

    features = torch.from_numpy(features).type(torch.FloatTensor)
    features_test = torch.from_numpy(features_test).type(torch.FloatTensor)
    print(features.shape, features_test.shape)

    train_loader = DataLoader(TensorDataset(features, labels), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(features_test, labels_test), batch_size=args.batch_size, shuffle=False)

    sample_rate = 1 / len(train_loader)

    print(args)
    print(len(train_loader))
    run_results = []
    best_acc = 0

    test_accs = []
    #print("Learning rate", lr, "noise sigma", args.sigma)
    classifier = nn.Linear(features.shape[-1], args.num_classes, bias=False).cuda()
    classifier.weight.data.zero_()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9)
    privacy_engine = None
    
    if not args.disable_dp and args.sigma is not None:
        privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
        model, optimizer, train_loader = privacy_engine.make_private(
            module=classifier,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier = args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
        )
    else:
        model = classifier
    test(model, args.device, test_loader)

    for epoch in range(1, args.epochs + 1):
        train(args, model, args.device, train_loader, optimizer, privacy_engine, epoch)
        test_acc = test(model, args.device, test_loader)
        best_acc = max(best_acc, test_acc)
        test_accs.append(test_acc)

    end = time.time()
    print(test_accs)
    print("Best acc: ", best_acc)
    print("Total time", end -start)

if __name__ == "__main__":
    main()
