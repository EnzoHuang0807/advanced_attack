import os
import torch
import argparse
from torch.utils.data import DataLoader

from src.attack_spgd import uap_spgd
from src.attacks_sga import uap_sga
from src.utils import load_model, save_universal_image
from src.dataset import UniversalDataset


model_name = "resnet20_cifar100"

def get_parser():
    parser = argparse.ArgumentParser(description='Generating adversarial examples')
    parser.add_argument('--input_dir', default='./data/images',
                        help='training set directory')
    parser.add_argument('--batch_size', type=int, help='batch size', default=200)
    parser.add_argument('--minibatch', type=int, help='inner batch size for SGA', default=10)
    parser.add_argument('--alpha', type=float, default=12, help='maximum perturbation value (L-infinity) norm')
    parser.add_argument('--beta', type=float, default=9, help='clamping value')
    parser.add_argument('--step_decay', type=float, default=0.1, help='step size')
    parser.add_argument('--epoch', type=int, default=20, help='epoch num')
    parser.add_argument('--spgd', type=int,default=1, help='loss type')
    parser.add_argument('--iter', type=int,default=4, help='inner iteration num')
    parser.add_argument('--Momentum', type=int, default=0, help='Momentum item')
    parser.add_argument('--cross_loss', type=int, default=0, help='loss type')
    parser.add_argument('--GPU_ID', default='1', type=str)
    return parser.parse_args()


def main():

    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(0)
    
    # load data 
    dataset = UniversalDataset(args.input_dir)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    loader_eval = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model(model_name, device, targeted=False)

    # attack
    nb_epoch = args.epoch
    eps = args.alpha / 255
    beta = args.beta
    step_decay = args.step_decay
    batch_size = args.batch_size
    minibatch = args.minibatch
    loss_function = args.cross_loss
    Momentum = args.Momentum
    iter = args.iter

    if args.spgd:
        uap, _ = uap_spgd(model, loader, nb_epoch, eps, beta, step_decay, 
                          loss_function, batch_size, loader_eval, Momentum)
    else:
        uap, _ = uap_sga(model, loader, nb_epoch, eps, beta, step_decay, 
                         loss_function, batch_size, minibatch, loader_eval, 
                         iter, Momentum)

    print(uap * 255 + 12)
    save_universal_image(uap, "universal.png")

if __name__ == '__main__':
    main()