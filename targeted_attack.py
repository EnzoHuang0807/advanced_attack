import os
from tqdm import tqdm
import torch
import argparse
import torchattacks
from torch.utils.data import DataLoader

from src.utils import load_model, save_images
from src.dataset import TargetedDataset


model_names = ["resnet110_cifar100", 
               "preresnet164bn_cifar100", 
               "seresnet110_cifar100", 
               "densenet40_k36_bc_cifar100", 
               "diaresnet164bn_cifar100"
]


def get_parser():
    parser = argparse.ArgumentParser(description='Generating adversarial examples')
    parser.add_argument('--batch_size', default=100, type=int, help='the bacth size')
    parser.add_argument('--method', default='PGD', type=str, help='the attack method')
    parser.add_argument('--eps', default=4/255, type=float, help='the step size to update the perturbation')
    parser.add_argument('--alpha', default=1/255, type=float, help='the step size to update the perturbation')
    parser.add_argument('--steps', default=10, type=int, help='the number of perturbation steps')
    parser.add_argument('--input_dir', default='./data/images', type=str, help='the path for custom benign images')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--GPU_ID', default='1', type=str)
    return parser.parse_args()


def main():

    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load data 
    test_dataset = TargetedDataset(args.input_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    # ensemble models
    model = load_model(model_names, device)

    # attack
    if args.method == "PGD":
        attack = torchattacks.PGD(model, eps=args.eps, alpha=args.alpha, steps=args.steps, random_start=True)
    attack.set_mode_targeted_by_label()

    # evaluation  
    correct = 0
    total = 0
    model.eval()

    for images, labels, path in tqdm(test_loader):

        images = images.to(device)
        labels = torch.zeros_like(labels).to(device)
        adv_images = attack(images, labels)
        
        save_images(args.output_dir, adv_images, path)

        with torch.no_grad():
            outputs = model(adv_images)

        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    print('Attack Success Rate on the test images: {} %'.format(100 * correct / total))
    with open('targeted_eval.txt', 'a') as f:
            f.write(f"| {args.method} | {int(args.alpha * 255)} | {args.steps} | {100 * correct / total} |\n")


    


if __name__ == '__main__':
    main()