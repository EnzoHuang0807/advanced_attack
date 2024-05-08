import os
from tqdm import tqdm
import torch
import argparse
import torchattacks
from torch.utils.data import DataLoader

from utils import load_model
from dataset import UniversalDataset


model_name = "resnet20_cifar100"

def get_parser():
    parser = argparse.ArgumentParser(description='Generating adversarial examples')
    parser.add_argument('--batch_size', default=100, type=int, help='the bacth size')
    parser.add_argument('--input_dir', default='../data/images', type=str, help='the path for custom benign images')
    parser.add_argument('--GPU_ID', default='1', type=str)
    return parser.parse_args()


def main():

    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load data and model
    test_dataset = UniversalDataset(args.input_dir, eval=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model(model_name, device, targeted=False)

    # evaluation  
    correct = 0
    total = 0
    model.eval()

    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        _, prediction = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (prediction != labels).sum().item()

    print('Attack Success Rate on the test images: {} %'.format(100 * correct / total))

if __name__ == '__main__':
    main()