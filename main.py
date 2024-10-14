from idlelib.pyparse import trans

import torch
from matplotlib.font_manager import weight_dict
from timm.models.vision_transformer import vit_base_patch16_224
from torch import nn
from bokeh.models import DataRange
from click.core import batch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.cifar import CIFAR100
from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
from torchvision.transforms import transforms
from cosloss import CosineLoss
from dataset import EmbededLabelDataset
import timm
from model import Model

class recorder:
    def __init__(self, ):
        self.latest = .0
        self.avg = .0
        self.num = 0
        self.sum = .0
    def next(self, grade):
        self.latest = grade
        self.sum += grade
        self.num += 1
        self.avg = round(float(self.sum / self.num), 4)

def generate_orthogonal_vectors( m, n):
    A = torch.randn(n, m)
    Q, _ = torch.linalg.qr(A)
    return Q.T[:m]


def write_log_file(filename, precision_recorder, loss_recorder):
    with open(filename, 'a') as file:
        file.write(f'Precision: {precision_recorder.avg:.4f}, Loss: {loss_recorder.latest:.4f}\n')


def get_precision(outputs, labels):
    outputs = outputs.to('cpu')
    labels = labels.to('cpu')
    predicted = torch.argmax(outputs, 1)
    precision = torch.sum(predicted == labels) / labels.size()[0]
    return precision

def train(model, criterion, optimizer, lr_scheduler, train_loader, test_loader, epoch):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    for i in range(epoch):
        loss_recorder = recorder()
        precision_recorder = recorder()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_recorder.next(loss.item())
            precision = get_precision(outputs, targets)
            precision_recorder.next(precision)
            print('\repoch:', i+1, f'[{batch_idx+1}/{len(train_loader)}]', 'lr:', optimizer.state_dict()['param_groups'][0]['lr'], 'loss:', loss_recorder.avg, 'acc:', precision_recorder.avg, end='')
        lr_scheduler.step()
        print('\n', end='')
        write_log_file('train_log1.txt', precision_recorder, loss_recorder)
        evaluate(model, test_loader, criterion)
        torch.save(model.state_dict(), f'./checkpoints/checkpoint_{i}.pth')


def evaluate(model, test_loader, criterion):
    model.eval()
    precisionRecorder = recorder()
    loss_recorder = recorder()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_recorder.next(loss.item())
            precision = get_precision(outputs, labels)
            precisionRecorder.next(precision)
            print('\ravg precision:', precisionRecorder.avg, end='')
        print('\n')
        write_log_file('test_log1.txt', precisionRecorder, loss_recorder)


if __name__ == '__main__':
    epoch = 100
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    # model = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, pretrained_cfg_overlay=dict(file="/home/suki/pycharm-professional-2024.2.3/checkpoints/vit/base/pytorch_model.bin"))

    model.fc = nn.Linear(2048, 100)
    # model.head = nn.Linear(768, 100)
    model = Model(model, use_matrix=False)
    # model.load_state_dict(torch.load("./checkpoints/checkpoint_3.pth"))
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = CIFAR100(root="./data", train=True, transform=transform, download=False)
    test_dataset = CIFAR100(root="./data", train=False, transform=transform, download=False)
    # train_dataset = EmbededLabelDataset(root="./data", train=True, transform=transform)
    # test_dataset = EmbededLabelDataset(root='./data', train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128, pin_memory=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=128, pin_memory=True, num_workers=4, drop_last=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)

    criterion = torch.nn.CrossEntropyLoss()
    # criterion = CosineLoss()
    # y = torch.load('./y.pth', weights_only=True)
    train(model, criterion, optimizer, lr_scheduler, train_loader, test_loader, epoch)
    # evaluate(model, test_loader)
