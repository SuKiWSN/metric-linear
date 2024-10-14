from idlelib.pyparse import trans

from torchvision.datasets import CIFAR100
from torchvision.transforms import transforms
import torch
from PIL import Image

class EmbededLabelDataset(CIFAR100):
    def __init__(self, root, train, transform):
        super(EmbededLabelDataset, self).__init__(root=root, train=train, transform=transform, download=False)
        self.EmbededLabel = torch.load('y.pth', weights_only=True)


    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        embededlabel = self.EmbededLabel[label-1]
        return img, embededlabel, label


def generate_orthogonal_vectors( m, n):
    A = torch.randn(n, m)
    Q, _ = torch.linalg.qr(A)
    return Q.T[:m]

if __name__ == '__main__':
    y = generate_orthogonal_vectors(100, 2048)
    torch.save(y, 'y.pth')
    # transform = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     # Add more transformations as needed...
    # ])
    # dataset = EmbededLabelDataset('./data', True, transform, )
    # print(dataset[0])

