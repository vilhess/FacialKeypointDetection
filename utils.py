import torch
import matplotlib.pyplot as plt


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))

def save_example(img, gt, pred, path):
    fig = plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.scatter(gt[0], gt[1], label='gt')
    plt.scatter(pred[0][0].item(), pred[0][1].item(), label = 'pred')
    plt.legend()
    fig.savefig(path)