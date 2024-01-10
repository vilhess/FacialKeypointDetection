import torch
import matplotlib.pyplot as plt


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path))

def save_example(img, gt, pred, path):
    fig = plt.figure()
    plt.imshow(img.permute(1, 2, 0))
    plt.scatter(gt[0], gt[1], label='gt', c='r')
    plt.scatter(gt[2], gt[3], c='r')
    plt.scatter(gt[4], gt[5], c='r')
    plt.scatter(pred[0][0].item(), pred[0][1].item(), label = 'pred', c='b')
    plt.scatter(pred[0][2].item(), pred[0][3].item(), c='b')
    plt.scatter(pred[0][4].item(), pred[0][5].item(), c='b')
    plt.legend()
    fig.savefig(path)
    plt.close()