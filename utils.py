import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# from tqdm import tqdm

# # Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# writer = SummaryWriter(log_dir='../runs')
#
# classes = ('Fire', 'Not fire')
#
# # Hyper-parameters
# # input_size = 50*50*3
# num_classes = 2
# batch_size = 16
#
# transform = transforms.Compose([
#         transforms.Resize((50, 50)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.6558, 0.4875, 0.2858), (0.3469, 0.3010, 0.2526))
#         ])
#
# data_path = '../dataset/'
#
# # data_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
# train_set = torchvision.datasets.ImageFolder(root='../split_dataset/train', transform=transform)
# val_set = torchvision.datasets.ImageFolder(root='../split_dataset/val', transform=transform)
# test_set = torchvision.datasets.ImageFolder(root='../split_dataset/test', transform=transform)
#
# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_set,
#                                            batch_size=batch_size,
#                                            shuffle=True)
#
# val_loader = torch.utils.data.DataLoader(dataset=val_set,
#                                          batch_size=batch_size,
#                                          shuffle=False)
#
# dataloaders = {'train': train_loader, 'val': val_loader}
# dataset_sizes = {'train': len(train_set), 'val': len(val_set)}


def plot_confusion_matrix(m, classes_names, image_name):
    df_cm = pd.DataFrame(m, index=[i for i in classes_names],
                         columns=[i for i in classes_names])
    plt.figure(figsize=(10, 7))
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues")
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction")
    plt.title("Normalized confusion matrix")
    fig = heatmap.get_figure()
    # fig.savefig('images/coatnet_4_and_v0_cm.png')
    fig.savefig(image_name)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


# helper functions
def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds_tensor = preds_tensor.to(torch.device('cpu'))
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    images = images.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # print(type(val))
        # print(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # print(fmtstr)
        # print(type(fmtstr))
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# train for one epoch
# def validate(val_loader, model, criterion, device, print_freq):
#     batch_time = AverageMeter('Time', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     # top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(val_loader),
#         [batch_time, losses, top1],  #, top5],
#         prefix='Test: ')
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         end = time.time()
#         for i, (images, target) in enumerate(val_loader):
#             images = images.to(device)
#             target = target.to(device)
#
#             # compute output
#             output = model(images)
#             loss = criterion(output, target)
#
#             # measure accuracy and record loss
#             acc1 = accuracy(output, target)
#             losses.update(loss.item(), images.size(0))
#             top1.update(acc1[0].item(), images.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#             if i % print_freq == 0:
#                 progress.display(i)
#
#         # TODO: this should also be done with the ProgressMeter
#         # print(top1.avg)
#         print(' * Acc@1 {acc:.3f}'
#               .format(acc=top1.avg))
#
#     return top1.avg, losses.avg
#
#
# def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
#     batch_time = AverageMeter('Time', ':6.3f')
#     data_time = AverageMeter('Data', ':6.3f')
#     losses = AverageMeter('Loss', ':.4e')
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     # top5 = AverageMeter('Acc@5', ':6.2f')
#     progress = ProgressMeter(
#         len(train_loader),
#         [batch_time, data_time, losses, top1],  # top5],
#         prefix="Epoch: [{}]".format(epoch))
#
#     # switch to train mode
#     model.train()
#
#     end = time.time()
#     for i, (images, targets) in enumerate(train_loader):
#         # measure data loading time
#         data_time.update(time.time() - end)
#
#         images = images.to(device)
#         targets = targets.to(device)
#
#         # compute outputs
#         output = model(images)
#         loss = criterion(output, targets)
#
#         # measure accuracy and record loss
#         acc1 = accuracy(output, targets)
#         losses.update(loss.item(), images.size(0))
#         top1.update(acc1[0].item(), images.size(0))
#         # print(acc1[0].item())
#         # print(type(acc1[0].item()))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#         if i % print_freq == 0:
#             progress.display(i)
#
#     return top1.avg, losses.avg


# object detection
# def train2(train_loader, model, criterion, optimizer, epoch, device, print_freq):
#     # batch_time = AverageMeter('Time', ':6.3f')
#     # data_time = AverageMeter('Data', ':6.3f')
#     # losses = AverageMeter('Loss', ':.4e')
#     # top1 = AverageMeter('Acc@1', ':6.2f')
#     # # top5 = AverageMeter('Acc@5', ':6.2f')
#     # progress = ProgressMeter(
#     #     len(train_loader),
#     #     [batch_time, data_time, losses, top1],  # top5],
#     #     prefix="Epoch: [{}]".format(epoch))
#
#     # switch to train mode
#     model.train()
#
#     # end = time.time()
#     # for i, (images, target) in enumerate(train_loader):
#     # loop = tqdm(train_loader, leave=True)
#     for i, (images, targets) in enumerate(train_loader):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         # measure data loading time
#         # data_time.update(time.time() - end)
#
#         # images = images.to(device)
#         # target = target.to(device)
#
#         # compute outputs
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         loss_value = losses.item()
#
#         # measure accuracy and record loss
#         # acc1 = accuracy(output, targets)
#         # losses.update(loss.item(), images.size(0))
#         # top1.update(acc1[0].item(), images.size(0))
#         # print(acc1[0].item())
#         # print(type(acc1[0].item()))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         losses.backward()
#         optimizer.step()
#
#         # measure elapsed time
#         # batch_time.update(time.time() - end)
#         # end = time.time()
#
#         # if i % print_freq == 0:
#         #     progress.display(i)
#
#     return  # top1.avg, losses.avg
#
#
# def validate2(val_loader, model, criterion, device, print_freq):
#     # batch_time = AverageMeter('Time', ':6.3f')
#     # losses = AverageMeter('Loss', ':.4e')
#     # top1 = AverageMeter('Acc@1', ':6.2f')
#     # # top5 = AverageMeter('Acc@5', ':6.2f')
#     # progress = ProgressMeter(
#     #     len(val_loader),
#     #     [batch_time, losses, top1],  #, top5],
#     #     prefix='Test: ')
#
#     # switch to evaluate mode
#     model.eval()
#     cpu_device = torch.device("cpu")
#     with torch.no_grad():
#         # end = time.time()
#         for i, (images, targets) in enumerate(val_loader):
#             images = list(image.to(device) for image in images)
#
#             # compute output
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             outputs = model(images)
#
#             outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#             res = {target["image_id"].item(): output for
#                    target, output in zip(targets, outputs)}
#
#             # measure accuracy and record loss
#             # acc1 = accuracy(output, target)
#             # losses.update(loss.item(), images.size(0))
#             # top1.update(acc1[0].item(), images.size(0))
#
#             # measure elapsed time
#             # batch_time.update(time.time() - end)
#             # end = time.time()
#
#             # if i % print_freq == 0:
#             #     progress.display(i)
#
#         # TODO: this should also be done with the ProgressMeter
#         # print(top1.avg)
#         # print(' * Acc@1 {acc:.3f}'
#         #       .format(acc=top1.avg))
#
#     return  # top1.avg, losses.avg
