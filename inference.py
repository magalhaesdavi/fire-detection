import torch.nn as nn
from torchvision import transforms
from models import coatnet
import torchvision
from utils import *
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes_names = ["fire", "non-fire"]
num_classes = 2
batch_size = 16

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

ff_model = coatnet.coatnet_4(one_fc=True, l1=16, l2=8)
ff_model.load_state_dict(torch.load('weights/ffcoatnet_4.ckpt'))
ff_model = ff_model.to(device)
ff_model.eval()

input_path = 'datasets/fullframe_test/'
test_set = torchvision.datasets.ImageFolder(root=input_path, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

confusion_matrix = torch.zeros(num_classes, num_classes)


def inference(test_loader, model, criterion, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1],  #, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            softmax = nn.Softmax(dim=1)
            output = softmax(output)
            _, predicted = torch.max(output.data, 1)

            # p_nonfire < 0.7, 0.67, 0.68, 0.69, 0.71, 0.72, 0.73, 0.74 -> mean = 0.81 (ffcoatnet472)
            for idx, p in enumerate(_):
                flag = False  # 69
                if predicted[idx] == 1 and p < 0.9:
                    predicted[idx] = 0
                #     flag = True
                # if predicted[idx] == 0 and p < 0.6 and not flag:
                #     predicted[idx] = 1

            # loss = criterion(output, target)
            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            # losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % print_freq == 0:
            progress.display(i)

            predicted = predicted.cpu().numpy()
            target = target.cpu().numpy()
            fn = len(np.where(predicted - target == 1)[0])  # 10
            fp = len(np.where(predicted - target == -1)[0])  # 01
            tp = len(np.where(predicted + target == 0)[0])  # 00
            tn = len(np.where(predicted + target == 2)[0])  # 11
            confusion_matrix[0, 0] += tp
            confusion_matrix[0, 1] += fn
            confusion_matrix[1, 0] += fp
            confusion_matrix[1, 1] += tn

        # TODO: this should also be done with the ProgressMeter
        # print(top1.avg)
        print(' * Acc@1 {acc:.3f}'
              .format(acc=top1.avg))

    return top1.avg, losses.avg


criterion = nn.CrossEntropyLoss()
inference(test_loader=test_loader, model=ff_model, criterion=criterion, device=device, print_freq=8)

confusion_matrix = confusion_matrix.numpy()
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
print(confusion_matrix)
print((confusion_matrix[0, 0] + confusion_matrix[1, 1]) / 2)
# plot_confusion_matrix(confusion_matrix, classes_names, 'images/ffcoatnet40_0.6_0.6_cm.png')
