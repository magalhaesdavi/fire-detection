from models import coatnet
from utils import *


def validate(val_loader, model, criterion, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],  #, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        # print(top1.avg)
        print(' * Acc@1 {acc:.3f}'
              .format(acc=top1.avg))

    return top1.avg, losses.avg


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],  # top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        targets = targets.to(device)

        # compute outputs
        output = model(images)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        acc1 = accuracy(output, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        # print(acc1[0].item())
        # print(type(acc1[0].item()))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = ('fire', 'non-fire')

# Hyper-parameters
# input_size = 50*50*3
num_classes = 2
batch_size = 16

# transform = transforms.Compose([
#         transforms.Resize((50, 50)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.6558, 0.4875, 0.2858), (0.3469, 0.3010, 0.2526))
#         ])

transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        # transforms.Resize((331, 331)),
        transforms.Resize((224, 224)),
        # transforms.RandomRotation(degrees=(0, 45)),
        # transforms.RandomAdjustSharpness(sharpness_factor=3),
        # transforms.RandomAutocontrast(p=0.5),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize((0.6558, 0.4875, 0.2858), (0.3469, 0.3010, 0.2526))
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

# data_path = '../dataset/'

# data_set = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
train_set = torchvision.datasets.ImageFolder(root='dataset/smallframe_dataset/train',
                                             transform=transform)
val_set = torchvision.datasets.ImageFolder(root='dataset/smallframe_dataset/val',
                                           transform=transform)

# train_set = torchvision.datasets.ImageFolder(root='../split_dataset/train', transform=transform)
# val_set = torchvision.datasets.ImageFolder(root='../split_dataset/val', transform=transform)
# test_set = torchvision.datasets.ImageFolder(root='../split_dataset/test', transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                         batch_size=batch_size,
                                         shuffle=False)

# test_loader = torch.utils.data.DataLoader(dataset=test_set,
#                                           batch_size=batch_size,
#                                           shuffle=False)


def save_checkpoint(state, model_name, is_best=0, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    # if is_best:
    #     shutil.copyfile(filename, 'model_best.pth.tar')
    torch.save(state, model_name + '.ckpt')


if __name__ == '__main__':
    # splitfolders.ratio('../dataset', output='split-dataset', seed=1337, ratio=(.8, 0.1, 0.1))
    # model = model.resnet1(num_classes=num_classes, device=device)
    # model = model.resnet1(num_classes=num_classes, device=device)
    # model_name = 'resnet1'
    # model = nasnetalarge.get_model(nb_classes=2)
    # model_name = 'nasnet'
    model = coatnet.coatnet_0()
    model.to(device)
    model_name = 'coatnet'
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_acc = 0
    # arch = 'coatnet'
    train_acc = []
    train_losses = []
    val_acc = []
    val_losses = []

    for epoch in range(0, 100):
        adjust_learning_rate(optimizer, epoch, lr=0.0001)
        # train for one epoch
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, device, print_freq=8)
        train_acc.append(acc)
        train_losses.append(loss)
        # evaluate on validation set
        acc, loss = validate(val_loader, model, criterion, device, print_freq=8)
        val_acc.append(acc)
        val_losses.append(loss)
        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

    save_checkpoint(state=model.state_dict(), model_name=model_name)

    plt.figure(figsize=(10, 5))
    plt.title("Training and validation loss")
    plt.plot(val_losses, label="val")
    plt.plot(train_losses, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('images/loss_' + model_name + '.png')

    plt.figure(figsize=(10, 5))
    plt.title("Training and validation loss")
    plt.plot(val_acc, label="val")
    plt.plot(train_acc, label="train")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig('images/acc_' + model_name + '.png')
