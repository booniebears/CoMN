import os
import time
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models.vgg import *
from models.mobilenet import *
from models.resnet import *
from models.stgcn import STGCN
from models.LeNet5 import Lenet5
from utils import *
from frozen_dir import app_path
from Network import *

from IRdrop import Eva_WMC

Trainfactors = trainParam(Dir="Parameters")


parser = argparse.ArgumentParser(description="PyTorch Training")

parser.add_argument(
    "--epochs",
    default=Trainfactors["epoch"],
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=Trainfactors["learning_rate"],
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=5e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 5e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=50,
    type=int,
    metavar="N",
    help="print frequency (default: 20)",
)
parser.add_argument(
    "--resume",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    default=Trainfactors["test_mode"],
    dest="evaluate",
    action="store_true",
    help="evaluate model on vali dation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--half", dest="half", action="store_true", help="use half-precision(16-bit) "
)
parser.add_argument("--cpu", dest="cpu", action="store_true", help="use cpu")
parser.add_argument(
    "--save-dir",
    dest="save_dir",
    help="The directory used to save the trained models",
    default=os.path.join(app_path(), "vgg11_cifar10_weight"),
    type=str,
)
# parser.add_argument('--user_weight', dest='save_dir',
#                     help='The directory used to save the trained weight for user',
#                     default=os.path.join(app_path(),'User_Weight'), type=str)


best_prec1 = 0

def train_epoch(training_input, training_target, batch_size, net, optimizer,
                A_wave, loss_criterion):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(A_wave, X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
    return sum(epoch_training_losses)/len(epoch_training_losses)

def accuracy_optimizer(user_name, weight_name, tid):
    Trainfactors = trainParam(Dir="Parameters")

    if Trainfactors["usermodel"] == True:
        net = Net(Dir="Parameters")
        path = "/userdefined_cifar10_weight"
    if Trainfactors["defaultmodel"] == True:
        if Trainfactors["vgg11"]:
            net = vgg11(Dir="Parameters")
            path = "/vgg11_cifar10_weight"
        elif Trainfactors["vgg13"]:
            net = vgg13(Dir="Parameters")
            path = "/vgg13_cifar10_weight"
        elif Trainfactors["vgg16"]:
            net = vgg16(Dir="Parameters")
            path = "/vgg16_cifar10_weight"
        elif Trainfactors["vgg19"]:
            net = vgg19(Dir="Parameters")
            path = "/vgg19_cifar10_weight"
        elif Trainfactors["resnet18"]:
            net = resnet18(Dir="Parameters")
            path = "/resnet18_cifar10_weight"
        elif Trainfactors["resnet34"]:
            net = resnet34(Dir="Parameters")
            path = "/resnet34_cifar10_weight"
        elif Trainfactors["resnet50"]:
            net = resnet50(Dir="Parameters")
            path = "/resnet50_cifar10_weight"
        elif Trainfactors["resnet101"]:
            net = resnet101(Dir="Parameters")
            path = "/resnet101_cifar10_weight"
        elif Trainfactors["resnet152"]:
            net = resnet152(Dir="Parameters")
            path = "/resnet152_cifar10_weight"
        elif Trainfactors["mobilenet"]:
            net = mobilenet(Dir="Parameters")
            path = "/mobilenet_cifar10_weight"
        elif Trainfactors["lenet"]:
            net = Lenet5(Dir="Parameters")
            path = "/lenet_mnist_weight"
        elif Trainfactors["stgcn"]:
            A, X, means, stds = load_metr_la_data() # A.shape = (207, 207)
            A_wave = get_normalized_adj(A)
            A_wave = torch.from_numpy(A_wave)
            # use past 12 data info to predict the future 3 data info
            num_timesteps_input = 12
            num_timesteps_output = 3
            split_line1 = int(X.shape[2] * 0.8)
            train_original_data = X[:, :, :split_line1]
            training_input, training_target = generate_dataset(train_original_data,
                                                        num_timesteps_input=num_timesteps_input,
                                                        num_timesteps_output=num_timesteps_output)
            net = STGCN(A_wave.shape[0], training_input.shape[3], num_timesteps_input,
                        num_timesteps_output, Dir="Parameters")
            path = "/stgcn_METRLA_weight"
    
    trainInfo = path[1:]
    trainInfo = trainInfo.split("_")
    
    accuracy_file = app_path() + "/generate_data/" + user_name + "/accuracy_out/accuracy_out" + str(tid) + ".txt"
    model = net
    path = "Weight/" + user_name + weight_name + path

    global args, best_prec1
    args = parser.parse_args()
    args.save_dir = os.path.join(app_path(), path, "checkpoint.tar")
    args.resume = os.path.join(app_path(), path, "checkpoint.tar")
    args.epochs = Trainfactors["epoch"]
    args.lr = Trainfactors["learning_rate"]
    args.evaluate = Trainfactors["test_mode"]
    args.resume_training = Trainfactors["resume"]
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(os.path.join(app_path(), path)):
        os.makedirs(os.path.join(app_path(), path))

    if args.evaluate:
        # No weights available
        if not os.path.isfile(args.resume):
            with open(accuracy_file, "w+", encoding="utf-8") as f:
                f.write("No weights available for inference. Please train the weights first.\n")
            return 0
                
    with open(accuracy_file, "w+", encoding="utf-8") as f:
        f.write(
            "====================================================================================================================\n"
        )
        f.write(
            "#############################Training neural network models to adapt to hardware nonidealities##########################\n"
        )
        f.write(
            "====================================================================================================================\n"
        )
        f.write(f"Training {trainInfo[0]} on {trainInfo[1]}!!!\n")

    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    if Trainfactors["stgcn"]:
        with open(accuracy_file, "a+", encoding="utf-8") as f:
            f.write("Training of STGCN on METR-LA will take quite a while...")
            f.write("\n")
        smallest_MAE = 10000000
        split_line1 = int(X.shape[2] * 0.8)
        train_original_data = X[:, :, :split_line1]
        val_original_data = X[:, :, split_line1:]

        val_input, val_target = generate_dataset(val_original_data,
                                                num_timesteps_input=num_timesteps_input,
                                                num_timesteps_output=num_timesteps_output)
        A_wave = get_normalized_adj(A)
        A_wave = torch.from_numpy(A_wave)
        A_wave = A_wave.to(device=args.device)
        
        if args.resume and args.resume_training == True:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                if args.evaluate and Trainfactors["irdrop"] == True: # When conducting evaluation, do irdrop first
                    checkpoint = Eva_WMC(checkpoint,tid,user_name)
                args.start_epoch = checkpoint["epoch"]
                smallest_MAE = checkpoint["smallest_MAE"]
                model.load_state_dict(checkpoint["state_dict"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.evaluate, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_criterion = nn.MSELoss()

        training_losses = []
        validation_losses = []
        validation_maes = []
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            torch.save(
                [args.start_epoch+args.epochs, epoch], os.path.join(app_path(), "Parameters/epoch.json")
            )
            # print(f"Epoch = {epoch}")
            loss = train_epoch(training_input, training_target, args.batch_size,
                               model, optimizer, A_wave, loss_criterion)
            training_losses.append(loss)
            # print('Start validation!!!')
            with torch.no_grad():
                net.eval()
                net = net.to(device=args.device)
                val_input = val_input.to(device=args.device)
                val_target = val_target.to(device=args.device)
                
                batch_size_val = 128
                num_val_samples = val_input.shape[0]
                num_val_batches = (num_val_samples + batch_size_val - 1) // batch_size_val
                
                val_loss = 0.0
                validation_mae = 0.0
                
                for i in range(num_val_batches):
                    start_idx = i * batch_size_val
                    end_idx = min((i + 1) * batch_size_val, num_val_samples)
                    
                    val_input_batch = val_input[start_idx:end_idx]
                    val_target_batch = val_target[start_idx:end_idx]
                    # print(f'val_input_batch.shape = {val_input_batch.shape}') # torch.Size([128, 207, 12, 2])
                    out = net(A_wave, val_input_batch)
                    val_loss += loss_criterion(out, val_target_batch).item()
                    
                    out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
                    target_unnormalized = val_target_batch.detach().cpu().numpy() * stds[0] + means[0]
                    mae = np.mean(np.absolute(out_unnormalized - target_unnormalized))
                    validation_mae += mae
                
                val_loss /= num_val_batches
                validation_mae /= num_val_batches
                
                validation_losses.append(val_loss)
                validation_maes.append(validation_mae)
                
                out = None
                val_input = val_input.to(device='cpu')
                val_target = val_target.to(device='cpu')
            with open(accuracy_file, "a+", encoding="utf-8") as f:
                f.write(f"Epoch = {epoch+1}, Training loss: {training_losses[-1]}, Validation loss: {validation_losses[-1]}, Validation MAE: {validation_maes[-1]}")
                f.write("\n")
            print(f"Epoch = {epoch+1}, Training loss: {training_losses[-1]}, Validation loss: {validation_losses[-1]}, Validation MAE: {validation_maes[-1]}",flush=True)
            
            if validation_maes[-1] < smallest_MAE:
                print("Smallest MAE found, save model!!!",flush=True)
                smallest_MAE = validation_maes[-1]
                torch.save({
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "smallest_MAE": smallest_MAE,
                }, args.save_dir)
        return smallest_MAE

    else:
        if args.resume and args.resume_training == True:
            if os.path.isfile(args.resume):
                print("First Loading!!")
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                if args.evaluate and Trainfactors["irdrop"] == True: # When conducting evaluation, do irdrop first
                    checkpoint = Eva_WMC(checkpoint,tid,user_name)
                args.start_epoch = checkpoint["epoch"]
                best_prec1 = checkpoint["best_prec1"]
                model.load_state_dict(checkpoint["state_dict"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        args.evaluate, checkpoint["epoch"]
                    )
                )
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))


        cudnn.benchmark = True

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        if Trainfactors["lenet"]: # MNIST dataset
            train_dataset = torchvision.datasets.MNIST(root=os.path.join(app_path(), "datasets"), 
                                                    train=True,
                                                    download=True,
                                                    transform=transforms.Compose([
                                                            transforms.Resize((32, 32)),
                                                            transforms.ToTensor()]))
            test_dataset = torchvision.datasets.MNIST(root=os.path.join(app_path(), "datasets"), 
                                                    train=False,
                                                    download=True,
                                                    transform=transforms.Compose([
                                                            transforms.Resize((32, 32)),
                                                            transforms.ToTensor()]))
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root=os.path.join(app_path(), "datasets"),
                    train=True,
                    transform=transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            # normalize,
                        ]
                    ),
                    download=True,
                ),
                batch_size=args.batch_size,
                shuffle=True,
                pin_memory=True,
            )

            val_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(
                    root=os.path.join(app_path(), "datasets"),
                    train=False,
                    transform=transforms.Compose(
                        [
                            transforms.ToTensor(),
                            # normalize,
                        ]
                    ),
                    download=True,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=True,
            )

        criterion = nn.CrossEntropyLoss()
        if args.cpu:
            criterion = criterion.cpu()
        else:
            criterion = criterion.cuda()

        if args.half:
            model.half()
            criterion.half()

        optimizer = torch.optim.Adam(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay
        )

        if args.evaluate:
            validate(val_loader, model, criterion, accuracy_file)
            return

        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            torch.save(
                [args.start_epoch+args.epochs, epoch], os.path.join(app_path(), "Parameters/epoch.json")
            )
            # with open(os.path.join(app_path(), "Parameters/epoch.txt"),"w") as f:
            #     f.write(str(args.epochs) + "\n")
            #     f.write(str(epoch) + "\n")
            adjust_learning_rate(optimizer, epoch)

            train(train_loader, model, criterion, optimizer, epoch, accuracy_file)

            prec1 = validate(val_loader, model, criterion, accuracy_file)

            is_best = prec1 > best_prec1

            best_prec1 = max(prec1, best_prec1)

            # save weight to the only User_Weight dir
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                },
                is_best,
                filename=args.save_dir,
            )
        # Conduct irdrop only at the end of training.
        if os.path.isfile(args.save_dir):
            print("=> loading checkpoint '{}'".format(args.save_dir))
            if Trainfactors["irdrop"] == True:
                checkpoint = torch.load(args.save_dir)
                checkpoint = Eva_WMC(checkpoint,tid,user_name)
                model.load_state_dict(checkpoint["state_dict"])
                prec1 = validate(val_loader, model, criterion, accuracy_file)
        else:
            print("=> no checkpoint found at '{}'".format(args.save_dir))

        return best_prec1


def train(train_loader, model, criterion, optimizer, epoch, accuracy_file):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # input = input * 3
        # torch.save(input, "./Images/image.pth")
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input = input.cuda()  # async=True
            target = target.cuda()  # async=True
        if args.half:
            input = input.half()

        output = model(input)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                )
            )
            with open(
                accuracy_file,
                "a+",
                encoding="utf-8",
            ) as f:
                f.write(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top1=top1,
                    )
                )
                f.write("\n")


def validate(val_loader, model, criterion, accuracy_file):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # input = input * 3
        if args.cpu == False:
            input = input.cuda()  # async=True
            target = target.cuda()  # async=True

        if args.half:
            input = input.half()

        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                )
            )
            with open(
                accuracy_file,
                "a+",
                encoding="utf-8",
            ) as f:
                f.write(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )
                f.write("\n")

    print(" * Prec@1 {top1.avg:.3f}".format(top1=top1))
    with open(
        accuracy_file, "a+", encoding="utf-8"
    ) as f:
        f.write(" * Prec@1 {top1.avg:.3f}".format(top1=top1))
        f.write("\n")

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    if is_best == True:
        torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main(user_name, weight_name, tid):
    updateParam("OptParam", "training_finish", False, Dir="Parameters")
    updateParam("OptParam", "evaluate_mode", False, Dir="Parameters")

    Trainfactors = trainParam(Dir="Parameters")
    Macrofactors = MacroParam(Dir="Parameters")
    Specfactors = SpecParam(Dir="Parameters")
    
    accuracy_file = app_path() + "/generate_data/" + user_name + "/accuracy_out/accuracy_out" + str(tid) + ".txt"
    if Trainfactors["quantization"] == True:
        if (
            Macrofactors["Weight_precision"] <= 1
            or Macrofactors["ADC_resolution"] <= 1
            or Macrofactors["DAC_resolution"] <= 1
        ):
            with open(
                accuracy_file,
                "w+",
                encoding="utf-8",
            ) as f:
                f.write("The precision of weight/input/output is not enough")
                f.write("\n")
            exit()

    if Trainfactors["variation"] == True:
        if Macrofactors["NVM_states"] <= 1 or Macrofactors["NVM_states"] > 8:
            with open(
                accuracy_file,
                "a+",
                encoding="utf-8",
            ) as f:
                f.write("The conductance states per NVM is too large")
                f.write("\n")
            exit()
    if Trainfactors["epoch"] < 50:
        with open(
            accuracy_file,
            "a+",
            encoding="utf-8",
        ) as f:
            f.write(
                "The training epoch is not enough, more than 50 epoches are recommended"
            )
            f.write("\n")
    if Trainfactors["learning_rate"] > 0.1:
        with open(
            accuracy_file,
            "a+",
            encoding="utf-8",
        ) as f:
            f.write("The learing rate is too large, 0.002 learing rate is recommended")
            f.write("\n")
        exit()
    if Trainfactors["learning_rate"] < 0.0001:
        with open(
            accuracy_file,
            "a+",
            encoding="utf-8",
        ) as f:
            f.write("The learing rate is too samll, 0.002 learing rate is recommended")
            f.write("\n")
        exit()
    if Specfactors["Subarray"][0] > 1024 or Specfactors["Subarray"][1] > 1024:
        with open(
            accuracy_file,
            "a+",
            encoding="utf-8",
        ) as f:
            f.write(
                "The subarray size is too large, 128*128 subarray size is recommended"
            )
            f.write("\n")
        exit()
    if Specfactors["Subarray"][0] < 64 or Specfactors["Subarray"][1] < 64:
        with open(
            accuracy_file,
            "a+",
            encoding="utf-8",
        ) as f:
            f.write(
                "The subarray size is too small, 128*128 subarray size is recommended"
            )
            f.write("\n")
        exit()

    best_prec1 = accuracy_optimizer(user_name, weight_name, tid)
    with open(accuracy_file, "a+", encoding="utf-8") as f:
        f.write(f"Best: {best_prec1}")
        f.write("\n")
    print("Best: ", best_prec1)

    updateParam("OptParam", "training_finish", True, Dir="Parameters")


if __name__ == "__main__":
    main("tcad", "", "1")
