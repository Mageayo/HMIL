
import os, shutil
import time
# from sampler import ImbalancedDatasetSampler
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
from model.two_pre import lstm_processing
from tools.get_data import Dataset
# from sample_work_test import MsCelebDataset_test
import numpy as np
from tools.args import parser
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '10,11,12,13,14'
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))



best_prec1 = 1

# best_acc = 0
best_prec2 = 1


def main():
    global args, best_prec1, best_acc, best_prec2
    args = parser.parse_args()
    train_file_path = ["./label/mooc/train.txt"]
    train_dataset = Dataset(train_file_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(train_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_file_path = ["./label/mooc/test.txt"]
    val_dataset = Dataset(val_file_path)
    # print(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        # sampler=ImbalancedDatasetSampler(val_dataset),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True)
    model = lstm_processing()
    # train_model = load_model(train_model, "/home/mjy/code/EmotiW_2019_engagement_regression-master/trained_model/lstm_mean/")
    # train_model = load_miml_model(train_model, "/home/mjy/code/EmotiW_2019_engagement_regression-master/experiment/different_feature/")
    # print(train_model.state_dict())
    # for item in train_model.state_dict():
    #     print(item)

    model = torch.nn.DataParallel(model).cuda()
    criterion3 = nn.MSELoss().cuda()
    criterion4 = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    continue_train = False
    if continue_train:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        best_prec2 = checkpoint['best_prec2']

    cudnn.benchmark = True
    print('args.evaluate', args.evaluate)
    if args.evaluate:
        validate(val_loader, model, criterion3, criterion4)  # , criterion2)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion3, criterion4, optimizer, epoch)

        # evaluate on validation set
        pred1, pred2 = validate(val_loader, model, criterion3, criterion4)

        # remember best prec@1 and save checkpoint
        regres_is_best = pred1 < best_prec1
        regres2_is_best = pred2 < best_prec2
        # acc_is_best = acc >= best_acc
        best_prec1 = min(pred1, best_prec1)
        best_prec2 = min(pred2, best_prec2)
        # best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec2': best_prec2,
            'optimizer': optimizer.state_dict(),
        }, regres_is_best, regres2_is_best)


def train(train_loader, model, criterion1, criterion2, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    regression_losses = AverageMeter()
    rank_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # aveacc = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (face_feature, head_feature, pose_feature, label, classification_label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        face_input = face_feature.float()
        pose_input = pose_feature.float()
        head_input = head_feature.float()
        target = label.float()
        target = target.cuda()
        target_class = classification_label
        target_class = target_class.cuda()
        face_input_var = torch.autograd.Variable(face_input)
        pose_input_var = torch.autograd.Variable(pose_input)
        head_input_var = torch.autograd.Variable(head_input)
        target_var = torch.autograd.Variable(target).unsqueeze(1)
        target_class_var = torch.autograd.Variable(target_class)

        # compute output
        # pdb.set_trace()
        pred_score1, pred_score2 = model(face_input_var, head_input_var, pose_input_var)
        regression_loss2 = criterion2(pred_score2, target_var)
        regression_loss1 = criterion1(pred_score1, target_var)
        #loss = 0.9 * regression_loss1 + 0.1 * regression_loss2
        loss = regression_loss1
        # pdb.set_trace()
        regression_loss = loss
        regression_losses.update(loss.item(), head_input.size(0))
        # rank_losses.update(rank_loss, input.size(0))
        losses.update(loss.item(), head_input.size(0))
        pred_score = pred_score1
        val_dif = label_dif(pred_score, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        # aveacc.update(acc, head_input.size(0))
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'regLoss {regression_loss}\t'
                  'Label_Dif {label_dif} \t'.format(
                epoch, i, len(train_loader), regression_loss=regression_loss.item(), label_dif=val_dif))


def validate(val_loader, model, criterion3, criterion4):
    batch_time = AverageMeter()
    cla_losses = AverageMeter()
    yaw_losses = AverageMeter()
    losses = AverageMeter()
    classification_losses = AverageMeter()
    regression_losses = AverageMeter()
    rank_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aveacc = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (face_feature, head_feature, pose_feature, label, classification_label) in enumerate(val_loader):
        face_input = face_feature.float()
        pose_input = pose_feature.float()
        head_input = head_feature.float()
        target = label.float()
        target = target.cuda()
        target_class = classification_label
        target_class = target_class.cuda()
        face_input_var = torch.autograd.Variable(face_input)
        pose_input_var = torch.autograd.Variable(pose_input)
        head_input_var = torch.autograd.Variable(head_input)
        ###head_pose
        target_var = torch.autograd.Variable(target).unsqueeze(1)
        target_class_var = torch.autograd.Variable(target_class)
        pred_score1, pred_score2 = model(face_input_var, head_input_var, pose_input_var)
        regression_loss1 = criterion3(pred_score1, target_var)
        regression_loss2 = criterion4(pred_score2, target_var)
        #loss = 0.9 * regression_loss1 + 0.1 * regression_loss2
        loss = regression_loss1
        # pdb.set_trace()
        regression_loss = loss
        regression_losses.update(loss.item(), head_input.size(0))
        # rank_losses.update(rank_loss, input.size(0))
        losses.update(loss.item(), head_input.size(0))
        pred_score = pred_score1
        # acc = get_accuracy(pred_score, target)
        aveacc.update(regression_loss1.item(), head_input.size(0))
        val_dif = label_dif(pred_score, target)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'regLoss {regression_loss} \t'
                  'reg2 {regression_loss2} \t'
                  'Label_Dif {label_dif} \t'
                .format(
                i, len(val_loader), regression_loss=regression_loss, regression_loss2=regression_loss2,
                label_dif=val_dif))

    print(' * ALL Prec @1 {regression_loss.avg} '
          .format(regression_loss=regression_losses))
    # print(' * Acc@1 {aveacc.avg} '
    #       .format(aveacc=aveacc))
    print(' * Prec2 @1 {aveacc.avg} '
          .format(aveacc=aveacc))

    return regression_losses.avg, aveacc.avg


def save_checkpoint(state, reg_is_best, acc_is_best, filename='checkpoint.pth.tar'):
    full_filename = os.path.join(args.model_dir, filename)
    full_bestname1 = os.path.join(args.model_dir, 'reg1_best.pth.tar')
    full_bestname2 = os.path.join(args.model_dir, 'reg2_best.pth.tar')
    torch.save(state, full_filename)
    epoch_num = state['epoch']
    if epoch_num % args.save_freq == 0 and epoch_num > 0:
        torch.save(state, full_filename.replace('checkpoint', 'checkpoint_' + str(epoch_num)))
    if reg_is_best:
        shutil.copyfile(full_filename, full_bestname1)
    if acc_is_best:
        shutil.copyfile(full_filename, full_bestname2)


def get_accuracy(output, label):
    # print("@@@@@@@@@@")

    standard_label = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
    label_np = np.array(standard_label)
    correct = 0
    for i, char in enumerate(output):

        cur_dif = label_np - char.item()
        # print(char.item())
        abs = np.maximum(cur_dif, -cur_dif)
        index = np.argmin(abs)
        cur_label = standard_label[index]
        # print(cur_label)
        if cur_label == label[i]:
            correct = correct + 1
    return correct / len(output)
    # print("@@@@@@@@@@")


def label_dif(output, label):
    dif = 0
    for i, char in enumerate(output):
        # print(char)
        # print(label[i])
        cur_dif = abs(char.item() - label[i])
        # @print(cur_dif)
        dif = dif + cur_dif
    dif = dif * 8 / len(output)
    return dif


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
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    if epoch in [int(args.epochs * 0.3), int(args.epochs * 0.5), int(args.epochs * 0.8)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # print("***********************")
    _, pred = output.topk(maxk, 1, True, True)
    # print(pred)
    pred = pred.t()
    ##print(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print(correct)
    # print("************************")
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



if __name__ == '__main__':
    main()