import argparse

parser = argparse.ArgumentParser(description='Training for learning engagement assessment')

#path
parser.add_argument('--feature_dir', metavar='DIR',
                    default='/data/MM/data/raw_feature/',
                    help='path to dataset')


#model
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=201, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')  ##16
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,  ##0.0001
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

#resume
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume',
                    default='',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', '-s', default=10, type=int,
                    metavar='N', help='save frequency (default: 10)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate train_model on validation set')

parser.add_argument('--model_dir', '-m',
                    default='/data/mjy/code/engage220308/save_model/three_feature_0407/', type=str)

parser.add_argument('--end2end', default=True,
                    help='if true, using end2end with dream block, else, using naive architecture')