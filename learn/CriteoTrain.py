import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import logging
import os, sys
import glob
import time
from sklearn.metrics import roc_auc_score, log_loss

current_file = os.path.abspath(__file__).replace("\\", '/')
current_path = os.path.dirname(current_file)  + "/"

import utils
from model.model import getmodel
from loader.Criteoloader import get_data


parser = argparse.ArgumentParser(description="DARTS for RS")
parser.add_argument('--debug_mode', type=int, default=0,
                    help='whether log and save the model and the output, 0 indicates yes, others indicate no')
parser.add_argument('--dataset', type=str, default='criteo',
                    help='select dataset')
parser.add_argument('--data_path', type=str, default=current_path + '../../datasets',
                    help='location of the data corpus')
parser.add_argument('--bsize', type=int, default=2, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='learning rate')
parser.add_argument('--l2', type=float, default=3e-8, 
                    help='weight decay')
parser.add_argument('--report_freq', type=int, default=1000, 
                    help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, 
                    help='num of training epochs')
parser.add_argument('--load', type=str, default='EXP-20210326-102728', help='load model arch')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--alpha_mode', type=int, default=1,
                    help='alpha_mode for Model')
parser.add_argument('--model', type=str, default='DNN_cart',
                    help='How to compute the feature interaction. \
                    Currently suppose DNN_cart, IPNN, FNN, DeepFM, FM, PIN.')
parser.add_argument('--orig_embedding_dim', type=int, default=20,
                    help='Original Feature Embedding Dims')
parser.add_argument('--comb_embedding_dim', type=int, default=10,
                    help='Combined Feature Embedding Dims')
parser.add_argument('--X', type=int, default=20, help='X')
parser.add_argument('--Y', type=int, default=20, help='Y')
parser.add_argument('--comb_field', type=int, default=325, help='num_of_selected_comb_field')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

args.load = 'logs/search-{}'.format(args.load)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.debug_mode == 0:
    args.save = 'logs/eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.8 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    # Select dataset and configurations
    if args.dataset == 'criteo':
        cont_field = 13
        cate_field = 26

        if args.X == 20:
            original_feature = 514758 + 13 + 26 + 1
        else:
            logging.info("X value error!!!")
            sys.exit(1)

        if args.Y == 20:
            combined_feature = 21037178 + 325 + 1
        else:
            logging.info("Y value error!!!")
            sys.exit(1)

        if args.model in ['DNN_cart', 'Poly2']:
            comb_field = args.comb_field
            if comb_field == 325:
                myfields = np.arange(26)
                selected_pairs = []
                for i in range(len(myfields)):
                    for j in range(i+1,len(myfields)):
                        selected_pairs.append((myfields[i],myfields[j]))

        if (args.alpha_mode in [0,1] and args.model in ['DNN_cart']) or args.model in ['Poly2']:
            use_comb = True
            dataset_folder = args.data_path + '/Criteo-new/X_' + str(args.X) + '/comb_' + \
                str(args.comb_field) + '_Y_' + str(args.Y)
        elif args.alpha_mode in [2,3] or args.model in ['LR', 'FM', 'FNN', 'IPNN', 'DeepFM', 'PIN']:
            use_comb = False
            dataset_folder = args.data_path + '/Criteo-new/X_' + str(args.X) + '/orig_39'
    else:
        logging.info("Dataset not supported!!!")
        sys.exit(1)

    # Config devices and loss
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        cudnn.benchmark = True
        cudnn.enabled = True
    else:
        device = torch.device('cpu')
    logging.info('device = %s', device)
    logging.info('args = %s', args)

    criterion = nn.BCELoss()
    criterion = criterion.to(device)

    # Select Model
    hidden_dims=[700,700,700,700,700]
    assert args.model in ['DNN_cart', 'LR', 'FM', 'Poly2', 'FNN', 'IPNN', 'DeepFM', 'PIN']
    assert args.alpha_mode >= 0 and args.alpha_mode <= 3
    if args.model == 'DNN_cart':
        logging.info('alpha_mode = %s', args.alpha_mode)
        if args.alpha_mode in [0]:
            cont_arch = utils.load_arch(os.path.join(args.load, 'arch_weight.npy'))
            dess_arch = np.zeros_like(cont_arch)
            dess_arch[np.arange(len(cont_arch)), cont_arch.argmax(1)] = 1
            logging.info('Searched arch = ' + ' '.join(str(num) for num in np.sum(dess_arch, axis=0)))
        else:
            dess_arch = None
        model = getmodel(args.model, cont_field, cate_field, 
                original_feature, combined_feature, 
                comb_field=comb_field, hidden_dims=hidden_dims, 
                selected_pairs=selected_pairs, arch=dess_arch,
                device=device, alpha_mode=args.alpha_mode,
                orig_embedding_dim=args.orig_embedding_dim,
                comb_embedding_dim=args.comb_embedding_dim)
    elif args.model in ['LR', 'FM', 'FNN', 'IPNN', 'DeepFM', 'PIN']:
        model = getmodel(args.model, cont_field, cate_field, 
                original_feature, combined_feature, 
                hidden_dims=hidden_dims,
                device=device, alpha_mode=args.alpha_mode,
                orig_embedding_dim=args.orig_embedding_dim)
    elif args.model in ['Poly2']:
        model = getmodel(args.model, cont_field, cate_field, 
                original_feature, combined_feature, 
                comb_field=comb_field, device=device)
    else:
        logging.info("Model not implemented!!!")
        sys.exit(1)
    model = model.to(device)
    logging.info("param size = %f GB", utils.count_parameters_in_GB(model))

    # Select Optimizer
    if args.model == 'DNN_cart':
        regularized_param = list()
        if hasattr(model, 'comb_embeddings_table'):
            regularized_param += list(model.comb_embeddings_table.parameters())
        if hasattr(model, 'addition_embeddings_table'):
            regularized_param += list(model.addition_embeddings_table.parameters())
        regularized_optim_config = {'params': regularized_param, 'lr': args.lr, 'weight_decay': args.l2}
        unregularized_param = list(model.cate_embeddings_table.parameters()) + \
                            list(model.fc_layers.parameters()) + \
                            list(model.output_layer.parameters())
        unregularized_optim_config = {'params': unregularized_param, 'lr': args.lr}
        optimizer = optim.Adam([regularized_optim_config, unregularized_optim_config])
    elif args.model in ['LR', 'FM', 'Poly2', 'FNN', 'IPNN', 'DeepFM', 'PIN']:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lr = args.lr
    logging.info("model = %s", model)
    test_losses = []
    test_aucs = []
    last_auc = 0.
    last_log_loss = 0.
    for epoch in range(args.epochs):
        starttime = time.time()
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_loss = train(model, criterion, optimizer, device, dataset_folder, use_comb, logging)
        logging.info('train_loss %f', train_loss)
        train_time = time.time()
        train_period = train_time - starttime

        # validation
        test_auc, test_log_loss = infer(model, criterion, device, dataset_folder, use_comb, logging)
        logging.info('test_auc %f', test_auc)
        logging.info('test_log_loss %f', test_log_loss)
        test_aucs.append(test_auc)
        test_losses.append(test_log_loss)
        test_period = time.time() - train_time

        if test_auc > last_auc:
            last_auc = test_auc
            last_log_loss = test_log_loss
            if args.debug_mode == 0:
                state = {
                    'epoch': epoch,
                    'args': args,
                    'best_auc': last_auc,
                    'best_log_loss': last_log_loss,
                    'optimizer_state': optimizer.state_dict(),
                    'model_state': model.state_dict()
                }
                torch.save(state, os.path.join(args.save, 'weight.pt'))
        else: 
            # Early stop
            logging.info("Early stopped!")
            break
        endtime=time.time()
        logging.info('Epoch time: %f s', endtime-starttime)
        logging.info('Train time: %f s', train_period)
        logging.info('Test time: %f s', test_period)

    logging.info('best_test_auc %f', last_auc)
    logging.info('best_log_loss %f', last_log_loss)

def infer(model, criterion, device, dataset_folder, use_comb, logging):
    losses = utils.AvgrageMeter()
    model.eval()
    pred_list = []
    label_list = []
    step = 0

    for x, y in get_data(dataset_folder, name='test', bsize=args.bsize, \
            use_comb=use_comb, comb_field=args.comb_field):
        step += 1
        batch_size = args.bsize * 1000
        if len(y) != args.bsize:
            continue
        target = torch.reshape(y.to(device), shape=(batch_size, -1))

        features = []
        for feature_name in ['feat_conts', 'feat_cates', 'feat_combs']:
            if feature_name in x.keys():
                features.append(torch.reshape(x[feature_name].to(device), shape=(batch_size, -1)))
            else:
                my_dtype = torch.float32 if feature_name == 'feat_conts' else torch.int64
                features.append(torch.tensor((), dtype=my_dtype, device=device).new_ones((batch_size, 0)))

        logits = model(features[0], features[1], features[2])
        loss = criterion(logits, target)

        pred_list.append(logits.cpu().data.numpy())
        label_list.append(target.cpu().data.numpy())
        
        n = features[0].size(0)
        losses.update(loss.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e', step, losses.avg)
        
    y_pred_list = np.concatenate(pred_list).astype("float64")
    y_true_list = np.concatenate(label_list).astype("float64")
    moving_roc = roc_auc_score(y_true_list, y_pred_list)
    moving_log_loss = log_loss(y_true_list, y_pred_list)
    return moving_roc, moving_log_loss


def train(model, criterion, optimizer, device, dataset_folder, use_comb, logging):
    losses = utils.AvgrageMeter()
    model.train()
    step = 0

    starttime = time.time()
    for x, y in get_data(dataset_folder, name='train', bsize=args.bsize, \
            use_comb=use_comb, comb_field=args.comb_field):
        step += 1
        batch_size = args.bsize * 1000
        target = torch.reshape(y.to(device), shape=(batch_size, -1))

        features = []
        for feature_name in ['feat_conts', 'feat_cates', 'feat_combs']:
            if feature_name in x.keys():
                features.append(torch.reshape(x[feature_name].to(device), shape=(batch_size, -1)))
            else:
                my_dtype = torch.float32 if feature_name == 'feat_conts' else torch.int64
                features.append(torch.tensor((), dtype=my_dtype, device=device).new_ones((batch_size, 0)))

        optimizer.zero_grad()
        logits = model(features[0], features[1], features[2])
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()

        n = features[0].size(0)
        losses.update(loss.item(), n)

        if step % args.report_freq == 0:
            endtime = time.time()
            logging.info('train %03d %e', step, losses.avg)
            logging.info('Time %f', endtime-starttime)
            starttime = time.time()

    return losses.avg

if __name__ == '__main__':
    main() 

