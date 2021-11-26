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
from model.model_search import DNN_cart_search
from loader.Avazuloader import get_data


parser = argparse.ArgumentParser(description="DARTS for RS")
parser.add_argument('--debug_mode', type=int, default=0,
                    help='whether log and save the model and the output, 0 indicates yes, others indicate no')
parser.add_argument('--dataset', type=str, default='avazu',
                    help='select dataset')
parser.add_argument('--data_path', type=str, default=current_path + '../../datasets',
                    help='location of the data corpus')
parser.add_argument('--bsize', type=int, default=2, help='batch size')
parser.add_argument('--lr', type=float, default=1e-3, 
                    help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, 
                    help='weight decay')
parser.add_argument('--report_freq', type=float, default=1000, 
                    help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, 
                    help='num of training epochs')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--search_freq', type=int, default=1, 
                    help='frequency of searching')
parser.add_argument('--unrolled', action='store_true', default=False, 
                    help='use one-step unrolled validation loss')
parser.add_argument('--arch_lr', type=float, default=1e-3, 
                    help='learning rate for arch encoding')
parser.add_argument('--orig_embedding_dim', type=int, default=40,
                    help='Original Feature Embedding Dims')
parser.add_argument('--comb_embedding_dim', type=int, default=4,
                    help='Original Feature Embedding Dims')                    
parser.add_argument('--X', type=int, default=5, help='X')
parser.add_argument('--Y', type=int, default=1, help='Y')
parser.add_argument('--comb_field', type=int, default=105, help='num_of_selected_comb_field')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.debug_mode == 0:
    args.save = 'logs/search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

def main():
    if args.dataset == 'avazu':
        cont_field = 0
        cate_field = 24

        if args.X == 5:
            original_feature = 1228770 + 24 + 1
        else:
            logging.info("X value error!!!")
            sys.exit(1)

        if args.Y == 1:
            combined_feature = 240329204 + 276 + 1
        else:
            logging.info("Y value error!!!")
            sys.exit(1)

        comb_field = args.comb_field
        if comb_field == 276:
            myfields = np.arange(24) 
            selected_pairs = []
            for i in range(len(myfields)):
                for j in range(i+1,len(myfields)):
                    selected_pairs.append((myfields[i],myfields[j]))

        dataset_folder = args.data_path + '/Avazu-new/X_' + str(args.X) + '/comb_' + \
            str(args.comb_field) + '_Y_' + str(args.Y)
    else:
        logging.info("Dataset not supported!!!")
        sys.exit(1)

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

    hidden_dims=[500,500,500,500,500]
    model = DNN_cart_search(cont_field, cate_field, comb_field, 
            original_feature, combined_feature, criterion, 
            selected_pairs, device=device, hidden_dims=hidden_dims,
            orig_embedding_dim = args.orig_embedding_dim, 
            comb_embedding_dim = args.comb_embedding_dim)
    model = model.to(device)
    logging.info("param size = %f GB", utils.count_parameters_in_GB(model))

    regularized_param = list(model.comb_embeddings_table.parameters())
    if hasattr(model, 'addition_embedding_table'):
        regularized_param += list(model.addition_embedding_table.parameters())
    regularized_optim_config = {
        'params': regularized_param,
        'lr': args.lr,
        'weight_decay': args.weight_decay
    }
    unregularized_param = list(model.orig_embeddings_table.parameters()) + \
                        list(model.fc_layers.parameters()) + \
                        list(model.output_layer.parameters())
    unregularized_optim_config = {
        'params': unregularized_param,
        'lr': args.lr
    }
    optimizer = optim.Adam([unregularized_optim_config, regularized_optim_config])

    arch_optim_config = {
        'params': model.arch_parameters,
        'lr': args.arch_lr
    }
    arch_optimizer = optim.Adam([arch_optim_config])

    lr = args.lr
    logging.info("model = %s", model)
    test_losses = []
    test_aucs = []
    last_auc = 0.
    last_log_loss = 0.
    for epoch in range(args.epochs):
        starttime = time.time()
        logging.info('epoch %d lr %e', epoch, lr)
        my_arch = model.arch_parameters.detach().cpu().numpy()
        current_arch = np.zeros_like(my_arch)
        current_arch[np.arange(len(my_arch)), my_arch.argmax(1)] = 1
        logging.info('arch = %s', np.sum(current_arch, axis=0))

        # training
        train_loss = train(model, criterion, optimizer, arch_optimizer, device, dataset_folder, logging)
        logging.info('train_loss %f', train_loss)
        train_time = time.time()
        train_period = train_time - starttime

        # validation
        test_auc, test_log_loss = infer(model, criterion, device, dataset_folder, logging)
        logging.info('test_auc %f', test_auc)
        logging.info('test_log_loss %f', test_log_loss)
        test_aucs.append(test_auc)
        test_losses.append(test_log_loss)
        test_period = time.time() - train_time

        if test_auc > last_auc:
            last_auc = test_auc
            if args.debug_mode == 0:
                np.save(os.path.join(args.save, 'arch_weight.npy'), model.get_arch_parameters().numpy())
                torch.save(model.state_dict(), os.path.join(args.save, 'model_weights.pth'))
        else:
            # Early stop
            logging.info("Early stopped!")
            break
            
        endtime = time.time()
        logging.info('Epoch time: %f s', endtime-starttime)
        logging.info('Train time: %f s', train_period)
        logging.info('Test time: %f s', test_period)

    logging.info('best_test_auc %f', last_auc)
    logging.info('best_log_loss %f', last_log_loss)


def infer(model, criterion, device, dataset_folder, logging):
    losses = utils.AvgrageMeter()
    model.eval()
    pred_list = []
    label_list = []
    step = 0

    for x, y in get_data(dataset_folder, name='test', bsize=args.bsize, \
            use_comb=True, comb_field=args.comb_field):
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

def train(model, criterion, optimizer, arch_optimizer, device, dataset_folder, logging):
    losses = utils.AvgrageMeter()
    step = 0

    starttime = time.time()
    for x, y in get_data(dataset_folder, name='train', bsize=args.bsize, \
            use_comb=True, comb_field=args.comb_field):
        model.train()
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

        arch_optimizer.zero_grad()
        optimizer.zero_grad()
        logits = model(features[0], features[1], features[2])
        loss = criterion(logits, target)

        loss.backward()
        arch_optimizer.step()
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

