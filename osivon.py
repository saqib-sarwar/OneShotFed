import numpy as np
import random
import copy
import argparse
import csv
import torch
from torchvision import datasets, transforms
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import get_dataset
from models import get_model
from train_model import LocalUpdate
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img, test_img_uncertainty, test_img_global_posterior_avg
from run_one_shot_algs import state_dict_to_vector

import optuna

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--algs_to_run', nargs = '+', type=str, required=True)
parser.add_argument('--seed', type=int, required=False, default = 0)
parser.add_argument('--alpha', type = float, required = False, default = 0.1)
parser.add_argument('--num_clients', type = int, required = False, default = 5)
parser.add_argument('--num_rounds', type = int, required = False, default = 1)
parser.add_argument('--local_epochs', type=int, required= False, default = 30)
parser.add_argument('--use_pretrained', type=bool, required = False, default = False) 

args_parser = parser.parse_args()

seed = args_parser.seed
dataset = args_parser.dataset
model_name = args_parser.model
algs_to_run = args_parser.algs_to_run
local_epochs = args_parser.local_epochs
use_pretrained = args_parser.use_pretrained
alpha = args_parser.alpha
num_clients = args_parser.num_clients
num_rounds = args_parser.num_rounds
print_every_test = 1
print_every_train = 1



filename = "ivon_one_shot_"+str(seed)+"_"+dataset+"_"+model_name+"_"+"_"+str(local_epochs)
filename_csv = filename + ".csv"


if(dataset=='CIFAR100'):
  n_c = 100
elif (dataset == 'GTSRB'):
  n_c = 43
else: n_c = 10




dict_results = {}


for alg in algs_to_run:
    print ("Running algorithm", alg)
    print ("Using pre-trained model:", use_pretrained)
    
    # Default arguments
    args = {
        # "bs": 32,
        "local_epochs": local_epochs,
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "rounds": num_rounds,
        "num_clients": num_clients,
        "augmentation": False,
        "eta": 0.1,
        "dataset": dataset,
        "model": model_name,
        "use_pretrained": use_pretrained,
        "n_c": n_c,
        # "ess": 15000,
        "weight_decay": 1e-4,
        "hess_init": 1,
        "mc_train": 1,
        "test_samples": 10
    }

    np.random.seed(3)
    dataset_train, dataset_train_global, dataset_test_global, net_cls_counts = get_dataset(dataset, num_clients, n_c, alpha, False)
    test_loader = DataLoader(dataset_test_global, batch_size=len(dataset_test_global))

    ind = np.random.choice(len(dataset_train_global), 500)
    dataset_val = torch.utils.data.Subset(dataset_train_global, ind)

# ### Default parameters
# args={
# "bs":64,
# "local_epochs":local_epochs,
# "device":'cuda',
# "rounds":num_rounds, 
# "num_clients": num_clients,
# "augmentation": False,
# "eta": 0.2,
# "dataset":dataset,
# "model":model_name,
# "use_pretrained":use_pretrained,
# "n_c":n_c,
# "ess": 20000,  ### For IVON#
# "weight_decay": 0.0001,
# "hess_init": 1,
# "mc_train": 1
# }


    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    net_glob_org = get_model(args['model'], n_c, bias = False, use_pretrained = use_pretrained).to(args['device'])  # n_c is the number of classes


    n = len(dataset_train)
    print ("No. of clients", n)

    ### Computing weights of the local models proportional to datasize
    p = np.zeros((n))
    for i in range(n):
        p[i] = len(dataset_train[i])
    p = p/np.sum(p)


    local_model_accs = []
    local_model_loss = []
    d = parameters_to_vector(net_glob_org.parameters()).numel()   ### Total number of parameters in the model
    net_glob = copy.deepcopy(net_glob_org)
    initial_vector = parameters_to_vector(net_glob.parameters())
    

    for t in range(0,args['rounds']):

        if(dataset=='CIFAR10' or dataset=='CIFAR100' or dataset == 'CINIC10' or dataset == 'GTSRB'):
            args['augmentation'] = True

        if(use_pretrained == True):
            args['eta'] = 0.001     ### Smaller learning rate if using pretrained model
            
        ind = [i for i in range(n)]

        model_vectors = []
        models = []
        hessians = []
        hessian_vectors = []
        F_kfac_list = []
        F_diag_list = []
        
        for i in ind:

            print ("Training Local Model ", i)
            
            if(len(dataset_train[i])>5000):
                args['bs'] = 64
            else:
                args['bs'] = 32
            
            net_glob.train()
            local = LocalUpdate(args=args, dataset=dataset_train[i])
            model_vector, model, hessian, hessian_vector, optimizer =local.train_ivon(copy.deepcopy(net_glob), i, args['n_c'])
            model_vectors.append(model_vector)
            models.append(model)
            hessian_vectors.append(hessian_vector)
            hessians.append(hessian)
            # loc_test_acc, loc_test_loss = test_img(model, dataset_test_global, args)
            loc_test_acc, loc_test_loss = test_img_uncertainty(model, dataset_test_global, optimizer, args)
            print ("IVON Local Model ", i, "Test Acc. ", loc_test_acc, "Test Loss ", loc_test_loss)
            local_model_accs.append(loc_test_acc.flatten()[0])
            local_model_loss.append(loc_test_loss)

            # client_hess_vec = state_dict_to_vector(client_hess)*p[i]  ## Weighting the hessian by the size of the dataset
            # convert the hessian to a vector
            # hessian_list.append(client_hess_vec)
            # hessian_list.append(client_hess)
    
    dict_results[alg +'local_model_test_accuracies_' + str(seed)+'_alpha_'+str(alpha)+'_eta_' + str(args['eta']) + 
                 '_h_init_' + str(args['hess_init']) + 
                 '_bs_' + str(args['bs'])+'_l_epoch_'+str(local_epochs)] = local_model_accs
    dict_results[alg +'local_model_test_losses_' + str(seed)+'_alpha_'+str(alpha)+"_eta_"+ str(args['eta']) + 
                 '_h_init_' + str(args['hess_init']) + 
                 '_bs_' + str(args['bs'])+'_l_epoch_'+str(local_epochs)] = local_model_loss

    ### Creating one-shot model depending on the algorithm
    # d is total number of parameters in the model
    # n is number of clients
    # p is the weights of the clients
    net_glob, model_mean, global_hessian = get_one_shot_model(alg, d,n,p,args,net_glob, models, model_vectors, \
    F_kfac_list, F_diag_list, hessian_vectors, dataset_val, dataset_train, dataset_train_global, \
    dataset_test_global, filename, net_cls_counts)
    
    # test_acc, test_loss = test_img(net_glob, dataset_test_global,args)
    test_acc, test_loss = test_img_global_posterior_avg(net_g=net_glob, datatest=dataset_test_global, 
                                                        args=args, posterior_mean=model_mean, posterior_cov=global_hessian)
    print (alg + " Test Acc. ", test_acc, "Test Loss", test_loss)
    dict_results[alg + '_test_loss_'+'seed_' + str(seed)+"_eta_"+ str(args['eta']) + 
                 '_h_init_' + str(args['hess_init']) + 
                 '_bs_' + str(args['bs'])+'_l_epoch_'+str(local_epochs)] = test_loss
    dict_results[alg + '_test_acc_'+'seed_' + str(seed)+"_eta_"+ str(args['eta']) + 
                 '_h_init_' + str(args['hess_init']) + 
                 '_bs_' + str(args['bs'])+'_l_epoch_'+str(local_epochs)] = test_acc
    
    with open(filename_csv, 'w') as csv_file:    
        writer = csv.writer(csv_file)
        for i in dict_results.keys():
            writer.writerow([i, dict_results[i]])
            

# with open(filename_csv, 'w') as csv_file:    
#   writer = csv.writer(csv_file)
#   for i in dict_results.keys():
#       writer.writerow([i, dict_results[i]])
