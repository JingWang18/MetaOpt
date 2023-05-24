import os
import time
import pprint

import copy
import time
import math
# import wandb
import argparse
import collections
from itertools import product
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import learn2learn as l2l
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functorch import make_functional_with_buffers, vmap, grad

from dppy.finite_dpps import FiniteDPP

from mixture import GaussianMixture
from nic import NIC

import pdb

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)

###################################################################################################

# ================ NTK ======================
def accuracy(predictions, targets):
    try:
        predictions = predictions.argmax(dim=1).view(targets.shape)
    except:
        targets = torch.nonzero(targets)[:,1]
        predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def flatten(grad: torch.Tensor):
    return torch.cat([grad_layer.reshape(-1) for grad_layer in grad])

def batch_flatten(batch_grad: torch.Tensor):
    batch_size = batch_grad[0].shape[0]
    return torch.cat([grad_layer.reshape(batch_size, -1) for grad_layer in batch_grad], dim=1)

def kernel_mats(net, gamma_train, gamma_test, device, kernels='both', batch_size=1):
    n_train = len(gamma_train)
    gamma_train = gamma_train.to(device)
    # the following computes the gradients with respect to all parameters
    grad_list = []
    for gamma in gamma_train:
        gamma = gamma.unsqueeze(0)
        loss = net(gamma)
        grad_list.append(torch.autograd.grad(
            loss, net.parameters(), retain_graph=True, create_graph=True))

    # testvstrain kernel
    if kernels=='both' or kernels=='testvtrain':
        n_pts = len(gamma_test)
        gamma_test = gamma_test.to(device)
        K_testvtrain = torch.zeros((n_pts, n_train)).to(device)
        for i, gamma in enumerate(gamma_test):
            gamma = gamma.unsqueeze(0)
            loss = net(gamma)
            grads = torch.autograd.grad(
                loss, net.parameters(), retain_graph=True, create_graph=True) # extract NN gradients

            for j in range(len(grad_list)):
                pt_grad = grad_list[j] # the gradients at the jth (out of 4) data point
                K_testvtrain[i, j] = sum(
                    [torch.sum(torch.mul(grads[u], pt_grad[u])) for u in range(len(grads))])

    # trainvstrain kernel
    if kernels=='both' or kernels=='trainvtrain':
        K_trainvtrain = torch.zeros((n_train, n_train)).to(device)
        for i in range(n_train):
            grad_i = grad_list[i]
            for j in range(i+1):
                grad_j = grad_list[j]
                K_trainvtrain[i, j] = sum(
                    [torch.sum(torch.mul(grad_i[u], grad_j[u])) for u in range(len(grad_j))])
                K_trainvtrain[j, i] = K_trainvtrain[i, j]

    if kernels=='both':
        return K_testvtrain, K_trainvtrain
    elif kernels=='trainvtrain':
        return K_trainvtrain
    elif kernels=='testvtrain':
        return K_testvtrain

def kernel_mats_batch(net, gamma_train, gamma_test, device, kernels='both', batch_size=4):
    gamma_train = gamma_train.to(device)
    n_train = len(gamma_train)
    # the following computes the gradients with respect to all parameters
    train_grad_list = []
    for i in range(0, n_train, batch_size):
        gamma = gamma_train[i:i+batch_size]
        batch = gamma.shape[0]
        loss = net(gamma)
        if len(loss.shape) > 1:
            loss = torch.sum(loss, dim=-1)
        for j in range(batch):
            grad = torch.autograd.grad(loss[j], net.parameters(), retain_graph=True, create_graph=True)
            train_grad_list.append(flatten(grad))

    # testvstrain kernel
    if kernels=='both' or kernels=='testvtrain':
        gamma_test = gamma_test.to(device)
        n_test = len(gamma_test)
        # the following computes the gradients with respect to all parameters
        K_testvtrain = torch.zeros((n_test, n_train)).to(device)
        test_grad_list = []
        for i in range(0, n_test, batch_size):
            gamma = gamma_test[i:i+batch_size]
            batch = gamma.shape[0]
            loss = net(gamma)
            if len(loss.shape) > 1:
                loss = torch.sum(loss, dim=-1)
            for j in range(batch):
                grad = torch.autograd.grad(loss[j], net.parameters(), retain_graph=True, create_graph=True)
                test_grad_list.append(flatten(grad))

        for test_start_idx in range(0, n_test, batch_size):
            test_end_idx = min(n_test, test_start_idx + batch_size)
            for train_start_idx in range(0, n_train, batch_size):
                train_end_idx = min(n_train, train_start_idx + batch_size)

                K_testvtrain[test_start_idx:test_end_idx, train_start_idx:train_end_idx] =\
                    torch.matmul(
                        torch.stack(test_grad_list[test_start_idx:test_end_idx]),
                        torch.stack(train_grad_list[train_start_idx:train_end_idx]).T)

    # trainvstrain kernel
    if kernels=='both' or kernels=='trainvtrain':
        K_trainvtrain = torch.zeros((n_train, n_train)).to(device)
        for start_idx1 in range(0, n_train, batch_size):
            end_idx1 = min(n_train, start_idx1 + batch_size)
            for start_idx2 in range(0, n_train, batch_size):
                end_idx2 = min(n_train, start_idx2 + batch_size)

                K_trainvtrain[start_idx1:end_idx1, start_idx2:end_idx2] =\
                    torch.matmul(
                        torch.stack(train_grad_list[start_idx1:end_idx1]),
                        torch.stack(train_grad_list[start_idx2:end_idx2]).T)

    if kernels=='both':
        return K_testvtrain, K_trainvtrain
    elif kernels=='trainvtrain':
        return K_trainvtrain
    elif kernels=='testvtrain':
        return K_testvtrain

def kernel_grads_fast(grad_vmap, gamma_train, params, buffers):
    # the following computes the gradients with respect to all parameters
    phi_list = []
    for i in range(0, gamma_train.shape[0], 128):
        phi = batch_flatten(grad_vmap(params, buffers, gamma_train[i:i+128]))
        phi_list.append(phi)
    return torch.cat(phi_list, dim=0)

def kernel_grads(net, gamma_train, device, batch_size=4):
    gamma_train = gamma_train.to(device)
    n_train = len(gamma_train)
    # the following computes the gradients with respect to all parameters
    phi = []
    for i in range(0, n_train, batch_size):
        gamma = gamma_train[i:i+batch_size]
        batch = gamma.shape[0]
        loss = net(gamma)
        for j in range(batch):
            grad = torch.autograd.grad(loss[j], net.parameters(), retain_graph=True, create_graph=True)
            phi.append(flatten(grad))

    phi = torch.stack(phi)
    return phi


# ======================= active learning ===========================
@torch.no_grad()
def active_dpp(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
               cand_ratio=0.25, per_class=False):
    learner.eval()
    fmodel, params, buffers = make_functional_with_buffers(learner)

    def _compute_loss(params, buffers, sample):
        batch = sample.unsqueeze(0)
        loss = torch.sum(fmodel(params, buffers, batch)) # sum to make 0-dim tensor
        return loss

    ft_compute_grad = grad(_compute_loss)
    grad_vmap = vmap(ft_compute_grad, in_dims=(None, None, 0))

    # data, labels, indices = batch

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    # pdb.set_trace()

    if features is not None:
        features.eval()
        batch_size = 840
        # total_data = dataset[cand_indices][0]
        total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1],
                                 dataset[0][0].shape[2])
        for idx, x in enumerate(cand_indices):
            total_data[idx] = dataset[x][0]

        phi_total = []
        for i in range(0, total_data.shape[0], batch_size):
            phi = features(total_data[i:i+batch_size].cuda())
            phi_total.append(phi)
        phi_total = torch.concat(phi_total).double()
        features.train()
    else:
        batch_size = 840
        # total_data = dataset[cand_indices][0]
        total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1],
                                 dataset[0][0].shape[2])
        for idx, x in enumerate(cand_indices):
            total_data[idx] = dataset[x][0]
        phi_total = []
        for i in range(0, total_data.shape[0], batch_size):
            phi = kernel_grads_fast(
                    grad_vmap, total_data[i:i+batch_size].cuda(), params, buffers).double() # phi: (N, d)
            phi_total.append(phi)
        phi_total = torch.concat(phi_total).double()


    selected_indices, selected_labels = [], []
    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            cls_cand_bool = cand_labels==cls
            phi = phi_total[cls_cand_bool]

            _, S, Vh = torch.linalg.svd(phi.T, full_matrices=False) # U: dxN, S: NxN, Vh: NxN
            eig_val, eig_vector = (S**2).detach().cpu().numpy(), (Vh.T).detach().cpu().numpy()
            DPP = FiniteDPP('likelihood', **{'L_eig_dec': (eig_val, eig_vector)})

            dpp_idx = DPP.sample_exact_k_dpp(size=num_query)

            selected_idx = cls_cand_indices[dpp_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        _, S, Vh = torch.linalg.svd(phi_total.T, full_matrices=False) # U: dxN, S: NxN, Vh: NxN
        eig_val, eig_vector = (S**2).detach().cpu().numpy(), (Vh.T).detach().cpu().numpy()
        DPP = FiniteDPP('likelihood', **{'L_eig_dec': (eig_val, eig_vector)})

        dpp_idx = DPP.sample_exact_k_dpp(size=ways * train_shots)
        selected_indices = cand_indices[dpp_idx]
        selected_labels = cand_labels[dpp_idx]

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    return data_support, labels_support, data_query, labels_query


@torch.no_grad()
def active_dpp_precomp(dataset, features, device,
                       cand_ratio=0.5, num_img_per_class=600, batch_size=600, per_class=False):
    assert features is not None
    features.eval()
    indices = list(range(len(dataset)))
    total_data = dataset[indices][0]
    phi_total = []
    for i in range(0, len(dataset), batch_size):
        phi = features(total_data[i:i+batch_size].to(device))
        phi_total.append(phi)

    phi_total = torch.concat(phi_total).view(
        -1, num_img_per_class, phi_total[0].shape[-1])
    kernel_total = torch.bmm(phi_total, phi_total.transpose(1, 2))

    def _dpp_query_func(dataset, batch, ways, train_shots, test_shots):
        start_time = time.time()
        data, labels, indices = batch
        adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
        selection = np.arange(ways) * (train_shots + test_shots)
        for offset in range(train_shots):
            adaptation_indices_bool[selection + offset] = True
        evaluation_indices_bool = ~adaptation_indices_bool
        adaptation_labels, adaptation_indices =\
            labels[adaptation_indices_bool], indices[adaptation_indices_bool]
        evaluation_data, evaluation_labels, evaluation_indices =\
            data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
        evaluation_indices = evaluation_indices.detach().cpu().numpy()

        images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

        (cand_indices, cand_labels) = dataset.load_candidates(
            adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
        random_indices = np.random.choice(
            len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
        cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

        selected_indices, selected_labels = [], []

        #for cls in images_per_class.keys():
        #    # detect outliers
        #    U, _, _ = torch.linalg.svd(phi_total[0], full_matrices=False)
        #    H = torch.matmul(U, U.T)
        #    threshold = 2 * torch.trace(H) /  H.shape[0]
        #    print(torch.sum(H > threshold))

        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            class_idx = int(cls_cand_indices[0] // num_img_per_class)
            indices_per_cls = cls_cand_indices % num_img_per_class

            #phi = phi_total[cls_cand_indices] # phi: (n, d)
            #_, S, Vh = torch.linalg.svd(phi.T, full_matrices=False) # U: dxN, S: nxn, Vh: nxn
            #eig_val, eig_vector = (S**2).detach().cpu().numpy(), (Vh.T).detach().cpu().numpy()
            #DPP = FiniteDPP('likelihood', **{'L_eig_dec': (eig_val, eig_vector)})
            kernel = kernel_total[class_idx][indices_per_cls][:, indices_per_cls].detach().cpu().numpy()
            DPP = FiniteDPP('likelihood', **{'L': kernel}) #phi: (N, d)
            dpp_idx = DPP.sample_exact_k_dpp(size=num_query)

            selected_idx = cls_cand_indices[dpp_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query

        data = torch.zeros_like(data)
        labels = torch.zeros_like(labels)
        indices = torch.zeros_like(indices)
        data[adaptation_indices_bool] = dataset[np.concatenate(selected_indices)][0]
        data[evaluation_indices_bool] = evaluation_data
        labels[adaptation_indices_bool] = torch.tensor(selected_labels)
        labels[evaluation_indices_bool] = evaluation_labels
        indices[adaptation_indices_bool] = torch.from_numpy(np.concatenate(selected_indices))
        indices[evaluation_indices_bool] = torch.from_numpy(evaluation_indices)
        batch = (data, labels, indices)
        return batch
    features.train()
    return _dpp_query_func


@torch.no_grad()
def active_nic(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
               cand_ratio=0.5, per_class=False):

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    features.eval()

    total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(cand_indices):
        total_data[idx] = dataset[x][0]

    phi_total = []
    for i in range(0, total_data.shape[0], 840):
        phi = features(total_data[i:i+840].cuda())
        phi_total.append(phi)
    phi_total = torch.cat(phi_total)

    selected_indices, selected_labels = [], []

    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            phi = phi_total[cand_labels==cls] # phi: (N, d)

            nic = NIC(k=num_query)
            nic.fit(phi)
            nic_idx = nic.get_nearest_samples(phi).detach().cpu().numpy()

            selected_idx = cls_cand_indices[nic_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        nic = NIC(k=ways * train_shots)
        nic.fit(phi_total)
        nic_idx = nic.get_nearest_samples(phi_total).detach().cpu().numpy()

        selected_indices = cand_indices[nic_idx]
        selected_labels = cand_labels[nic_idx]

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    features.train()
    return data_support, labels_support, data_query, labels_query



@torch.no_grad()
def active_typiclust(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
               cand_ratio=0.5, per_class=False):

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    features.eval()
    total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(cand_indices):
        total_data[idx] = dataset[x][0]

    phi_total = []
    for i in range(0, total_data.shape[0], 840):
        phi = features(total_data[i:i+840].cuda())
        phi_total.append(phi)
    phi_total = torch.cat(phi_total)

    selected_indices, selected_labels = [], []

    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            phi = phi_total[cand_labels==cls] # phi: (N, d)

            gmm = GaussianMixture(
                n_components=num_query, n_features=phi.shape[1], covariance_type="diag").cuda()
            gmm.fit(phi, n_iter=0)
            assignments = gmm.predict(phi)

            typiclust_idx = []
            for i in range(num_query):
                local_phi = phi[assignments==i]
                local_indices = np.arange(local_phi.shape[0])
                typicality = calculate_typicality(local_phi, 20)
                local_idx = typicality.argmax()
                typiclust_idx.append(local_indices[local_idx])

            selected_idx = cls_cand_indices[typiclust_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        gmm = GaussianMixture(
            n_components=ways * train_shots, n_features=phi_total.shape[1], covariance_type="diag").cuda()
        gmm.fit(phi_total, n_iter=0)
        assignments = gmm.predict(phi_total)

        typiclust_idx = []
        for i in range(ways * train_shots):
            local_phi = phi_total[assignments==i]
            local_indices = np.arange(local_phi.shape[0])
            typicality = calculate_typicality(local_phi, 20)
            local_idx = typicality.argmax()
            typiclust_idx.append(local_indices[local_idx])

        selected_indices = cand_indices[typiclust_idx]
        selected_labels = cand_labels[typiclust_idx]

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    features.train()
    return data_support, labels_support, data_query, labels_query


@torch.no_grad()
def active_gmm(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
               cand_ratio=0.5, per_class=False):

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    features.eval()
    # total_data = dataset[cand_indices][0]

    total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(cand_indices):
        total_data[idx] = dataset[x][0]

    phi_total = []
    for i in range(0, total_data.shape[0], 64):
        phi = features(total_data[i:i+64].cuda())
        phi_total.append(phi)
    phi_total = torch.cat(phi_total)

    selected_indices, selected_labels = [], []

    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            phi = phi_total[cand_labels==cls] # phi: (N, d)

            gmm = GaussianMixture(
                n_components=num_query, n_features=phi.shape[1], covariance_type="diag").cuda()
            gmm.fit(phi, n_iter=100)
            gmm_idx = gmm.get_nearest_samples(phi).detach().cpu().numpy()

            selected_idx = cls_cand_indices[gmm_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        gmm = GaussianMixture(
            n_components=ways * train_shots, n_features=phi_total.shape[1], covariance_type="diag").cuda()
        gmm.fit(phi_total, n_iter=100)
        gmm_idx = gmm.get_nearest_samples(phi_total).detach().cpu().numpy()

        selected_indices = cand_indices[gmm_idx]
        selected_labels = cand_labels[gmm_idx]

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]

    # data_support = dataset[selected_indices][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    # data[adaptation_indices_bool] = dataset[selected_indices][0]
    # data[evaluation_indices_bool] = evaluation_data
    # labels[adaptation_indices_bool] = torch.tensor(selected_labels)
    # labels[evaluation_indices_bool] = evaluation_labels
    # indices[adaptation_indices_bool] = torch.from_numpy(selected_indices)
    # indices[evaluation_indices_bool] = torch.from_numpy(evaluation_indices)
    # batch = (data, labels, indices)
    features.train()
    return data_support, labels_support, data_query, labels_query



@torch.no_grad()
def active_gmm_train(learner, dataset, batch, ways, features, train_shots, test_shots,
                     device, cand_ratio=0.5, per_class=False):
    data, labels, indices = batch

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_data, adaptation_labels, adaptation_indices =\
        data[adaptation_indices_bool], labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_labels, evaluation_indices =\
        labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, adaptation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    features.eval()
    total_data = dataset[cand_indices][0]
    phi_total = []
    for i in range(0, total_data.shape[0], 840):
        phi = features(total_data[i:i+840].to(device))
        phi_total.append(phi)
    phi_total = torch.cat(phi_total)

    selected_indices, selected_labels = [], []

    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            phi = phi_total[cand_labels==cls] # phi: (N, d)

            gmm = GaussianMixture(
                n_components=num_query, n_features=phi.shape[1], covariance_type="diag").to(device)
            gmm.fit(phi, n_iter=100)
            gmm_idx = gmm.get_nearest_samples(phi).detach().cpu().numpy()

            selected_idx = cls_cand_indices[gmm_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        gmm = GaussianMixture(
            n_components=ways * train_shots, n_features=phi_total.shape[1], covariance_type="diag").to(device)
        gmm.fit(phi_total, n_iter=100)
        gmm_idx = gmm.get_nearest_samples(phi_total).detach().cpu().numpy()

        selected_indices = cand_indices[gmm_idx]
        selected_labels = cand_labels[gmm_idx]

    data[adaptation_indices_bool] = adaptation_data
    data[evaluation_indices_bool] = dataset[selected_indices][0]
    labels[adaptation_indices_bool] = adaptation_labels
    labels[evaluation_indices_bool] = torch.tensor(selected_labels)
    indices[adaptation_indices_bool] = torch.from_numpy(evaluation_indices)
    indices[evaluation_indices_bool] = torch.from_numpy(selected_indices)
    batch = (data, labels, indices)
    features.train()
    return batch


@torch.no_grad()
def active_gmm_precomp(learner, dataset, features, device,
                       cand_ratio=1.0, num_img_per_class=600, batch_size=600, per_class=False):
    learner.eval()
    assert features is not None
    features.eval()
    indices = list(range(len(dataset)))
    total_data = dataset[indices][0]
    phi_total = []
    for i in range(0, len(dataset), batch_size):
        phi = features(total_data[i:i+batch_size].to(device))
        phi_total.append(phi)

    phi_total = torch.concat(phi_total).view(
        -1, num_img_per_class, phi_total[0].shape[-1])

    def _gmm_query_func(dataset, batch, ways, train_shots, test_shots):
        data, labels, indices = batch
        adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
        selection = np.arange(ways) * (train_shots + test_shots)
        for offset in range(train_shots):
            adaptation_indices_bool[selection + offset] = True
        evaluation_indices_bool = ~adaptation_indices_bool
        adaptation_labels, adaptation_indices =\
            labels[adaptation_indices_bool], indices[adaptation_indices_bool]
        evaluation_data, evaluation_labels, evaluation_indices =\
            data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
        evaluation_indices = evaluation_indices.detach().cpu().numpy()

        #(cand_indices, cand_labels) = dataset.load_candidates(
        #    adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
        (cand_indices, cand_labels) = dataset.load_candidates(
            adaptation_indices, np.array([]), adaptation_labels)
        sorted_indices = np.argsort(cand_indices)
        cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
        images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

        selected_adaptation_indices, selected_adaptation_labels = [], []
        selected_evaluation_indices, selected_evaluation_labels = [], []

        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            class_idx = int(cls_cand_indices[0] // num_img_per_class)
            #indices_per_cls = cls_cand_indices % num_img_per_class

            phi = phi_total[class_idx] # phi: (N, d)

            #gmm = GaussianMixture(
            #    n_components=num_query, n_features=phi.shape[1], covariance_type="diag").to(device)
            #gmm.fit(phi, n_iter=100)
            #adaptation_idx = gmm.get_nearest_samples(phi).detach().cpu().numpy()
            #indices = np.random.choice(np.array(
            #    list(set(np.arange(phi.shape[0])).difference(set(adaptation_idx)))),
            #    size=test_shots * 10, replace=False)

            #logits = learner(phi[indices])
            #log_probs = torch.log_softmax(logits, dim=-1)
            #entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
            #evaluation_idx = indices[
            #    torch.topk(entropy, k=test_shots).indices.detach().cpu().numpy()]

            # ============ best so far for 1shot =========================
            gmm = GaussianMixture(
                n_components=num_query, n_features=phi.shape[1], covariance_type="diag").to(device)
            gmm.fit(phi)

            evaluation_idx = gmm.get_nearest_samples(phi).detach().cpu().numpy()
            adaptation_idx = np.random.choice(
                list(set(np.arange(phi.shape[0])).difference(set(evaluation_idx))),
                size=num_query, replace=False)
            # ==================================================
            #adaptation_idx = gmm.get_nearest_samples(phi).detach().cpu().numpy()
            #evaluation_idx = np.random.choice(
            #    list(set(np.arange(phi.shape[0])).difference(set(adaptation_idx))),
            #    size=num_query, replace=False)

            # done so far: adapt gmm (okay but not as good), adapt + eval gmm (bad), adapt gmm +
            # eval farest (bad), adapt farest + eval gmm (bad), adapt random + eval gmm (okay
            # but not as good)
            selected_adaptation_indices.append(cls_cand_indices[adaptation_idx])
            selected_adaptation_labels += [cls] * num_query
            selected_evaluation_indices.append(cls_cand_indices[evaluation_idx])
            selected_evaluation_labels += [cls] * test_shots

        data = torch.zeros_like(data)
        labels = torch.zeros_like(labels)
        indices = torch.zeros_like(indices)
        data[adaptation_indices_bool] = dataset[
            np.concatenate(selected_adaptation_indices)][0]
        data[evaluation_indices_bool] = dataset[
            np.concatenate(selected_evaluation_indices)][0]
        #data[evaluation_indices_bool] = evaluation_data
        labels[adaptation_indices_bool] = torch.tensor(selected_adaptation_labels)
        labels[evaluation_indices_bool] = torch.tensor(selected_evaluation_labels)
        #labels[evaluation_indices_bool] = evaluation_labels
        indices[adaptation_indices_bool] = torch.from_numpy(np.concatenate(selected_adaptation_indices))
        indices[evaluation_indices_bool] = torch.from_numpy(np.concatenate(selected_evaluation_indices))
        batch = (data, labels, indices)
        return batch
    learner.train()
    features.train()
    return _gmm_query_func

@torch.no_grad()
def active_margin(learner, dataset, batch, ways, features, train_shots, test_shots,
                  cand_ratio=0.25, per_class=False):
    learner.eval()
    data, labels, indices = batch

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    if features is not None:
        features.eval()
        batch_size = 840
        total_data = dataset[cand_indices][0]
        phi_total = []
        for i in range(0, total_data.shape[0], batch_size):
            phi = features(total_data[i:i+batch_size].cuda())
            phi_total.append(phi)
        phi_total = torch.concat(phi_total)
        logits_total = learner(phi_total)
        probs = F.softmax(logits_total, dim=-1)
    else:
        batch_size = 840
        total_data = dataset[cand_indices][0]
        logits_total = []
        for i in range(0, total_data.shape[0], batch_size):
            logits = learner(total_data[i:i+batch_size].cuda())
            logits_total.append(logits)
        logits_total = torch.concat(logits_total)
        probs = F.softmax(logits_total, dim=-1)

    topk_vals, _ = torch.topk(probs, k=2)
    diffs = topk_vals[:,0] - topk_vals[:,1]

    selected_indices, selected_labels = [], []
    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            diff = diffs[cand_labels==cls] # phi: (N, d)

            _, margin_idx = torch.topk(diff, k=num_query, largest=False)
            margin_idx = margin_idx.detach().cpu().numpy()

            selected_idx = cls_cand_indices[margin_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        _, margin_idx = torch.topk(diffs, k=ways * train_shots, largest=False)
        margin_idx = margin_idx.detach().cpu().numpy()

        selected_indices = cand_indices[margin_idx]
        selected_labels = cand_labels[margin_idx]

    data = torch.zeros_like(data)
    labels = torch.zeros_like(labels)
    indices = torch.zeros_like(indices)
    data[adaptation_indices_bool] = dataset[selected_indices][0]
    data[evaluation_indices_bool] = evaluation_data
    labels[adaptation_indices_bool] = torch.tensor(selected_labels)
    labels[evaluation_indices_bool] = evaluation_labels
    indices[adaptation_indices_bool] = torch.tensor(selected_indices)
    indices[evaluation_indices_bool] = torch.tensor(evaluation_indices)
    batch = (data, labels, indices)
    if features is not None:
        features.train()
    return batch

@torch.no_grad()
def active_ent(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
               cand_ratio=0.25, per_class=False):
    learner.eval()

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    cand_indices = cand_indices.tolist()

    if features is not None:
        features.eval()
        # batch_size = 840

        total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])

        for idx, x in enumerate(cand_indices):
            total_data[idx] = dataset[x][0]
            # total_label[idx] = cand_labels[x]

        # phi_total = []
        # for i in range(0, total_data.shape[0], batch_size):
        #     print(total_data.shape)
        #     phi = features(total_data[i:i+batch_size].cuda())
        #     # print(phi.shape)
        #     phi_total.append(phi)

        # phi_total = torch.concat(phi_total)
        # print(phi_total.shape)
        phi_total = features(total_data.cuda())

        phi_total = phi_total.view(1, -1, 2560)

        phi_labels = cand_labels.reshape(1,-1)
        phi_labels = torch.from_numpy(phi_labels).cuda()

        logits_total = learner(phi_total[:,5*15+1:,:], phi_total[:,:5*15,:], phi_labels[:,:5*15], 5, 15)
        probs = F.softmax(logits_total, dim=-1)

        cand_labels = cand_labels.reshape(-1)
    # else:
    #     batch_size = 840
    #     total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    #     for idx, x in enumerate(cand_indices):
    #         total_data[idx] = dataset[x][0]
    #
    #     logits_total = []
    #     for i in range(0, total_data.shape[0], batch_size):
    #         logits = learner(phi_total[i:i+int(batch_size*0.75)], phi_total[i+int(batch_size*0.75)+1:i+batch_size], cand_labels[i+int(batch_size*0.75)+1:i+batch_size], 5, 15)
    #         #logits = learner(total_data[i:i+batch_size].cuda())
    #         logits_total.append(logits)
    #     logits_total = torch.concat(logits_total)
    #     probs = F.softmax(logits_total, dim=-1)

    entropies = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)

    selected_indices, selected_labels = [], []
    if per_class:
        for cls in images_per_class.keys():
            num_query = images_per_class[cls]
            cls_cand_indices = cand_indices[cand_labels==cls]
            entropy = entropies[cand_labels==cls] # phi: (N, d)

            _, entropy_idx = torch.topk(entropy, k=num_query, largest=True)
            entropy_idx = entropy_idx.detach().cpu().numpy()

            selected_idx = cls_cand_indices[entropy_idx]
            selected_indices.append(selected_idx)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        _, entropy_idx = torch.topk(entropies, k=ways * train_shots, largest=True)
        entropy_idx = entropy_idx.detach().cpu().numpy()
        # pdb.set_trace()
        selected_indices = cand_indices[entropy_idx]
        selected_labels = cand_labels[entropy_idx]

    data = torch.zeros_like(data)
    labels = torch.zeros_like(labels)
    indices = torch.zeros_like(indices)

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]

    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = labels[evaluation_indices_bool] = evaluation_labels

    # data[adaptation_indices_bool] = dataset[selected_indices][0]
    # data[evaluation_indices_bool] = evaluation_data
    # labels[adaptation_indices_bool] = torch.tensor(selected_labels)
    # labels[evaluation_indices_bool] = evaluation_labels
    # indices[adaptation_indices_bool] = torch.tensor(selected_indices)
    # indices[evaluation_indices_bool] = torch.tensor(evaluation_indices)
    # batch = (data, labels, indices)
    if features is not None:
        features.train()
    return data_support, labels_support, data_query, labels_query



@torch.no_grad()
def active_prob_cover(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
                      device, cand_ratio=1.0, p=2.0, images_per_class=600, per_class=False):

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    assert features is not None
    features.eval()
    batch_size = 840

    total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(cand_indices):
        total_data[idx] = dataset[x][0]

    phi_total = []
    for i in range(0, total_data.shape[0], batch_size):
        phi = features(total_data[i:i+batch_size].to(device))
        phi_total.append(phi)
    phi_total = torch.concat(phi_total)

    selected_indices, selected_labels = [], []
    if per_class:
        for cls in images_per_class.keys():
            # distance comparisons are done in batches to reduce memory consumption
            cls_cand_indices = cand_indices[cand_labels==cls]
            phi = phi_total[cand_labels==cls]
            dist = torch.cdist(phi, phi)
            #mask = dist < torch.quantile(dist, p) # TODO: change it to a val
            #print(torch.quantile(dist, p))
            mask = dist < p
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T

            graph_df = pd.DataFrame({
                'x': x.detach().cpu().numpy(), 'y': y.detach().cpu().numpy(), 'd': dist[mask].detach().cpu().numpy()})

            #print(f'Finished constructing graph for class{cls} using p={p}')
            #print(f'Graph contains {len(graph_df)} edges.')

            cur_df = graph_df.copy()
            prob_cover_indices = []
            covered_samples = np.array([])
            num_query = images_per_class[cls]
            for i in range(num_query):
                coverage = len(covered_samples) / phi.shape[0]

                # selecting the sample with the highest degree
                degrees = np.bincount(cur_df.x, minlength=phi.shape[0])
                prob_cover_cls_idx = degrees.argmax()
                # prob_cover_idx = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

                # removing incoming edges to newly covered samples
                new_covered_samples = cur_df.y[(cur_df.x == prob_cover_cls_idx)].values
                assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
                cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

                covered_samples = np.concatenate([covered_samples, new_covered_samples])
                prob_cover_indices.append(cls_cand_indices[prob_cover_cls_idx])
                #print(f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')

            assert len(prob_cover_indices) == num_query, 'added a different number of samples'
            selected_indices.append(prob_cover_indices)
            selected_labels += [cls] * num_query
        selected_indices = np.concatenate(selected_indices)
    else:
        dist = torch.cdist(phi_total, phi_total)
        mask = dist < torch.quantile(dist, p) # TODO: change it to a val
        # saving edges using indices list - saves memory.
        x, y = mask.nonzero().T

        graph_df = pd.DataFrame({
            'x': x.detach().cpu().numpy(), 'y': y.detach().cpu().numpy(), 'd': dist[mask].detach().cpu().numpy()})

        #print(f'Finished constructing graph for class{cls} using p={p}')
        #print(f'Graph contains {len(graph_df)} edges.')

        cur_df = graph_df.copy()
        covered_samples = np.array([])
        for i in range(ways * train_shots):
            coverage = len(covered_samples) / phi_total.shape[0]

            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=phi_total.shape[0])
            prob_cover_cls_idx = degrees.argmax()
            # prob_cover_idx = np.random.choice(degrees.argsort()[::-1][:5]) # the paper randomizes selection

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == prob_cover_cls_idx)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected_indices.append(cand_indices[prob_cover_cls_idx])
            selected_labels.append(cand_labels[prob_cover_cls_idx])

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    features.train()
    return data_support, labels_support, data_query, labels_query


@torch.no_grad()
def active_coreset(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
                      device, cand_ratio=1.0, p=0.1, images_per_class=600, per_class=False, is_mip=False):

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(
        adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    random_indices = np.random.choice(
        len(cand_indices), replace=False, size=int(len(cand_indices) * cand_ratio))
    cand_indices, cand_labels = cand_indices[random_indices], cand_labels[random_indices]

    assert features is not None
    features.eval()
    batch_size = 840

    total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(cand_indices):
        total_data[idx] = dataset[x][0]

    phi_total = []
    for i in range(0, total_data.shape[0], batch_size):
        phi = features(total_data[i:i+batch_size].to(device))
        phi_total.append(phi)
    phi_total = torch.concat(phi_total)

    selected_indices, selected_labels = [], []
    if per_class:
        for cls in images_per_class.keys():
            budget_size = images_per_class[cls]
            # distance comparisons are done in batches to reduce memory consumption
            cls_cand_indices = cand_indices[cand_labels==cls]
            phi = phi_total[cand_labels==cls]

            greedy_indices = [None for i in range(budget_size)]
            greedy_indices_counter = 0

            init_idx = np.random.choice(phi.shape[0], size=1, replace=False)
            labeled = phi[[init_idx.item()]]
            unlabeled_indices = torch.from_numpy(
                np.array(list(set(range(phi.shape[0])).difference(set(init_idx.tolist()))))).to(device)
            unlabeled = phi[unlabeled_indices]
            greedy_indices[0] = init_idx.item()
            greedy_indices_counter += 1
            min_dist = None
            amount = budget_size - 1

            for i in range(amount):
                dist = compute_dists(phi[
                    greedy_indices[greedy_indices_counter-1], :].reshape((1, unlabeled.shape[1])), unlabeled)

                if min_dist is None:
                    min_dist = dist
                else:
                    min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))

                min_dist, _ = torch.min(min_dist, dim=0)
                min_dist = min_dist.reshape((1, min_dist.shape[0]))
                _, farthest = torch.max(min_dist, dim=1)
                greedy_indices[greedy_indices_counter] = farthest.item()
                greedy_indices_counter += 1

            selected_indices.append(cls_cand_indices[greedy_indices])
            selected_labels += [cls] * len(greedy_indices)
        selected_indices = np.concatenate(selected_indices)
    else:
        budget_size = ways * train_shots
        greedy_indices = [None for i in range(budget_size)]
        greedy_indices_counter = 0

        init_idx = np.random.choice(len(cand_indices), size=1, replace=False)
        labeled = phi_total[[init_idx.item()]]
        unlabeled_indices = torch.from_numpy(np.array(
            list(set(range(len(cand_indices))).difference(set(init_idx.tolist()))))).to(device)
        unlabeled = phi_total[unlabeled_indices]
        greedy_indices[0] = init_idx.item()
        greedy_indices_counter += 1

        min_dist = None
        amount = budget_size - 1

        for i in range(amount):
            dist = compute_dists(phi_total[
                greedy_indices[greedy_indices_counter-1], :].reshape((1, unlabeled.shape[1])), unlabeled)

            if min_dist is None:
                min_dist = dist
            else:
                min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))

            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices[greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        selected_indices = cand_indices[greedy_indices]
        selected_labels = cand_labels[greedy_indices]

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    features.train()
    return data_support, labels_support, data_query, labels_query


@torch.no_grad()
def active_random(dataset, cand_indices, query, query_labels, ways, train_shots):

    random_idx = np.random.choice(
        len(cand_indices), size=8*ways * train_shots, replace=False)   # 8 is the episode per batch

    selected_idx = [cand_indices[i] for i in random_idx]

    data_support = torch.randn(len(selected_idx), dataset[0][0].shape[0], dataset[0][0].shape[1],
                               dataset[0][0].shape[2])
    labels_support = torch.randn(len(selected_idx), 1)

    for idx, x in enumerate(selected_idx):
        data_support[idx] = dataset[x][0]
        labels_support[idx] = dataset[x][1]

    data_query = query
    labels_query = query_labels

    return data_support, labels_support, data_query, labels_query


@torch.no_grad()
def active_vopt_total(learner, dataset, data, labels, indices, ways, features, train_shots, test_shots,
                     device, cand_ratio=1.0, per_class=False):

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels, adaptation_indices =\
        labels[adaptation_indices_bool], indices[adaptation_indices_bool]
    evaluation_data, evaluation_labels, evaluation_indices =\
        data[evaluation_indices_bool], labels[evaluation_indices_bool], indices[evaluation_indices_bool]
    evaluation_indices = evaluation_indices.detach().cpu().numpy()

    images_per_class = Counter(adaptation_labels.detach().cpu().numpy().tolist())

    # find all candidate points
    (cand_indices, cand_labels) = dataset.load_candidates(adaptation_indices, evaluation_indices, adaptation_labels) # cand_labels: task labels (0~ways)
    sorted_indices = np.argsort(cand_indices)
    cand_indices, cand_labels = cand_indices[sorted_indices], cand_labels[sorted_indices]

    total_data = torch.randn(len(cand_indices), dataset[0][0].shape[0], dataset[0][0].shape[1], dataset[0][0].shape[2])
    for idx, x in enumerate(cand_indices):
        total_data[idx] = dataset[x][0]

    features.eval()
    phi_total = features(total_data.cuda()) # phi: (N, d)

    # ========================== vopt =================================
    #U, _, _ = torch.linalg.svd(phi_total.double(), full_matrices=False)
    #H = torch.matmul(U, U.T)
    #p = (torch.diag(H) / torch.trace(H)).detach().cpu().numpy()
    #vopt_idx = np.random.choice(len(p), size=ways * train_shots, replace=False, p=p)

    # ========================== bias =================================
    # Adapt the model

    tmp_cand_indices = copy.deepcopy(cand_indices)
    rand_indices = np.arange(len(tmp_cand_indices))
    np.random.shuffle(rand_indices)
    rand_indices = rand_indices[:1000]
    #tmp_cand_indices = tmp_cand_indices[rand_indices]
    tmp_cand_labels = copy.deepcopy(cand_labels)
    adaptation_data = torch.tensor([]).to(device)
    selected_indices, selected_labels = [], []
    for k in range(ways * train_shots):
        costs = []
        #for i in range(len(tmp_cand_indices)):
        for i in rand_indices:
            print(f'{k},{i}')
            if adaptation_data.shape[0] == 0:
                tmp_adaptation_data = phi_total[[i]]
            else:
                tmp_adaptation_data = torch.cat((adaptation_data, phi_total[[i]]), dim=0)
            print('here')
            K_testvtrain = torch.matmul(phi_total, tmp_adaptation_data.T).double()
            K_trainvtrain = torch.matmul(tmp_adaptation_data, tmp_adaptation_data.T).double()
            print(' or here')
            try:
                LU, pivots = torch.linalg.lu_factor(K_trainvtrain + 1e-8 * torch.eye(tmp_adaptation_data.shape[0]).to(device))
                print('or or here')
                K_trainvtrain_inv_K_testvtrain_T = torch.linalg.lu_solve(
                    LU, pivots, K_testvtrain.T)
            except:
                import IPython; IPython.embed()
            cost = torch.trace(
                torch.matmul(K_testvtrain, K_trainvtrain_inv_K_testvtrain_T)).detach()
            #K_trainvtrain_inv = torch.inverse(K_trainvtrain + 1e-8 * torch.eye(tmp_adaptation_data.shape[0]).to(device))
            #cost = torch.trace(
            #    torch.matmul(torch.matmul(K_testvtrain, K_trainvtrain_inv), K_testvtrain.T)).detach()
            costs.append(cost)

        idx = torch.argmax(torch.tensor(costs)).item()
        idx = rand_indices[idx]
        adaptation_data = torch.cat((adaptation_data, phi_total[[idx]]), dim=0)

        selected_indices.append(tmp_cand_indices[idx])
        selected_labels.append(tmp_cand_labels[idx])
        #vopt_idx.append(idx)
        tmp_cand_indices = np.delete(tmp_cand_indices, [idx])
        tmp_cand_labels = np.delete(tmp_cand_labels, [idx])
        phi_total = torch_delete(phi_total, [idx], dim=0)
    # ===================================================================

    data_support = torch.randn(len(selected_indices), dataset[0][0].shape[0], dataset[0][0].shape[1],
                               dataset[0][0].shape[2])
    for idx, x in enumerate(selected_indices):
        data_support[idx] = dataset[x][0]
    data_query = evaluation_data
    labels_support = torch.tensor(selected_labels)
    labels_query = evaluation_labels

    features.train()
    return data_support, labels_support, data_query, labels_query


# =============== t-sne ==================
@torch.no_grad()
def save_tsne_features(dataset, features, num_classes, device, save_path, batch_size=840, step_size=1):
    features.eval()
    indices = list(range(0, 600 * num_classes, step_size))
    total_data, total_labels = dataset[indices]
    total_phi = []
    for i in range(0, total_data.shape[0], batch_size):
        phi = features(total_data[i:i+batch_size].to(device))
        total_phi.append(phi.detach().cpu().numpy())

    total_phi = np.concatenate(total_phi, axis=0)
    tsne = TSNE(n_components=2)
    phi_tsne = tsne.fit_transform(total_phi)
    #total_phi = torch.cat(total_phi, dim=0)
    #tsne = TSNE(n_components=2, n_iter=10000, verbose=True)
    #phi_tsne = tsne.fit_transform(total_phi) # (N, D) -> (N, 2)

    tsne_result_df = pd.DataFrame(
        {'tsne_1': phi_tsne[:,0],
         'tsne_2': phi_tsne[:,1],
         'label': np.array(total_labels, dtype=int)})

    tsne_result_df.to_csv(save_path)
    features.train()
    #fig, ax = plt.subplots(1)
    #sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    #lim = (phi_tsne.min()-5, phi_tsne.max()+5)
    #ax.set_xlim(lim)
    #ax.set_ylim(lim)
    #ax.set_aspect('equal')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

# =============== dpp ==================
def map(kernel_matrix, max_length, device, epsilon=1E-10):
    """
    Our proposed fast implementation of the greedy algorithm
    :param kernel_matrix: 2-d array
    :param max_length: positive int
    :param epsilon: small positive scalar
    :return: list
    """
    #kernel_matrix = kernel_matrix.detach().cpu().numpy()
    item_size = kernel_matrix.shape[0]
    cis = torch.zeros((max_length, item_size))#.to(device)
    di2s = torch.diag(kernel_matrix)
    selected_items = list()
    selected_item = torch.argmax(di2s)
    selected_items.append(selected_item.item())
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        try:
            eis = (elements - torch.dot(ci_optimal, cis[:k, :])) / di_optimal
        except:
            eis = (elements - torch.zeros(cis.shape[1]).to(device)) / di_optimal

        cis[k, :] = eis
        di2s -= torch.square(eis)

        di2s[selected_item] = -np.inf
        selected_item = torch.argmax(di2s)
        #if di2s[selected_item] < epsilon:
        #    break
        selected_items.append(selected_item.item())
    return selected_items


# =========== argparse ===========
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def unflatten_dot(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def update(d, u):
    # normal python update method on a dict will overwritten nested siblings that aren't updated
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def split_dict(d):
    to_product = []  # [[('a', 1), ('a', 2)], [('b', 3),], ...]
    for k, v in d.items():
        if isinstance(v, list):
            to_product.append([(k, i) for i in v])
        elif isinstance(v, dict):
            to_product.append([(k, i) for i in split_dict(v)])
        else:
            to_product.append([(k, v)])
    return [dict(l) for l in product(*to_product)]

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# =========== ode for t > 0. ===========
@torch.jit.script
def torch_pade13(A):  # pragma: no cover
    # avoid torch select operation and unpack coefs
    (b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13) = (
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    )

    ident = torch.eye(A.shape[1], dtype=A.dtype, device=A.device)
    A2 = torch.matmul(A, A)
    A4 = torch.matmul(A2, A2)
    A6 = torch.matmul(A4, A2)
    U = torch.matmul(
        A,
        torch.matmul(A6, b13 * A6 + b11 * A4 + b9 * A2)
        + b7 * A6
        + b5 * A4
        + b3 * A2
        + b1 * ident,
    )
    V = (
        torch.matmul(A6, b12 * A6 + b10 * A4 + b8 * A2)
        + b6 * A6
        + b4 * A4
        + b2 * A2
        + b0 * ident
    )
    return U, V


@torch.jit.script
def matrix_2_power(x, p):  # pragma: no cover
    for _ in range(int(p)):
        x = x @ x
    return x


@torch.jit.script
def expm_one(A):  # pragma: no cover
    # no checks, this is private implementation
    # but A should be a matrix
    A_fro = torch.norm(A)

    # Scaling step

    n_squarings = torch.clamp(
        torch.ceil(torch.log(A_fro / 5.371920351148152).div(0.6931471805599453)), min=0
    )
    scaling = 2.0 ** n_squarings
    Ascaled = A / scaling

    # Pade 13 approximation
    U, V = torch_pade13(Ascaled)
    P = U + V
    Q = -U + V

    R = torch.linalg.solve(P, Q)  # solve P = Q*R
    expmA = matrix_2_power(R, n_squarings)
    return expmA

# =============== proto funcs =============================
def pairwise_distances_logits(a, b):
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

# ============= models ==============
class CifarCNN(torch.nn.Module):
    """
    Example of a 4-layer CNN network for FC100/CIFAR-FS.
    """

    def __init__(self, output_size=5, hidden_size=32, layers=4):
        super(CifarCNN, self).__init__()
        self.hidden_size = hidden_size
        features = l2l.vision.models.ConvBase(
            hidden=hidden_size,
            channels=3,
            max_pool=False,
            layers=layers,
            max_pool_factor=0.5,
        )
        self.features = torch.nn.Sequential(
            features,
            l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])),
            l2l.nn.Flatten(),
        )
        self.linear = torch.nn.Linear(self.hidden_size, output_size, bias=True)
        l2l.vision.models.maml_init_(self.linear)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x

# ============== typicality ==================

def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


# =============== misc =====================

def torch_delete(tensor, indices, dim=0):
    mask = torch.ones(tensor.shape[dim], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def compute_dists(M1, M2):
    """
    Computes L2 norm square on gpu
    Assume
    M1: M x D matrix
    M2: N x D matrix
    output: M x N matrix
    """
    M1_norm = (M1**2).sum(1).reshape(-1, 1)

    M2_t = torch.transpose(M2, 0, 1)
    M2_norm = (M2**2).sum(1).reshape(1, -1)
    dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
    return dists

def compute_entropy(batch, ways, train_shots, test_shots):
    data, labels, indices = batch

    adaptation_indices_bool = np.zeros(len(indices), dtype=bool)
    selection = np.arange(ways) * (train_shots + test_shots)
    for offset in range(train_shots):
        adaptation_indices_bool[selection + offset] = True
    evaluation_indices_bool = ~adaptation_indices_bool
    adaptation_labels = labels[adaptation_indices_bool]

    label_counts = torch.zeros(ways)
    for label in adaptation_labels:
        label_counts[label] += 1
    label_counts = label_counts / ways

    entropy = -torch.sum(label_counts * torch.log(label_counts + 1e-12))
    return entropy