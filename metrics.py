import networkx as nx
import numpy as np
from torch_two_sample import MMDStatistic
from scipy.sparse.csgraph import connected_components
import pickle
import scipy.sparse as sp
import torch
from torch.nn import CrossEntropyLoss
import time
import os
import random
from operator import itemgetter
import argparse

def statistics_triangle_count(A_in):
    """
    Compute the triangle count of the input graph

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Triangle count
    """

    triangles = nx.triangles(A_in)
    t = np.sum(list(triangles.values())) / 3
    return int(t)

def statistics_LCC(A_in):
    """
    Compute the size of the largest connected component (LCC)

    Parameters
    ----------
    A_in: sparse matrix or np.array
          The input adjacency matrix.
    Returns
    -------
    Size of LCC

    """

    unique, counts = np.unique(connected_components(A_in)[1], return_counts=True)
    LCC = np.where(connected_components(A_in)[1] == np.argmax(counts))[0]
    return LCC

def metapath_count(A, node_type):
    if dataset == 'ml-100k':
        metapaths = {2: [[0, 1], [0, 3], [1, 2]], 3: [[0, 1, 2]]}
    elif dataset == 'dblp_kdd':
        metapaths = {2: [[0, 1], [0, 2], [1, 3], [1, 1]], 3: [[0, 1, 3], [0, 1, 1], [1, 1, 3], [1, 1, 1]]}
    elif dataset == 'ml-20m' or dataset == 'taobao':
        metapaths = {2: [[0, 1], [1, 2]], 3: [[0, 1, 2]]}
    count = {2: [], 3: []}
    A_A = A @ A
    sum_2 = A.sum()
    sum_3 = A_A.sum()
    for metapath in metapaths[2]:
        cnt = 0
        for i in range(A.shape[0]):
            if node_type[i] == metapath[0]:
                for j in A[i].nonzero()[-1]:
                    if node_type[j] == metapath[1]:
                        cnt += 1
        if cnt != 0: cnt = cnt/sum_2
        count[2].append(cnt)
    count[2].append(1-sum(count[2]))
    for metapath in metapaths[3]:
        cnt = 0
        for i in range(A.shape[0]):
            if node_type[i] == metapath[0]:
                for j in A[i].nonzero()[-1]:
                    if node_type[j] == metapath[1]:
                        for t in A[j].nonzero()[-1]:
                            if node_type[t] == metapath[2]:
                                cnt += 1
        if cnt != 0: cnt = cnt/sum_3
        count[3].append(cnt)
    count[3].append(1-sum(count[3]))
    return count

def evaluation(real_A, syn_A, node_type, timestamp):
    try:
        real_A = sp.csc_matrix(real_A)
    except:
        pass

    try:
        syn_A = sp.csc_matrix(syn_A)
    except:
        pass

    if real_A.shape[0] > syn_A.shape[0]:
        syn_A = sp.csc_matrix((syn_A.data, (syn_A.nonzero()[0], syn_A.nonzero()[1])), shape=real_A.shape)

    lcc_real = statistics_LCC(real_A)
    lcc_syn = statistics_LCC(syn_A)

    if dataset == 'ml-20m' or dataset == 'taobao':
        if not os.path.exists('{}_{}.p'.format(dataset, timestamp)):
            lcc = random.sample(list(lcc_real), 10000)
            node_type = list(itemgetter(*lcc)(node_type))
            pickle.dump(lcc, open('{}_{}.p'.format(dataset, timestamp), 'wb'))
        else:
            lcc = pickle.load(open('{}_{}.p'.format(dataset, timestamp), 'rb'))
            node_type = list(itemgetter(*lcc)(node_type))
    else:
        lcc = range(real_A.shape[0])

    syn_full, real_full = syn_A, real_A
    orig_G_full = nx.from_scipy_sparse_array(real_full)
    syn_G_full = nx.from_scipy_sparse_array(syn_full)

    real_A = real_A[lcc,:][:, lcc]
    syn_A = syn_A[lcc,:][:, lcc]

    orig_G = nx.from_scipy_sparse_array(real_A)
    syn_G = nx.from_scipy_sparse_array(syn_A)

    results = {}
    #Clustering coefficient
    results["coef"] = [nx.average_clustering(orig_G), nx.average_clustering(syn_G)]
    #Triangle count
    results["tc"] =[statistics_triangle_count(orig_G), statistics_triangle_count(syn_G)]
    results["lcc"] = [len(lcc_real), len(lcc_syn)]
    test_degree_sequence = sorted([d for n, d in orig_G.degree()], reverse=True)  # degree sequence
    syn_degree_sequence = sorted([d for n, d in syn_G.degree()], reverse=True)  # degree sequence
    mmd_test = MMDStatistic(len(test_degree_sequence), len(syn_degree_sequence))
    test_degree_sequence = torch.tensor(test_degree_sequence).unsqueeze(-1)
    syn_degree_sequence = torch.tensor(syn_degree_sequence).unsqueeze(-1)
    results["deg_mmd"] = mmd_test(test_degree_sequence, syn_degree_sequence, alphas=[4.], ret_matrix=False)
    orig_G_edge = set(orig_G_full.edges())
    syn_G_edge = set(syn_G_full.edges())
    intersecting_edges = orig_G_edge & syn_G_edge
    if len(syn_G_edge) == 0: results["eo_rate"] = 0
    else: results["eo_rate"] = len(intersecting_edges)/len(syn_G_edge)
    results["meta"] = {}
    real_meta = metapath_count(real_A, node_type)
    syn_meta = metapath_count(syn_A, node_type)
    cross_entropy = CrossEntropyLoss()
    results["meta"][2] = cross_entropy(torch.FloatTensor(syn_meta[2]), torch.FloatTensor(real_meta[2]))
    results["meta"][3] = cross_entropy(torch.FloatTensor(syn_meta[3]), torch.FloatTensor(real_meta[3]))
    return results

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=['ml-100k', 'ml-20m', 'taobao', 'dblp_kdd'],
                    default="ml-100k", help="The choice of dataset.")
args = parser.parse_args()
dataset = args.dataset

if dataset == 'ml-100k': timerange = 8
elif dataset == 'ml-20m': timerange = 22
else: timerange = 11

results = {"coef":[], 'tc':[], "lcc":[], "deg_mmd":[], "eo_rate":[], "meta":{2:[], 3:[]}}

syns = {}
N = -1
with open("./data/"+args.dataset+"/generated/"+args.dataset+"_gen.p", 'rb') as f:
    syn_graph = pickle.load(f)
with open("./data/" + dataset + "/node_index.p", 'rb') as f:
    node_index = pickle.load(f)
with open("./data/" + dataset + "/node_value.p", 'rb') as f:
    node_value = pickle.load(f)
with open("./data/" + dataset + "/node_type.p", 'rb') as f:
    node_type = pickle.load(f)
with open("./data/" + dataset + "/original_node_index.p", 'rb') as f:
    original_node_index = pickle.load(f)
with open("../data/" + dataset + "/node_dict.p", 'rb') as f:
    node_dict = pickle.load(f)
for time in range(0, timerange):
    syns[time] = [[], []]
for i, j in zip(syn_graph.nonzero()[0], syn_graph.nonzero()[1]):
    node1 = node_value[i]
    node2 = node_value[j]
    type1 = node_type[node_index[node1]]
    type2 = node_type[node_index[node2]]
    node1, time1 = node1.split('_')
    node2, time2 = node2.split('_')
    node1_idx = original_node_index[node1]
    node2_idx = original_node_index[node2]
    syns[int(time1)][0].append(node1_idx)
    syns[int(time1)][1].append(node2_idx)
    syns[int(time2)][0].append(node1_idx)
    syns[int(time2)][1].append(node2_idx)
    N = max(N, node1_idx, node2_idx)

new_node_type = {}

edges = {}
nodes = set()
for node in node_dict:
    node1, time1 = node.split('_')
    time1 = int(time1)
    if time1 not in edges:
        edges[time1] = set()
    type1 = node_type[node_index[node]]
    for (node2, time2) in node_dict[node]:
        assert time1 == time2
        node_ = '{}_{}'.format(node2, time2)
        type2 = node_type[node_index[node_]]
        node1_idx = original_node_index[node1]
        node2_idx = original_node_index[node2]
        if not (node1_idx, node2_idx) in edges[time1]:
            edges[time1].add((node1_idx, node2_idx))
            if node1_idx not in nodes:
                new_node_type[node1_idx] = type1
            if node2_idx not in nodes:
                new_node_type[node2_idx] = type2

node_type = new_node_type

syn_0 = sp.coo_matrix((np.ones(len(syns[0][0])), (syns[0][0], syns[0][1])), (N+1,N+1))

for timestamp in range(1, timerange):
    syn = sp.coo_matrix((np.ones(len(syns[timestamp][0])), (syns[timestamp][0], syns[timestamp][1])), (N+1,N+1))
    syn = syn + syn_0
    edge_time = np.array(list(edges[timestamp]))
    real = sp.csc_matrix((np.ones(edge_time.shape[1]), (edge_time[0,:], edge_time[1,:])), (N+1,N+1))
    result = evaluation(real, syn, node_type, timestamp)
    for metric in results:
        if metric == "meta":
            results[metric][2].append(result[metric][2])
            results[metric][3].append(result[metric][3])
        else:
            results[metric].append(result[metric])

for metric in ["coef", "tc", "lcc"]:
    result = results[metric]
    mmd_test = MMDStatistic(len(result), len(result))
    result = np.array(result).transpose()
    real_ = torch.tensor(result[0, :]).unsqueeze(-1)
    syn_ = torch.tensor(result[1, :]).unsqueeze(-1)
    results[metric] = mmd_test(real_, syn_, alphas=[4.], ret_matrix=False).item()

for metric in ['deg_mmd', 'eo_rate']:
    results[metric] = np.array(results[metric]).mean()

for length in results["meta"]:
    results["meta"][length] = np.array(results["meta"][length]).mean()

print(results, flush=True)

