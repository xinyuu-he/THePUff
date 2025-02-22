import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable
import os
import pickle
import random
from collections import Counter

def symmetric(directed_adjacency, clip_to_one=True):
    A_symmetric = directed_adjacency + directed_adjacency.T
    if clip_to_one:
        A_symmetric[A_symmetric > 1] = 1
    return A_symmetric


def random_walk(node_dict, node_index, degrees, num_of_walks, length_of_walks, output_file):
    with open(output_file, 'wb') as f:
        sequences = []
        for node in node_dict.keys():
            for i in range(min(num_of_walks, degrees[node])):
                v = node
                sequence = [node_index[node]]
                if degrees[v] == 0:
                    continue
                for j in range(length_of_walks):
                    if degrees[v] == 0:
                        break
                    p = 1/degrees[v]
                    r = random.random()
                    k = int(r/p)
                    v_ = '{}_{}'.format(node_dict[v][k][0], node_dict[v][k][1])
                    sequence.append(node_index[v_])
                    v = v_
                sequences.append(sequence)
        pickle.dump(sequences, f)


def one_hot_encoder(data, max_value):
    shape = (data.size, max_value)
    one_hot = np.zeros(shape)
    rows = np.arange(data.size)
    one_hot[rows, data] = 1

    return one_hot


def pad_along_axis(array, target_length, axis):
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array

    padding = np.zeros([pad_size, array.shape[1]])
    for i in range(len(padding)):
        padding[i][-1] = 1

    return np.concatenate([array, padding], axis=axis)


def pad_list(list, target_leagth):
    pad_size = target_leagth - len(list)
    if pad_size <= 0:
        return list
    
    for i in range(pad_size):
        list.append(-1)

    return list


def get_walk_data(path, N, node_type, args):
    walk_data = []
    with open(path, 'rb') as f:
        paths = pickle.load(f)
    for path in paths:
        temp_type = one_hot_encoder(np.array([node_type[i] for i in path]), args.node_classes+1)
        temp_idx = path
        # Padding random walks to the max length
        if temp_type.shape[0] < args.max_path_len:
            temp_type = pad_along_axis(temp_type, args.max_path_len, axis=0).astype(np.float32)
            temp_idx = pad_list(temp_idx, args.max_path_len)
        walk_data.append((temp_type, temp_idx))
    return walk_data


def make_noise(shape, type="Gaussian"):
    """
    Generate random noise.
    Parameters
    ----------
    shape: List or tuple indicating the shape of the noise
    type: str, "Gaussian" or "Uniform", default: "Gaussian".
    Returns
    -------
    noise tensor
    """

    if type == "Gaussian":
        noise = Variable(torch.randn(shape))
    elif type == 'Uniform':
        noise = Variable(torch.randn(shape).uniform_(-1, 1))
    else:
        print("ERROR: Noise type {} not supported".format(type))
    return noise


class HTNDataset(torch.utils.data.Dataset):
    def __init__(self, data, embs, level_embs):
        super().__init__()
        self.data = data
        self.embs = embs
        self.level_embs = level_embs

    def __getitem__(self, item):
        types, idx = self.data[item]
        return types, self.embs[idx], self.level_embs[idx]

    def __len__(self):
        return len(self.data)


def preprocess_edgelist(data_directory, node_type_dict, directed):
    node_dict = dict()
    node_index = dict()
    node_value = dict()
    node_type = dict()
    original_node_index = dict()
    count = 0
    original_node_count = 0
    min_time_stamp = np.inf
    max_time_stamp = 0
    # find the minimal timestamp
    with open(data_directory, 'r') as f:
        for line in f:
            line = line.split()
            line[4] = int(line[4])
            if line[4] < min_time_stamp:
                min_time_stamp = line[4]
            if line[4] > max_time_stamp:
                max_time_stamp = line[4]
    interval = max_time_stamp - min_time_stamp + 1
    time_slice = 1#int((max_time_stamp - min_time_stamp + 1) / interval)
    assert (time_slice != 0), "Please check if interval is correctly set!"
    with open(data_directory, 'r') as f:
        #with open(data_directory[:-4] + '_new' + data_directory[-4:], 'w') as f_out:
        for line in f:
            nodes = line.split()
            nodes[4] = int(nodes[4])
            nodes[4] = int((nodes[4] - min_time_stamp) / time_slice)
            # index dictionary by which we could map newly-created nodes back to original graph
            if nodes[0] not in original_node_index:
                original_node_index[nodes[0]] = original_node_count
                original_node_count += 1
            if nodes[1] not in original_node_index:
                original_node_index[nodes[1]] = original_node_count
                original_node_count += 1
            # map original nodes to newly-created nodes to encode the time info
            if '{}_{}'.format(nodes[0], nodes[4]) not in node_index:
                node_value[count] = '{}_{}'.format(nodes[0], nodes[4])
                node_index['{}_{}'.format(nodes[0], nodes[4])] = count
                node_dict['{}_{}'.format(nodes[0], nodes[4])] = []
                node_type[count] = node_type_dict[nodes[2]]
                count += 1
            # if the second node does not exist in the dictionary, then add it to the dictionary
            if '{}_{}'.format(nodes[1], nodes[4]) not in node_index:
                node_value[count] = '{}_{}'.format(nodes[1], nodes[4])
                node_index['{}_{}'.format(nodes[1], nodes[4])] = count
                node_dict['{}_{}'.format(nodes[1], nodes[4])] = []
                node_type[count] = node_type_dict[nodes[3]]
                count += 1
            if (nodes[1], nodes[4]) not in node_dict['{}_{}'.format(nodes[0], nodes[4])]:
                node_dict['{}_{}'.format(nodes[0], nodes[4])].append((nodes[1], nodes[4]))
            if not directed:
                if (nodes[0], nodes[4]) not in node_dict['{}_{}'.format(nodes[1], nodes[4])]:
                    node_dict['{}_{}'.format(nodes[1], nodes[4])].append((nodes[0], nodes[4]))
            if max_time_stamp < nodes[4]:
                max_time_stamp = nodes[4]
    node_level = dict()
    for i in range(len(node_index)):
        node_level[i] = [i]
    for node in node_dict.keys():
        nodes = node.split('_')
        nodes[1] = int(nodes[1])
        # expand the neighbour of this node if the time stamp is within a certain range
        for i in range(interval):
            if i != nodes[1] and '{}_{}'.format(nodes[0], i) in node_dict:
                node_level[node_index[node]].append(node_index['{}_{}'.format(nodes[0], i)])
    print("Finish Remapping nodes! Total number of nodes = {}".format(count))
    return node_dict, node_index, original_node_index, node_value, node_level, node_type


def preprocess_dp_edgelist(data_directory, node_type_dict, node_index, node_value, node_type):
    count = len(node_value)
    node_dict_dp = dict()
    min_time_stamp = np.inf
    max_time_stamp = 0
    with open(data_directory, 'r') as f:
        for line in f:
            line = line.split()
            line[4] = int(line[4])
            if line[4] < min_time_stamp:
                min_time_stamp = line[4]
            if line[4] > max_time_stamp:
                max_time_stamp = line[4]
    interval = max_time_stamp - min_time_stamp + 1
    with open(data_directory, 'r') as f:
        #with open(data_directory[:-4] + '_new' + data_directory[-4:], 'w') as f_out:
        for line in f:
            nodes = line.split()
            nodes[4] = int(nodes[4])
            if '{}_{}'.format(nodes[0], nodes[4]) not in node_index:
                node_value[count] = '{}_{}'.format(nodes[0], nodes[4])
                node_index['{}_{}'.format(nodes[0], nodes[4])] = count
                node_type[count] = node_type_dict[nodes[2]]
                count += 1
            if '{}_{}'.format(nodes[1], nodes[4]) not in node_index:
                node_value[count] = '{}_{}'.format(nodes[1], nodes[4])
                node_index['{}_{}'.format(nodes[1], nodes[4])] = count
                node_type[count] = node_type_dict[nodes[3]]
                count += 1
            if '{}_{}'.format(nodes[0], nodes[4]) not in node_dict_dp:
                node_dict_dp['{}_{}'.format(nodes[0], nodes[4])] = []
            if '{}_{}'.format(nodes[1], nodes[4]) not in node_dict_dp:
                node_dict_dp['{}_{}'.format(nodes[1], nodes[4])] = []
            if (nodes[1], nodes[4]) not in node_dict_dp['{}_{}'.format(nodes[0], nodes[4])]:
                node_dict_dp['{}_{}'.format(nodes[0], nodes[4])].append((nodes[1], nodes[4]))
    node_level_dp = dict()
    for i in range(len(node_index)):
        node_level_dp[i] = [i]
    for node in node_dict_dp.keys():
        nodes = node.split('_')
        nodes[1] = int(nodes[1])
        # expand the neighbour of this node if the time stamp is within a certain range
        for i in range(interval):
            if i != nodes[1] and '{}_{}'.format(nodes[0], i) in node_dict_dp:
                node_level_dp[node_index[node]].append(node_index['{}_{}'.format(nodes[0], i)])
    return node_dict_dp, node_index, node_value, node_level_dp, node_type

def get_embeddings(dataset, node_dict, node_type, node_index, output_path):
    if dataset == 'dblp_kdd':
        with open('./data/'+dataset+'/edgelist.txt', 'w') as fl, open('./data/'+dataset+'/edgetype.txt', 'w') as ft:
            for node0 in node_dict.keys():
                idx0 = node_index[node0]
                type0 = node_type[idx0]
                for (node1, time) in node_dict[node0]:
                    idx1 = node_index['{}_{}'.format(node1, time)]
                    type1 = node_type[idx1]
                    if type0 == 0 and type1 == 1:
                        fl.write('a' + str(idx0) + ' p' + str(idx1) + '\n')
                    elif type0 == 1 and type1 == 1:
                        fl.write('p' + str(idx0) + ' p' + str(idx1) + '\n')
                    elif type0 == 0 and type1 == 2:
                        fl.write('a' + str(idx0) + ' o' + str(idx1) + '\n')
                    elif type0 == 1 and type1 == 3:
                        fl.write('p' + str(idx0) + ' f' + str(idx1) + '\n')
                    else:
                        print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                        assert False, 'unexpected edge type'
            ft.write('a : p' + '\n')
            ft.write('p : p' + '\n')
            ft.write('a : o' + '\n')
            ft.write('p : f' + '\n')
    elif dataset == 'ml-100k':
        with open('./data/'+dataset+'/edgelist.txt', 'w') as fl, open('./data/'+dataset+'/edgetype.txt', 'w') as ft:
            for node0 in node_dict.keys():
                idx0 = node_index[node0]
                type0 = node_type[idx0]
                for (node1, time) in node_dict[node0]:
                    idx1 = node_index['{}_{}'.format(node1, time)]
                    type1 = node_type[idx1]
                    if type0 == 0 and type1 == 1:
                        fl.write('u' + str(idx0) + ' m' + str(idx1) + '\n')
                    elif type0 ==0  and type1 == 3:
                        fl.write('u' + str(idx0) + ' o' + str(idx1) + '\n')
                    elif type0 == 1 and type1 == 2:
                        fl.write('m' + str(idx0) + ' g' + str(idx1) + '\n')
                    else:
                        print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                        assert False, 'unexpected edge type'
            ft.write('u : m' + '\n')
            ft.write('u : o' + '\n')
            ft.write('m : g' + '\n')
    elif dataset == 'ml-20m':
        with open('./data/'+dataset+'/edgelist.txt', 'w') as fl, open('./data/'+dataset+'/edgetype.txt', 'w') as ft:
            for node0 in node_dict.keys():
                idx0 = node_index[node0]
                type0 = node_type[idx0]
                for (node1, time) in node_dict[node0]:
                    idx1 = node_index['{}_{}'.format(node1, time)]
                    type1 = node_type[idx1]
                    if type0 == 0 and type1 == 1:
                        fl.write('u' + str(idx0) + ' m' + str(idx1) + '\n')
                    elif type0 == 1 and type1 == 2:
                        fl.write('m' + str(idx0) + ' g' + str(idx1) + '\n')
                    else:
                        print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                        assert False, 'unexpected edge type'
            ft.write('u : m' + '\n')
            ft.write('m : g' + '\n')
    elif dataset == 'taobao':
        with open('./data/'+dataset+'/edgelist.txt', 'w') as fl, open('./data/'+dataset+'/edgetype.txt', 'w') as ft:
            for node0 in node_dict.keys():
                idx0 = node_index[node0]
                type0 = node_type[idx0]
                for (node1, time) in node_dict[node0]:
                    idx1 = node_index['{}_{}'.format(node1, time)]
                    type1 = node_type[idx1]
                    if type0 == 0 and type1 == 1:
                        fl.write('u' + str(idx0) + ' i' + str(idx1) + '\n')
                    elif type0 == 1 and type1 == 2:
                        fl.write('i' + str(idx0) + ' c' + str(idx1) + '\n')
                    else:
                        print(node0, '{}_{}'.format(node1, time), idx0, idx1, type0, type1)
                        assert False, 'unexpected edge type'
            ft.write('u : i' + '\n')
            ft.write('i : c' + '\n')
    else:
        assert False, 'unexpected dataset'
    os.system('python3 ./JUST/src/main.py --input ./data/'+dataset+'/edgelist.txt --node_types ./data/'+dataset+'/edgetype.txt'\
                   +' --dimensions 128 --walk_length 100 --num_walks 10 --window-size 10 --alpha 0.5'+ \
                   ' --output '+output_path)


def get_node_level_embeddings(path, node_level, node_value, embs):
    role_level_emb = dict()
    for idx in range(len(embs)):
        role_level_emb[node_value[idx]] = embs[idx]
    with open(path, 'w') as f_emb:
        f_emb.write('{} {}\n'.format(len(node_level), len(embs[0])))
        for i in range(len(node_level)):
            sum = role_level_emb[node_value[node_level[i][0]]]
            for j in range(1, len(node_level[i])):
                sum = sum + role_level_emb[node_value[node_level[i][j]]]
            f_emb.write('{} '.format(i) + ' '.join(map(str, sum/len(node_level[i]))) + '\n')


def load_level_embeddings(N, real_level_path, dp_level_path):
    real_level_emb = np.zeros((N+1, 128))
    with open(real_level_path, 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            real_level_emb[int(line[0])] = np.array(list(map(float, line[1:])))
    dp_level_emb = np.zeros((N+1, 128))
    with open(dp_level_path, 'r') as f_emb:
        next(f_emb)
        for line in f_emb:
            line = line.split()
            dp_level_emb[int(line[0])] = np.array(list(map(float, line[1:])))
    return real_level_emb.astype(np.float32), dp_level_emb.astype(np.float32)


def get_classified(node_type, node_embs, device):
    node_type_classfied = dict()
    for i in node_type:
        if node_type[i] not in node_type_classfied:
            node_type_classfied[node_type[i]] = []
        node_type_classfied[node_type[i]].append(i)
    node_embs_classified = dict()
    for i in node_type_classfied.keys():
        temp = []
        for j in node_type_classfied[i]:
            temp.append(torch.tensor(node_embs[j]).unsqueeze(0))
        node_embs_classified[i] = torch.cat(temp, 0).to(device)
    return node_type_classfied, node_embs_classified


def heterogeneous_graph_assemble(scores, n_edges, meta_path_freq, node_type):
    """ Assemble a heterogeneous graph based on the meta-path ratio"""
    if len(scores.nonzero()[0]) < n_edges:
        return symmetric(scores) > 0

    target_g = sp.csr_matrix(scores.shape)  # initialize target graph
    scores_int = scores.toarray().copy()  # internal copy of the scores matrix
    scores_int[np.diag_indices_from(scores_int)] = 0  # set diagonal to zero
    degrees_int = scores_int.sum(0)  # The row sum over the scores.
    N = scores.shape[0]

    node_type_degs = []
    for i in range(len(node_type)):
        node_type_degs.append(dict(zip(node_type[i], [degrees_int[i] for i in node_type[i]])))

    count = 0
    while target_g.sum() <= n_edges:
        sampled_metapath = sample_from_dict(meta_path_freq, 1)[0]
        actual_path = []
        for i in range(len(sampled_metapath) - 1):
            if i == 0:
                n = sample_from_dict(node_type_degs[sampled_metapath[i]], 1)[0]
                actual_path.append(n)
            else:
                n = actual_path[-1]
            row = scores_int[n, :].copy()
            next_node_type_degs = list(node_type_degs[sampled_metapath[i + 1]].keys())
            row[array_subtraction(range(N), next_node_type_degs)] = 0
            probs = row / row.sum()
            try:
                target = np.random.choice(N, p=probs)
            except ValueError:
                print(n)
            target_g[n, target] = 1
            target_g[target, n] = 1

            actual_path.append(target)
            count += 1
        if count % 10000 == 0:
            print("Generating {} of {} edges...".format(target_g.sum(), n_edges))

    target_g = symmetric(target_g)
    return target_g

def sample_from_dict(d, sample):
    """ Random sample from a dict """
    key = random.choices(list(d.keys()), list(d.values()), k=sample)
    return key

def array_subtraction(x, y):
    return np.array(list(set(x) - set(y)))

def frequent_meta_path_pattern(smpls_type):
    bb = []
    for i in smpls_type:
        bb.append(tuple(i))
    count = Counter(bb)

    return count.most_common()
    

def meta_path_frequency(smpls_type_2, smpls_type_3, smpls_type_4, smpls_type_5):
    len_2 = frequent_meta_path_pattern(smpls_type_2)
    len_3 = frequent_meta_path_pattern(smpls_type_3)
    len_4 = frequent_meta_path_pattern(smpls_type_4)
    len_5 = frequent_meta_path_pattern(smpls_type_5)
    len_2, len_3, len_4, len_5 = len_2[:len(len_2)//2], len_3[:len(len_3)//2], len_4[:len(len_4)//2], len_5[:len(len_5)//2]
    
    return dict(len_2 + len_3 + len_5)

def delete_from_tail(array, target):
    updated_array = []
    for i in array:
        temp = i.copy()
        while len(temp) > 0 and temp[-1] == target:
            temp = temp[:-1]
        if len(temp) > 0:
            updated_array.append(temp)
    
    return updated_array

def score_matrix_from_random_walks(random_walks, N, symmetric=True):
    """
    Compute the transition scores, i.e. how often a transition occurs, for all node pairs from
    the random walks provided.
    Parameters
    ----------
    random_walks: np.array of shape (n_walks, rw_len, N)
                  The input random walks to count the transitions in.
    N: int
       The number of nodes
    symmetric: bool, default: True
               Whether to symmetrize the resulting scores matrix.

    Returns
    -------
    scores_matrix: sparse matrix, shape (N, N)
                   Matrix whose entries (i,j) correspond to the number of times a transition from node i to j was
                   observed in the input random walks.

    """

    random_walks = np.array(random_walks)
    bigrams = np.array(list(zip(random_walks[:, :-1], random_walks[:, 1:])))
    bigrams = np.transpose(bigrams, [0, 2, 1])
    bigrams = bigrams.reshape([-1, 2])
    if symmetric:
        bigrams = np.row_stack((bigrams, bigrams[:, ::-1]))

    mat = sp.coo_matrix((np.ones(bigrams.shape[0]), (bigrams[:, 0], bigrams[:, 1])),
                        shape=[N, N])
    return mat

def save_graph(path, graph, node_value, node_type):
    with open(path+'.txt', 'w') as f:
        for i, j in zip(graph.nonzero()[0], graph.nonzero()[1]):
            node0 = node_value[i]
            node1 = node_value[j]
            time0 = int(node0.split("_")[1])
            time1 = int(node1.split("_")[1])
            f.write(node0+" "+node1+" "+node0[0]+" "+node1[0]+" "+str(max(time0, time1))+"\n")
    with open(path+'.p', 'wb') as f:
        pickle.dump(graph, f)

