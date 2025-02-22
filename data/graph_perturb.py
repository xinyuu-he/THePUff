import random
import numpy as np
import math
from scipy.sparse import csr_array

dataset = 'ml-100k'

node_value = {}
node_index = {}
edges = {}
counts = {}
types = set()
with open('./'+dataset+'/'+dataset+'.txt', 'r') as f:
    count = 0
    for line in f:
        line = line.split()
        timestamp = int(line[4])
        node0 = line[0]
        node1 = line[1]
        type0 = line[2]
        type1 = line[3]
        type = '{}_{}'.format(type0, type1)
        types.add(type)
        if type0 not in counts:
            counts[type0] = 0
            node_index[type0] = {}
            node_value[type0] = {}
        if type1 not in counts:
            counts[type1] = 0
            node_index[type1] = {}
            node_value[type1] = {}
        if node0 not in node_index[type0]:
            node_index[type0][node0] = counts[type0]
            node_value[type0][counts[type0]] = node0
            counts[type0] += 1
        if node1 not in node_index[type1]:
            node_index[type1][node1] = counts[type1]
            node_value[type1][counts[type1]] = node1
            counts[type1] += 1
        if timestamp not in edges:
            edges[timestamp] = {}
        if type not in edges[timestamp]:
            edges[timestamp][type] = [[node_index[type0][node0], node_index[type1][node1]]]
        else:
            edges[timestamp][type].append([node_index[type0][node0], node_index[type1][node1]])

eps_del = 8
p_del = 1/math.exp(eps_del)
p_add = {}

n2 = 0
print(types)
for type in types:
    type0, type1 = type.split("_")
    n2 += counts[type0] * counts[type1]

for time in edges:
    m_t = 0
    for type in edges[time]: m_t += len(edges[time][type])
    p_add[time] = p_del * m_t / (n2 - m_t)

print(p_del, p_add)

del_edgelist = {}

cnt_del = 0
for time in edges:
    del_edgelist[time] = {}
    print(time, flush=True)
    for type in edges[time]:
        del_edgelist[time][type] = []
        edgelist = edges[time][type]
        for edge in edgelist:
            p = random.random()
            if p <= p_del:
                del_edgelist[time][type].append(edge)
                cnt_del += 1
print('deleted', cnt_del, flush=True)

cnt_add = 0
for time in edges:
    print(time, flush=True)
    for type in edges[time]:
        type0, type1 = type.split("_")
        edgelist = edges[time][type]
        num_add = int(round(counts[type0]*counts[type1]*p_add[time]))
        cnt_add += num_add
        print(num_add, flush=True)
        idxs = random.sample(range((counts[type0]*counts[type1])), num_add)
        for idx in idxs:
            node0 = idx / counts[type1]
            node1 = idx % counts[type1]
            edges[time][type].append([node0, node1])

for time in edges:
    for type in edges[time]:
        type0, type1 = type.split("_")
        edgelist = edges[time][type]
        inds = np.array(edgelist)
        g = csr_array((np.ones(inds.shape[0]), (inds[:, 0], inds[:, 1])), shape = (counts[type0], counts[type1]))
        for edge in del_edgelist[time][type]:
            g[edge[0], edge[1]] = 0
        edges[time][type] = g.nonzero()


with open('./'+dataset+'/'+dataset+'_dp.txt', 'w') as f:
    for time in edges:
        for type in edges[time]:
            type0, type1 = type.split("_")
            for (node0, node1) in zip(edges[time][type][0], edges[time][type][1]):
                f.write(node_value[type0][node0]+" "+node_value[type1][node1]+" "+type0+" "+type1+" "+str(time)+"\n")

print('added %d and deleted %d edges' % (cnt_add, cnt_del))