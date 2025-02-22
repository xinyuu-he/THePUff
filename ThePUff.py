import argparse
import os
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
import generator
import discriminator
from gensim.models import KeyedVectors
import time

import utils

st = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64,
                    help="Batch size.")
parser.add_argument("--gpu-num", type=int, default=0,
                    help="GPU id.")
parser.add_argument('--W-down-generator-size', type=int, default=128,
                    help="The dimension of lstm embeddings in generator.")
parser.add_argument("--noise-dim", type=int, default=64,
                    help="The dim of the random noise that is used as input.")
parser.add_argument("--noise-type", choices=["Gaussian", "Uniform"], default="Uniform",
                    help="The noise type to feed into the generator.")
parser.add_argument("--hidden-units", type=int, default=128,
                    help="The dimension of the hidden unit in lstm cells in generator.")
parser.add_argument("--num-G-layer", type=int, default=5,
                    help="The number of layers in lstm cells of Generator.")
parser.add_argument("--max-path-len", type=int, default=5,
                    help="The maximum meta path length")
parser.add_argument("--h", type=int, default=4,
                    help="Number of heads in attention layer in discriminator.")
parser.add_argument("--N", type=int, default=1,
                    help="Number of layers in encoder layer of discriminator.")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="Dropout rate of discriminator.")
parser.add_argument("--d-model", type=int, default=128,
                    help="The dimension of the hidden unit in attention layer of discriminator.")
parser.add_argument("--d-ff", type=int, default=128,
                    help="The dimension of the hidden unit in feed forward layer of discriminator.")
parser.add_argument("--lr_gen", type=float, default=1e-4,
                    help="Learning Rate of generator")
parser.add_argument("--lr_dis1", type=float, default=1e-3,
                    help="Learning Rate of descriminator1")
parser.add_argument("--lr_dis2", type=float, default=1e-4,
                    help="Learning Rate of descriminator2")
parser.add_argument("--n-critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--n-epochs", type=int, default=5,
                    help="The number of epochs.")
parser.add_argument("--n-epochs-pre", type=int, default=10,
                    help="The number of pretraining epochs.")
parser.add_argument("--dataset", choices=['ml-100k', 'ml-20m', 'taobao', 'dblp_kdd'],
                    default="ml-100k", help="The choice of dataset.")
parser.add_argument("--load-d2", type=bool, default=False,
                    help="Whether to load pretrained d2.")
parser.add_argument("--load-model", type=bool, default=False,
                    help="Whether to load model.")

args = parser.parse_args()
device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

print(args)

if args.dataset == 'dblp_kdd':
    args.node_classes = 4
    node_type_dict = {'a': 0, 'p': 1, 'o': 2, 'f': 3}
if args.dataset == 'ml-100k':
    args.node_classes = 4
    node_type_dict = {'u': 0, 'm': 1, 'g': 2, 'o': 3}
if args.dataset == 'ml-20m':
    args.node_classes = 3
    node_type_dict = {'u': 0, 'm': 1, 'g': 2}
if args.dataset == 'taobao':
    args.node_classes = 3
    node_type_dict = {'u': 0, 'i': 1, 'c': 2}
else:
    assert False, "Unsupported dataset!"

'''
node_index: node_time->new_node
node_value: new_node->node_time
node_dict: edgelist node_time->list of (node, time)
'''

print("Preprocessing original graph...")
#data reading
real_data_path = "./data/"+args.dataset+"/"+args.dataset+".txt"
data_dir = "./data/"+args.dataset+"/"
if not os.path.exists(data_dir+"node_dict.p") \
    or not os.path.exists(data_dir+"node_index.p") \
    or not os.path.exists(data_dir+"original_node_index.p") \
    or not os.path.exists(data_dir+"node_value.p") \
    or not os.path.exists(data_dir+"node_level.p") \
    or not os.path.exists(data_dir+"node_type.p"):
    node_dict, node_index, original_node_index, node_value, node_level, node_type = \
        utils.preprocess_edgelist(real_data_path, node_type_dict, True)
    with open("./data/"+args.dataset+"/node_dict.p", 'wb') as f:
        pickle.dump(node_dict, f)
    with open("./data/"+args.dataset+"/node_index.p", 'wb') as f:
        pickle.dump(node_index, f)
    with open("./data/"+args.dataset+"/original_node_index.p", 'wb') as f:
        pickle.dump(original_node_index, f)
    with open("./data/"+args.dataset+"/node_value.p", 'wb') as f:
        pickle.dump(node_value, f)
    with open("./data/"+args.dataset+"/node_level.p", 'wb') as f:
        pickle.dump(node_level, f)
    with open("./data/"+args.dataset+"/node_type.p", 'wb') as f:
        pickle.dump(node_type, f)
else:
    with open("./data/"+args.dataset+"/node_dict.p", 'rb') as f:
        node_dict = pickle.load(f)
    with open("./data/"+args.dataset+"/node_index.p", 'rb') as f:
        node_index = pickle.load(f)
    with open("./data/"+args.dataset+"/original_node_index.p", 'rb') as f:
        original_node_index = pickle.load(f)
    with open("./data/"+args.dataset+"/node_value.p", 'rb') as f:
        node_value = pickle.load(f)
    with open("./data/"+args.dataset+"/node_level.p", 'rb') as f:
        node_level = pickle.load(f)
    with open("./data/"+args.dataset+"/node_type.p", 'rb') as f:
        node_type = pickle.load(f)

print("Preprocessing dp graph...")
dp_data_path = "./data/"+args.dataset+"/"+args.dataset+"_dp.txt"
if not os.path.exists(data_dir+"node_dict_dp.p") \
    or not os.path.exists(data_dir+"node_level_dp.p"):
    node_dict_dp, node_index, node_value, node_level_dp, node_type = \
        utils.preprocess_dp_edgelist(real_data_path, node_type_dict, node_index, node_value, node_type)
    with open("./data/"+args.dataset+"/node_dict_dp.p", 'wb') as f:
        pickle.dump(node_dict_dp, f)
    with open("./data/"+args.dataset+"/node_index.p", 'wb') as f:
        pickle.dump(node_index, f)
    with open("./data/"+args.dataset+"/node_value.p", 'wb') as f:
        pickle.dump(node_value, f)
    with open("./data/"+args.dataset+"/node_level_dp.p", 'wb') as f:
        pickle.dump(node_level_dp, f)
    with open("./data/"+args.dataset+"/node_type.p", 'wb') as f:
        pickle.dump(node_type, f)
else:
    with open("./data/"+args.dataset+"/node_dict_dp.p", 'rb') as f:
        node_dict_dp = pickle.load(f)
    with open("./data/"+args.dataset+"/node_index.p", 'rb') as f:
        node_index = pickle.load(f)
    with open("./data/"+args.dataset+"/node_value.p", 'rb') as f:
        node_value = pickle.load(f)
    with open("./data/"+args.dataset+"/node_level_dp.p", 'rb') as f:
        node_level_dp = pickle.load(f)
    with open("./data/"+args.dataset+"/node_type.p", 'rb') as f:
        node_type = pickle.load(f)

N = len(node_index)
args.N_ = N
M = 0 
for node in node_dict_dp.keys():
    M += len(node_dict_dp[node])

print("Learning temporal node embeddings...")
real_emb_path = "./data/"+args.dataset+'/'+args.dataset+"_emb.bin"
if not os.path.exists(real_emb_path):
    utils.get_embeddings(args.dataset, node_dict, node_type, node_index, real_emb_path)

dp_emb_path = "./data/"+args.dataset+'/'+args.dataset+"_emb_dp.bin"
if not os.path.exists(dp_emb_path):
    utils.get_embeddings(args.dataset, node_dict_dp, node_type, node_index, dp_emb_path)

node_embs = KeyedVectors.load_word2vec_format(real_emb_path, binary=True)
node_embs_dp = KeyedVectors.load_word2vec_format(dp_emb_path, binary=True)

print("Getting node level embeddings...")
real_level_path = "./data/"+args.dataset+"/"+args.dataset+"_level"
if not os.path.exists(real_level_path):
    utils.get_node_level_embeddings(real_level_path, node_level, node_value, node_embs)

dp_level_path = "./data/"+args.dataset+"/"+args.dataset+"_level_dp"
if not os.path.exists(dp_level_path):
    utils.get_node_level_embeddings(dp_level_path, node_level_dp, node_value, node_embs_dp)

print("Loading embeddings...")
real_level_embs, dp_level_embs = utils.load_level_embeddings(N, real_level_path, dp_level_path)
node_type_classified, node_embs_classified = utils.get_classified(node_type, node_embs_dp, device)

node_embs.add_vector(-1, np.zeros(128))
node_embs_dp.add_vector(-1, np.zeros(128))

print("Initialing model...")
#initializing generator and discriminators
generator = generator.Generator(node_type_classified, N, node_embs_classified, device, args).to(device)
# D1: distinguish dp graph and generated graph
# D2: distinguish original graph and dp graph
discriminator1 = discriminator.Discriminator(args).to(device)
discriminator2 = discriminator.Discriminator(args).to(device)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=args.lr_gen)
optimizer_D1 = torch.optim.SGD(discriminator1.parameters(), lr=args.lr_dis1, momentum=0.7)
optimizer_D2 = torch.optim.SGD(discriminator2.parameters(), lr=args.lr_dis2, momentum=0.7)

print("Extracting random walks...")
#walk sampling
degrees = {}
for (node, edges) in node_dict.items():
    degrees[node] = len(edges)
real_path_path = "./data/"+args.dataset+"/"+args.dataset+"_p.pickle"
if not os.path.exists(real_path_path):
    utils.random_walk(node_dict, node_index, degrees, 10, args.max_path_len, real_path_path)

degrees_dp = {}
for (node, edges) in node_dict_dp.items():
    degrees_dp[node] = len(edges)
dp_path_path = "./data/"+args.dataset+"/"+args.dataset+"_dp_p.pickle"
if not os.path.exists(dp_path_path):
    utils.random_walk(node_dict_dp, node_index, degrees_dp, 10, args.max_path_len, dp_path_path)

print("Constructing data loaders...")
real_walk_data = utils.get_walk_data(real_path_path, N, node_type, args)
dp_walk_data = utils.get_walk_data(dp_path_path, N, node_type, args)

real_dataloader = DataLoader(utils.HTNDataset(real_walk_data, node_embs, real_level_embs), shuffle=True
                             , batch_size=args.batch_size
                             , num_workers=0, drop_last=True)
dp_dataloader = DataLoader(utils.HTNDataset(dp_walk_data, node_embs_dp, dp_level_embs), shuffle=True
                           , batch_size=args.batch_size
                           , num_workers=0, drop_last=True)
print('All Data Processed!', flush=True)

def pretrain():
    #pretrain D2
    print('Starts pretraining...')
    for epoch in range(args.n_epochs_pre):
        for i, (real_type, real_embs, real_node_embs) in enumerate(real_dataloader):
            initial_noise = utils.make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
            discriminator2.train()
            optimizer_D2.zero_grad()
            fake_result = generator(initial_noise)
            fake_type = fake_result[0].detach()
            fake_node = fake_result[1].detach()
            fake_idxs = fake_result[2]
            fake_embs = []
            for idxs in fake_idxs:
                fake_embs.append(node_embs[idxs])
            fake_embs = torch.from_numpy(np.stack(fake_embs, axis=0)).to(device)
            fake_node_embs = []
            for idxs in fake_idxs:
                fake_node_embs.append(real_level_embs[idxs])
            fake_node_embs = torch.from_numpy(np.stack(fake_node_embs, axis=0)).to(device)
            loss_D2 = torch.mean(discriminator2(fake_embs.transpose(0, 1), fake_node_embs.transpose(0, 1), fake_type)[:, 0])\
                      - torch.mean(discriminator2(real_embs.to(device), real_node_embs.to(device), real_type.to(device))[:, 0])
            loss_D2.backward()
            optimizer_D2.step()
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                    % (epoch + 1, args.n_epochs_pre, i+1, len(real_dataloader), loss_D2.item()))
    torch.save(discriminator2.state_dict(), "./data/"+args.dataset+"/"+args.dataset+'_d2.pt')


def train():
    #train G and D1
    for epoch in range(args.n_epochs):
        for i, (dp_type, dp_embs, dp_node_embs) in enumerate(dp_dataloader):
            initial_noise = utils.make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
            discriminator1.train()
            optimizer_D1.zero_grad()
            fake_result = generator(initial_noise)
            fake_type = fake_result[0].detach()
            fake_node = fake_result[1].detach()
            fake_idxs = fake_result[2]
            fake_embs = []
            for idxs in fake_idxs:
                fake_embs.append(node_embs_dp[idxs])
            fake_embs = torch.from_numpy(np.stack(fake_embs, axis=0)).to(device)
            fake_node_embs = []
            for idxs in fake_idxs:
                fake_node_embs.append(dp_level_embs[idxs])
            fake_node_embs = torch.from_numpy(np.stack(fake_node_embs, axis=0)).to(device)
            loss_D1 = torch.mean(discriminator1(fake_embs.transpose(0, 1), fake_node_embs.transpose(0, 1), fake_type)[:, 0]) - \
                     torch.mean(discriminator1(dp_embs.to(device), dp_node_embs.to(device), dp_type.to(device))[:, 0])
            loss_D1.backward()
            optimizer_D1.step()
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f]"
                    % (epoch + 1, args.n_epochs, i+1, len(dp_dataloader), loss_D1.item()))

            if i % args.n_critic == 0:
                generator.train()
                optimizer_G.zero_grad()
                syn_type, syn_node, syn_idxs = generator(initial_noise)
                syn_embs = []
                for idxs in syn_idxs:
                    syn_embs.append(node_embs[idxs])
                syn_embs_dp = []
                for idxs in syn_idxs:
                    syn_embs_dp.append(node_embs_dp[idxs])
                syn_embs = torch.from_numpy(np.stack(syn_embs, axis=0)).to(device)
                syn_embs_dp = torch.from_numpy(np.stack(syn_embs_dp, axis=0)).to(device)
                syn_node_embs = []
                for idxs in syn_idxs:
                    syn_node_embs.append(real_level_embs[idxs])
                syn_node_embs_dp = []
                for idxs in syn_idxs:
                    syn_node_embs_dp.append(dp_level_embs[idxs])
                syn_node_embs = torch.from_numpy(np.stack(syn_node_embs, axis=0)).to(device)
                syn_node_embs_dp = torch.from_numpy(np.stack(syn_node_embs_dp, axis=0)).to(device)
                loss_G1 = - torch.mean(discriminator1(syn_embs_dp.transpose(0, 1), syn_node_embs_dp.transpose(0, 1), syn_type)[:, 0])
                loss_G2 = - torch.mean(discriminator2(syn_embs.transpose(0, 1), syn_node_embs.transpose(0, 1), syn_type)[:, 0])
                loss_G = loss_G1 + loss_G2
                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [G1 loss: %f] [G2 loss: %f]"
                    % (epoch + 1, args.n_epochs, i+1, len(dp_dataloader),
                       loss_G.item(), loss_G1.item(), loss_G2.item()))
    torch.save(discriminator1.state_dict(), "./data/"+args.dataset+"/"+args.dataset+'_d1.pt')
    torch.save(generator.state_dict(), "./data/"+args.dataset+"/"+args.dataset+'_g.pt')

def generate_walks():
    print("Start Generating Graph...")
    transitions_per_walk = 4-1
    transitions_per_iter = 20e4
    eval_transitions = 60e7
    sample_many_count = int(np.round(transitions_per_iter/transitions_per_walk))
    n_eval_walks = eval_transitions/transitions_per_walk
    n_eval_iters = int(np.round(n_eval_walks/sample_many_count))

    smpls_type, smpls_node = [], []
    for i in range(n_eval_iters):
        initial_noise = utils.make_noise((args.batch_size, args.noise_dim), args.noise_type).to(device)
        synthetic_type, synthetic_node, synthetic_idx = generator(initial_noise)
        synthetic_type = torch.argmax(synthetic_type.cpu(), dim=2).numpy().astype(np.int32)
        synthetic_node = torch.argmax(synthetic_node.cpu(), dim=2).numpy().astype(np.int32)
        smpls_type += utils.delete_from_tail(synthetic_type, 4)
        smpls_node += utils.delete_from_tail(synthetic_node, N)
        if i % 100 == 0:
            print("Done, generating {} of {} batches meta-paths...".format(i, n_eval_iters),time.ctime(), flush=True)

    return smpls_type, smpls_node

def assemble():
    smpls_type, smpls_node = generate_walks()
    smpls_type_2 = [i for i in smpls_type if i.shape[0] == 2 and 4 not in i]
    smpls_type_3 = [i for i in smpls_type if i.shape[0] == 3 and 4 not in i]
    smpls_type_4 = [i for i in smpls_type if i.shape[0] == 4 and 4 not in i]
    smpls_type_5 = [i for i in smpls_type if i.shape[0] == 5 and 4 not in i]
    
    smpls_node_2 = [i for i in smpls_node if i.shape[0] == 2 and N not in i]
    smpls_node_3 = [i for i in smpls_node if i.shape[0] == 3 and N not in i]
    smpls_node_4 = [i for i in smpls_node if i.shape[0] == 4 and N not in i]
    smpls_node_5 = [i for i in smpls_node if i.shape[0] == 5 and N not in i]
    
    meta_path_freq = utils.meta_path_frequency(smpls_type_2, smpls_type_3, smpls_type_4, smpls_type_5)
    
    score_matrix = utils.score_matrix_from_random_walks(smpls_node_2, N)
    if len(smpls_node_3) != 0:
        score_matrix += utils.score_matrix_from_random_walks(smpls_node_3, N)
    if len(smpls_node_4) != 0:
        score_matrix += utils.score_matrix_from_random_walks(smpls_node_4, N)
    if len(smpls_node_5) != 0:
        score_matrix += utils.score_matrix_from_random_walks(smpls_node_5, N)
    score_matrix = score_matrix.tocsr()
    syn_graph = utils.heterogeneous_graph_assemble(score_matrix, M, meta_path_freq, node_type_classified)

    return syn_graph

print(time.ctime())

if args.load_model == True:
    generator.load_state_dict(torch.load("./data/"+args.dataset+"/"+args.dataset+'_g.pt'))
    discriminator1.load_state_dict(torch.load("./data/"+args.dataset+"/"+args.dataset+'_d1.pt'))
    discriminator2.load_state_dict(torch.load("./data/"+args.dataset+"/"+args.dataset+'_d2.pt'))
    if args.load_d2 == False:
        print("Training model...")
        train()
else:
    if args.load_d2 == True:
        print("Loading pretrained Discriminator2...")
        discriminator2.load_state_dict(torch.load("./data/"+args.dataset+"/"+args.dataset+'_d2.pt'))
    else:
        print("Pretraining Discriminator2...")
        pretrain()
    print("Training model...")
    train()


end = time.time()
elapsed_time = end - st
print('Execution time:', elapsed_time, 'seconds')

print("Assembling graph...")
syn_graph = assemble()

print(time.ctime())

print("Saving generated graph...")
os.makedirs("./data/"+args.dataset+"/generated/", exist_ok=True)
utils.save_graph("./data/"+args.dataset+"/generated/"+args.dataset+"_gen", syn_graph, node_value, node_type)
