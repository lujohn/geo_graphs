
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import schur
from sklearn.metrics import auc

import os
import json
import pickle
import time



def read_graph(file):
    G = nx.read_adjlist(file, nodetype=int)    
    return G

def get_dists(G, kMax):
    path_lengths = nx.all_pairs_shortest_path_length(G, cutoff=kMax)
    N = len(G.nodes()) 
    D = np.sqrt(path_lengths_to_array(path_lengths, kMax))
    D = double_center(D**2)
    return D

######################### Clean Data ########################
def double_center(D, vectorized=True): 
    """
    Inputs:
        D := N x N squared distance matrix
    """
    N = D.shape[0]
    
    if vectorized:
        # Centering Matrix
        e = np.ones((N,1))
        H = np.identity(N) - (1./N)*np.matmul(e,e.T)
        
        # Double center data
        A = (-1./2)*np.matmul(H, np.matmul(D,H))
    else:
        A = np.zeros(D.shape)
        tmp = (1./(N**2)) * (D).sum()
        for i in range(N):
            for j in range(N):
                A[i,j] = -(1./2) * (D[i,j] - np.mean(D[i,:]) - np.mean(D[:,j]) + tmp)  
    return A


def path_lengths_to_array(path_lens, K_max):
    """
    Input:
        G := networkx Graph
        path_lengths := path_lengths[i] is a dictionary {node => degree}.
    Ouputs:
        D := distance matrix
    """
    D = np.zeros((len(path_lens), len(path_lens)))
    
    for u, adj_dict in path_lens.items():
        for v, length in adj_dict.items():
            if length == 0:
                D[u,v] = K_max
            else: 
                D[u,v] = length
    return D

def rescale(X_hat): 
    """
    Inputs:
        X_hat := r x N matrix where each column is a coordinate in r space
    """
    # Rescale each column in X_hat to be in the unit cube
    X_hat_rescaled = np.zeros(X_hat.shape)
    r = X_hat.shape[0]
    for i in range(r):
        minimum = np.min(X_hat[i,:])
        maximum = np.max(X_hat[i,:])
        X_hat_rescaled[i,:] = (X_hat[i,:] - minimum) / (maximum - minimum)
    return X_hat_rescaled

def save_dists(dname, fname, dists):
    """
    path_lengths := 2-d arrays of double centered distances
    """
    misc_dir = os.path.join(dname, 'misc')
    if not os.path.isdir(misc_dir):
        os.makedirs(misc_dir)
    with open(os.path.join(misc_dir, 'dists_'+fname),'wb') as f:
        print('Saving doubled centered distances for ' + str(fname))
        pickle.dump(dists, f)

def load_dists(dname, fname): 
    misc_dir = os.path.join(dname, 'misc')
    fpath = os.path.join(misc_dir, 'dists_'+fname)
    if os.path.isfile(fpath):
        with open(fpath, 'rb') as f:
            print('Loading double centered dists for ' + str(fname))
            dists = pickle.load(f)
        return dists
    else:
        return None


######################## Create the Embedding ########################
def get_embedding_coords(SIGMA, U):
    X_hat = np.sqrt(np.diag(SIGMA)).dot(U.T)
    # Rescale into the unit square
    X_hat_rescaled = rescale(X_hat)
    return X_hat_rescaled

def save_embedding_coords(dname, fname, X_hat):
    misc_dir = os.path.join(dname, 'misc')
    if not os.path.isdir(misc_dir):
        os.makedirs(misc_dir)

    with open(os.path.join(misc_dir, 'embedding_coords_'+fname),'wb') as f:
        print('Saving Embedding Coords for ' + str(fname))
        pickle.dump(X_hat, f)

def load_embedding_coords(dname, fname):
    misc_dir = os.path.join(dname, 'misc')
    fpath = os.path.join(misc_dir, 'embedding_coords_'+fname)
    if os.path.isfile(fpath):
        with open(fpath, 'rb') as f:
            print('Loading Embedding Coordinates from file...')
            X_hat = pickle.load(f)
        return X_hat
    else:
        return None

def make_embedding(X_hat, G, epsilon):
    """
    Inputs: 
        X_hat := r x N matrix representing the embedding of network in R^r space
        G := original graph
    Outputs:
        H := Graph of embedding where edge [x_i, x_j] exists iff their euclidean distance is < epsilon
    """
    
    # euclidean_distances calculates pairwise dists based on rows, so we take the transpose
    Dists = euclidean_distances(X_hat.T)
    N = X_hat.shape[1]
    H = nx.Graph()
    H.add_nodes_from(G)
    assert(G.nodes() == H.nodes())
    for i in range(N):
        for j in range(i+1, N):
            if Dists[i,j] < epsilon:
                H.add_edge(i,j)
                
    return H

############################# Evaluation ###############################
# def evaluate_embedding(G, H):
#     TN = TP = FP = FN = 0
    
#     D = G - H
#     FN = (D == 1).sum()
#     FP = (D == -1).sum()
    
#     TP = ((D == 0) & (G == 1)).sum()
#     TN = ((D == 0) & (G == 0)).sum()
    
#     return TP, TN, FP, FN

def eval_embedding(G, H):
    assert(len(G.nodes()) == len(H.nodes()))
    TN = TP = FP = FN = 0

    # count TP and FN
    for (u,v) in G.edges_iter():
        if H.has_edge(u,v):
            TP = TP + 1
        else:
            FN = FN + 1

    # count FP
    for (u,v) in H.edges_iter():
        if not G.has_edge(u,v):
            FP = FP + 1

    # Calculate TN
    N = len(H.nodes())
    TN = (N*(N-1)/2) - TP - FN - FP
    return TP, TN, FP, FN     

def get_stat(X_hat, G, ep):
    H = make_embedding(X_hat, G, ep)
    TP, TN, FP, FN = eval_embedding(G, H)
    sens, spec = float(TP)/(TP+FN), float(TN)/(TN+FP)
    return H, sens, spec, (TP, TN, FP, FN)


############################# Save Results ###############################
def save_embedding(dname, fname, ep, H):
    if not os.path.isdir(dname):
        os.makedirs(dname)
    # Save graph
    graphs_dir = os.path.join(dname, 'graphs')
    if not os.path.isdir(graphs_dir):
        os.makedirs(graphs_dir)

    graphs_dir = os.path.join(graphs_dir, fname)
    if not os.path.isdir(graphs_dir):
        os.makedirs(graphs_dir)

    with open(os.path.join(graphs_dir, 'ep_'+str(ep)+'.el'), 'w') as f:
        for (u,v) in H.edges_iter():
            f.write("%d %d \n" % (u,v))

def save_stats(dname, fname, stats):
    # Save stats
    if not os.path.isdir(dname):
        os.makedirs(dname)

    stats_dir = os.path.join(dname, 'stats')
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)

    fpath = os.path.join(stats_dir, fname)
    print('Saving stats to: ', fpath)
    with open(fpath, 'w') as f:
        f.write(json.dumps(stats))

def plot_roc(dname, fname, specs, senses, save=True):
    print('Plotting ROC')
    plt.figure()
    plt.plot(1-specs, senses)
    ax = plt.gca()
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("sensitivity")
    ax.set_title(fname)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    area = auc(1-specs, senses)
    plt.text(.7, .1, "AUC: " + str(area) , bbox=dict(facecolor='red', alpha=0.5))

    # save ROC plot
    if not os.path.isdir(dname):
        os.makedirs(dname)

    roc_dir = os.path.join(dname, 'roc')
    if not os.path.isdir(roc_dir):
        os.makedirs(roc_dir)

    fpath = os.path.join(roc_dir, fname)
    plt.savefig(fpath + '_roc.png')
    
    # Save the AUC
    with open(os.path.join(roc_dir, 'auc.el'),'w') as f:
        f.write(str(area))

    
### For Testing ###
def get_subgraph(G, n_nodes=100):
    idx = np.random.choice(np.arange(len(G.nodes())), n_nodes)
    H = nx.subgraph(G, nbunch=idx)
    H = nx.convert_node_labels_to_integers(H)
    return H


def get_best_epsilon(stats):
    best_ep = -1
    best_dist = np.inf  # distance to (0,1)
    best_sense = -1
    best_spec = -1
    for ep, stat in stats.items():
        sense = float(stat[0])
        spec = float(stat[1])

        d = np.sqrt((1-spec)**2 + (sense-1)**2)
        if d < best_dist:
            best_dist = d
            best_ep = ep
            best_sense = sense
            best_spec = spec

    print('best_ep found', best_ep)
    print('dist to (0,1)', best_dist)
    print("sense: " + str(best_sense) + " 1-spec: " + str(1-best_spec))
    return best_ep

def load_stats(fname):
    with open(fname) as f:
        stat = json.load(f)
    return stat

def run_file(dname, fname, epsilons, dim=2, kMax=7):
    print(dname)
    print(fname)
    fPath = os.path.join(dname, fname)
    G = read_graph(fPath)
    print("num nodes: " + str(len(G.nodes())))
    print("num edges: " + str(len(G.edges())))

    # Load Centered Distances from disk if exists
    D = load_dists(dname, fname)
    if D == None:
        print('Calculating Distances...')
        D = get_dists(G, kMax)
        save_dists(dname, fname, D)

    # Load embedding coordinates from disk if exists
    X_hat = load_embedding_coords(dname, fname)
    if X_hat == None:
        print("Calculating Embedding Coordinates...")
        U,S,V = np.linalg.svd(D)
        # Take the 'dim' largest eigenvalues and eigenvectors
        S = S[:dim]
        U = U[:,:dim]
        X_hat = get_embedding_coords(S, U)
        save_embedding_coords(dname, fname, X_hat)

    print("Caclulating Sensistivity / Specificity for several epsilons...")
    senses = []
    specs = []
    # for each epsilon, store relevant statistics
    stats = {}
    it_count = 0
    print('Trying ' + str(len(epsilons)) + ' epsilons...')
    for ep in epsilons:
        H, sens, spec, (TP, TN, FP, FN) = get_stat(X_hat, G, ep)
        senses.append(sens)
        specs.append(spec)
        stats[ep] = (sens, spec, TP, TN, FP, FN)

        print("Finished ep =: ", ep)
        #it_count += 1

        # Save embedding edge list to a file
        save_embedding(dname, fname, ep, H)
        
    # Save stats
    save_stats(dname, fname, stats)

    # Plot the ROC
    plot_roc(dname, fname, np.array(specs), np.array(senses), save=True)

def run_dir(dname, dim=2, kMax=7):
    # Run on a folder of networks and save the results
    #epsilons = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.3, 1.4]
    epsilons = [1e-5, 1e-4, 1e-3, 1e-2] + list (np.arange(0.1, np.sqrt(1.4), .05))
    # epsilons = np.arange(0.35, 1.4, 0.05)
    #epsilons = [1e-5, 1e-3, .1]
    print('Reading from ' + dname)
    print('eps: ', epsilons)
    for fname in os.listdir(dname):
        if not fname.endswith('.el'):
            continue
        tic = time.clock()
        run_file(dname, fname, epsilons, dim, kMax)
        toc = time.clock()
        print('Time: ', toc-tic)


import sys
if __name__ == "__main__":
    kMax = 5
    dim = 2
    f = sys.argv[1]
    run_dir(f, dim, kMax)


    
