import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sp


G = nx.read_graphml("examples/geom_50_50_50f.graphml")
G = nx.convert_node_labels_to_integers(G)
G.remove_edges_from(nx.selfloop_edges(G))
nlist = list(G.nodes())

# Get Adjacency Matrix
Adj_G = nx.adjacency_matrix(G)

# Get Degree Matrix
Deg_G = np.diag(Adj_G.sum(axis = 1))

# Get Laplacian Matrix from directed Graph
Lap_G = Deg_G - Adj_G.T
#Lap_G = nx.laplacian_matrix(G, nodelist=nlist, weight='weight')
n = Lap_G.shape[0]
Lap_G = Lap_G + 1e-6 * sp.eye(n, format='csr')
Lap_G = sp.csr_matrix(Lap_G)

######################################################################################################### 5.1 Initial Sparsifier Construction 

A_G = sp.csr_matrix(Adj_G)            
n = A_G.shape[0]

# 1) Row-normalize the transpose of A_G
row_sums = np.array(A_G.sum(axis=1)).flatten()
row_sums[row_sums == 0] = 1          # avoid div0
D_inv = sp.diags(1.0 / row_sums)
A_un = D_inv.dot(A_G.T)              # (D^-1 * A_G')

# 2) Symmetrize and take upper triangle
A_sym = (A_un + A_un.T) * 0.5
A_uper = sp.triu(A_sym, format='coo')  # only i<=j entries

# 3) Invert weights
rows, cols, vals = A_uper.row, A_uper.col, A_uper.data
inv_vals = 1.0 / vals

# 4) Build Graph and add edges with inverted weights
G_mod = nx.Graph()
for i, j, w in zip(rows, cols, inv_vals):
    G_mod.add_edge(int(i), int(j), weight=float(w))

# 5) Compute the minimum spanning tree (Kruskal)
MST = nx.minimum_spanning_tree(G_mod, algorithm="kruskal")

S = nx.DiGraph()
S.add_nodes_from(nlist)

# Edge direction recovery
for u, v, data in MST.edges(data=True):
    if Adj_G[u, v] > 0:
        w = Adj_G[u, v]
        S.add_edge(u, v, weight=w)

# Each Node has an outgoing edge
for u in G.nodes():
    if G.out_degree(u) >= 1 and S.out_degree(u) == 0:

        try:
            _, v_max, w_max = max(G.out_edges(u, data="weight", default=0.0), key=lambda tup: tup[2])

        except ValueError:
            continue
        
        S.add_edge(u, v_max, weight=w_max)


# Get Laplacian of S
Adj_S = nx.adjacency_matrix(S, nodelist=nlist)
Deg_S = np.diag(Adj_S.sum(axis = 1))
Lap_S = Deg_S - Adj_S.T
Lap_S = Lap_S + 1e-6 * sp.eye(n, format='csr')
Lap_S = sp.csr_matrix(Lap_S)


######################################################################################################### 5.2 / 5.3


def compute_dominant_generalized_eigenvector(Lap_G, Lap_S, r):
    Lap_SU = Lap_S @ Lap_S.T
    Lap_GU = Lap_G @ Lap_G.T
    
    epsilon = 1e-4
    Lap_SU_reg = Lap_SU + epsilon * sp.eye(Lap_SU.shape[0], format='csr')
    Lap_GU_reg = Lap_GU + epsilon * sp.eye(Lap_GU.shape[0], format='csr')
    
    eigvals, eigvecs = sp.linalg.eigs(Lap_GU_reg, k=r, M=Lap_SU_reg, which='LM')
    lambda_max = eigvals[0].real
    h_ts = [eigvecs[:, i].real / np.linalg.norm(eigvecs[:, i].real) for i in range(r)]
    
    eigvals, eigvecs = sp.linalg.eigs(Lap_GU_reg, k=2, M=Lap_SU_reg, sigma=0.0, which='LM')
    real_eigs = np.sort(eigvals.real)
    lambda_min = real_eigs[1]
    
    return lambda_min, lambda_max, h_ts

# Compute spectral sensitivites: zeta 
def spectral_sensitivites_candidates(Lap_S, h, off_edges, alpha):
    
    edge_zeta_pairs = []
    
    for p, q, w in off_edges:
        lsep = Lap_S[:, p].toarray().flatten() 
        diff = h[p] - h[q]
        proj = np.dot(lsep, h)
        zeta = diff * (proj + proj)  # (ht_p - ht_q) * (lsep' * ht + ht' * lsep)
        zeta *= w                    
        edge_zeta_pairs.append(((p, q, w), zeta))
        
    edge_zeta_pairs.sort(key=lambda item: item[1], reverse=True)

    number = round(n * alpha)
    positive_edges = [pair for pair in edge_zeta_pairs if pair[1] > 0]
    n_positive = len(positive_edges)
    n_ch = min(n_positive, number)
    # Final candidate list of edges (just the (p, q, w) part)
    candidate_edges = [pair[0] for pair in positive_edges[:n_ch]]
    if (len(candidate_edges) == 0):
        candidate_edges = off_edges
    return candidate_edges

######################################################################################################### 5.4 Edge similarity

# Step 1: Defined in Digrass algoritm
# Step 2: compute r-dimenstional embedding_vector for each edge (r = 5 in our case)
def spectral_embedding_vectors(Lap_S, h_ts, candidate_edges, lambda_max, r):

    embeddings = {}
    ht = h_ts[0]
    for (p, q, w) in candidate_edges:
        vec = np.zeros(r)
        lsep = Lap_S[:, p].toarray().flatten()

        for j in range(r):
            h_j = h_ts[j]
            lsep_ht = np.dot(lsep, h_j)
            epq_ht = h_j[p] - h_j[q]
            vec[j] = lambda_max * (2 * epq_ht * lsep_ht) 

        embeddings[(p, q)] = vec

    return embeddings

def edge_similarity_filter(candidate_edges, embeddings, rho, d_out, S):
    selected_edges = []
    similarity_list = []
    outdeg_counter = np.fromiter((S.out_degree(u) for u in range(n)), dtype=int)

    # Always add first edge -> achieves full recovery of graph
    (p0, q0, w0) = candidate_edges[0]
    selected_edges.append((p0, q0, w0))
    similarity_list.append(embeddings[(p0, q0)])
    outdeg_counter[p0] += 1  

    for i in range(1, len(candidate_edges)):
        p, q, w = candidate_edges[i]
        vec_i = embeddings[(p, q)]
        norm_i = np.linalg.norm(vec_i)
        if norm_i == 0:
            continue

        # Outdegree check 
        if outdeg_counter[p] >= d_out:
            continue

        normalized_diffs = [
            np.linalg.norm(vec_i - vec_j) / norm_i
            for vec_j in similarity_list
        ]
        similarity_list.append(vec_i)

        if max(normalized_diffs) < rho:
            selected_edges.append((p, q, w))
            outdeg_counter[p] += 1
    
    return selected_edges

######################################################################################################### 5.5 Without LAMG


def diGRASS (G, S, Lap_G, Lap_S, d_out, limit, alpha, rho):
    # Step 1: Compute the dominant generalized eigenvector h_t and its eigenvalue lambda_max
    lambda_min, lambda_max, h_ts = compute_dominant_generalized_eigenvector(Lap_G, Lap_S, 1)
    h = h_ts[0]
    
    # Step 2: While lambda_max > lambda_limit 
    # while(lambda_max > limit):
    plot_data_lam = []
    plot_data_lam.append((round(nx.number_of_edges(S)/nx.number_of_edges(G), 2), round(float(lambda_max / lambda_min), 2)))
    while(lambda_max > limit):
        # Step 3/4: Compute the spectral sensitivity Zeta_pq of each off-subgraph edge (p,q), Sort edges in descending order and include top alpha% in the candidate edge list
        off_edges = [(u, v, Adj_G[u, v]) 
           for u, v in G.edges() if not S.has_edge(u, v)]

        candidate_edges = spectral_sensitivites_candidates(Lap_S, h, off_edges, alpha)

        # Step 5: Form the final edge list that only includes spectrally-dissimilar off-subgraph edges obtained by using edge_similarities_checking
        final_edges = edge_similarities_checking(S, candidate_edges, Lap_G, Lap_S, lambda_max, d_out, rho)
        
        # Step 6: Update S = S + final_edges 
        for p, q, w in final_edges:
            S.add_edge(p, q, weight=w)

        Adj_S = nx.adjacency_matrix(S, nodelist=nlist)
        Deg_S = np.diag(Adj_S.sum(axis = 1))
        Lap_S = Deg_S - Adj_S.T
        Lap_S = Lap_S + 1e-6 * sp.eye(n, format='csr')
        Lap_S = sp.csr_matrix(Lap_S)
    
        lambda_min, lambda_max, h_ts = compute_dominant_generalized_eigenvector(Lap_G, Lap_S, 1)
        h = h_ts[0]
        
        print("candidates:", len(candidate_edges))
        print("final:", len(final_edges))
        print("total:",nx.number_of_edges(S))
        print("lambda_max:", lambda_max)
        print("lambda_min:", lambda_min)
        print("%","Edges:", round(nx.number_of_edges(S)/nx.number_of_edges(G), 2))
        plot_data_lam.append((round(nx.number_of_edges(S)/nx.number_of_edges(G), 2), round(float(lambda_max / lambda_min), 2)))
        
    return plot_data_lam, Lap_S, lambda_max

def edge_similarities_checking(S, candidate_edges, Lap_G, Lap_S, lambda_max, d_out, rho):
     # Step 1: We compute first r dominant generalized eigenvectors
     r = int(np.log(Lap_G.shape[0]))
     # in this case r = 5
     r = 5
     _, _, h_ts = compute_dominant_generalized_eigenvector(Lap_G, Lap_S, r)
     
     # Step 2: Compute r-dimenstional embedding vectors
     embeddings = spectral_embedding_vectors(Lap_S, h_ts, candidate_edges, lambda_max, r)

     # Step 3-9: Calculate the spectral similarity score between every edge and if they are dissimilar and the outdegree is not reached, add it to the final_edge list$
     final_edges = edge_similarity_filter(candidate_edges, embeddings, rho, d_out, S)
     return final_edges
    
    
######################################################################################################################## experiments

_, lambda_init, _ = compute_dominant_generalized_eigenvector(Lap_G, Lap_S, 1)
S_copy1 = S.copy()
S_copy2 = S.copy()
plot_data_lam1, Lap_S1, lambda_max1 = diGRASS(G, S, Lap_G, Lap_S, d_out=40, limit=1.03, alpha=0.5, rho=0.9)
plot_data_lam2, Lap_S2, lambda_max2 = diGRASS(G, S_copy1, Lap_G, Lap_S, d_out=40, limit=1.03, alpha=0.5, rho=0.5)
plot_data_lam3, Lap_S3, lambda_max3 = diGRASS(G, S_copy2, Lap_G, Lap_S, d_out=40, limit=1.03, alpha=0.5, rho=0.1)


######################################################################################################################## plotting

edg1, lams1 = zip(*plot_data_lam1)
edg2, lams2 = zip(*plot_data_lam2)
edg3, lams3 = zip(*plot_data_lam3)

plt.close('all')

colors = ['#0072B2', '#D55E00', '#009E73']
plt.plot(edg1, lams1, linestyle='-', linewidth=2, color='C0', label='rho=0.9')
plt.plot(edg2, lams2, linestyle='-', linewidth=2, color='C1', label='rho=0.5')
plt.plot(edg3, lams3, linestyle='-', linewidth=2, color='C2', label='rho=0.1')

plt.yscale('log')
plt.xlabel(r'$|E_S|/|E_G|$')
plt.ylabel(r'$\lambda_{\max}/\lambda_{\min}^{+}$ (log scale)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(loc='best')            
plt.tight_layout()
plt.show()

# build iteration indices
it1 = np.arange(1, len(lams1)+1)
it2 = np.arange(1, len(lams2)+1)
it3 = np.arange(1, len(lams3)+1)

plt.close('all')
plt.plot(it1, lams1, linestyle='-', linewidth=2, color='C0', label='rho=0.9')
plt.plot(it2, lams2, linestyle='-', linewidth=2, color='C1', label='rho=0.5')
plt.plot(it3, lams3, linestyle='-', linewidth=2, color='C2', label='rho=0.1')

plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel(r'$\lambda_{\max}/\lambda_{\min}^{+}$ (log scale)')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend(loc='best')
plt.tight_layout()
plt.show()