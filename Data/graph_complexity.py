import os
import networkx as nx
from grakel import Graph, GraphKernel
from grakel.kernels import WeisfeilerLehman, VertexHistogram
from gensim.models import Word2Vec
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import random
import numpy as np
import argparse
import json

random.seed(42)

from graph_utils import *
from story_utils import *


def nx_to_grakel(G):
    # 使用节点度作为标签
    labels = {node: G.degree(node) for node in G.nodes()}
    return Graph(list(G.edges()), node_labels=list(labels.values()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--node_count", type=int, default=4)
    parser.add_argument("--edge_count", type=int, default=5)
    parser.add_argument("--container_count", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dag_all = generate_all_dag(args.node_count, args.edge_count)
    print(f'the number of dag ({args.node_count}, {args.edge_count}): {len(dag_all)}')


    selected_dag = []
    selected_dag_vec = []
    selected_dag_adjmatrix = []
    dag_num = 0
    for dag in dag_all:
        if dag_num >= 144:
            continue

        topo_node_list = dag.nodes()
        edge_list = dag.edges()
        parents = find_parents(topo_node_list, edge_list)
        error, character_order = generate_character_order_for_dag(args.node_count, parents, edge_list)

        if error == 1:
            continue
        
        if args.verbose:
            print(dag_num)
            # draw_dag(dag)
            print(f'拓扑序列：{topo_node_list}')
            print(f'有向边列表：{edge_list}')
            print(f'父亲节点：{parents}')

        # node2vec = Node2Vec(dag, dimensions=64, walk_length=30, num_walks=200, workers=4)
        # model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        # embedding_vector = np.array(model.wv[str(0)], copy=True)
        # for node_id in range(1, args.node_count):
        #     embedding_vector += np.array(model.wv[str(node_id)], copy=True)
        # embedding_vector /= args.node_count

        adj_matrix = nx.adjacency_matrix(dag)

        selected_dag.append(dag)
        # selected_dag_vec.append(embedding_vector)
        selected_dag_adjmatrix.append(adj_matrix)
        dag_num += 1

    sim_list = []
    for i in range(128, 144):
        dag = selected_dag[i]
        tmp_sim = 0
        for j in range(128):
            dag_ = selected_dag[j]
            # graph1 = Graph(dag.edges(), node_labels={node: dag.degree(node) for node in dag.nodes()})
            # graph2 = Graph(dag_.edges(), node_labels={node: dag_.degree(node) for node in dag_.nodes()})
            # wl_kernel = WeisfeilerLehman(n_iter=5, base_graph_kernel=VertexHistogram)
            # sim = wl_kernel.fit_transform([graph1, graph2])
            # tmp_sim += sim[0][1]

            # dot_product = np.dot(selected_dag_vec[i], selected_dag_vec[j])
            # norm_vec1 = np.linalg.norm(selected_dag_vec[i])
            # norm_vec2 = np.linalg.norm(selected_dag_vec[j])
            # tmp_sim += (dot_product / (norm_vec1 * norm_vec2))

            adjm1 = np.array(selected_dag_adjmatrix[i].todense())
            adjm2 = np.array(selected_dag_adjmatrix[j].todense())
            equal_elements = adjm1 == adjm2
            count_equal = np.sum(equal_elements)
            print(count_equal)
            tmp_sim += count_equal

        sim_list.append(tmp_sim / 128)

    print(sim_list)

    
