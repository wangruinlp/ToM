import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import itertools

random.seed(42)


def draw_dag(dag):
    pos = nx.spring_layout(dag)  # 使用 spring 布局
    nx.draw(dag, pos, with_labels=True, node_size=500, node_color="lightblue",
            arrowsize=20, font_size=10, font_weight="bold")
    plt.title("Random Directed Acyclic Graph (DAG)")
    plt.savefig('dag.png')





def find_parents(node_list, edge_list):
    parents = {node: [] for node in node_list}
    for edge in edge_list:
        parents[edge[1]].append(edge[0])

    return parents


def generate_random_dag(node_count, edge_count):
    # 创建一个有向图
    dag = nx.DiGraph()

    # 添加节点
    dag.add_nodes_from(range(node_count))

    # 确保图是无环的
    while dag.number_of_edges() < edge_count:
        # 随机选择两个不同的节点
        u, v = random.sample(dag.nodes, 2)

        # 检查是否可行（u -> v 没有反向路径，并且没有同样的边）
        if not dag.has_edge(u, v) and not nx.has_path(dag, v, u):
            dag.add_edge(u, v)

    return dag



def are_all_nodes_covered(edges, num_nodes):
    # 创建一个集合来存储所有出现在边中的节点
    covered_nodes = set()

    # 遍历每条边，将出现在边中的节点添加到集合中
    for edge in edges:
        covered_nodes.update(edge)

    # 检查集合中的节点数是否等于节点总数
    return len(covered_nodes) == num_nodes


def generate_all_dag(node_count, edge_count):
    '''
    给定节点数和边数，构建出所有可能的有向无环图
    '''
    dag_all = []

    node_ids = [i for i in range(node_count)]
    node_permutations = list(itertools.permutations(node_ids))

    for node_list in node_permutations:
        edge_ids = []
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                edge_ids.append((node_list[j], node_list[i]))
        edge_combinations = list(itertools.combinations(edge_ids, edge_count))
        for edge_list in edge_combinations:
            # 不能有孤立node
            if not are_all_nodes_covered(edge_list, node_count):
                continue
            G = nx.DiGraph()
            G.add_nodes_from(list(node_list))
            G.add_edges_from(list(edge_list))
            dag_all.append(G)

    return dag_all
