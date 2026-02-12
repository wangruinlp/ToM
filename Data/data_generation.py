import os
import networkx as nx
import matplotlib.pyplot as plt
import random
import argparse
import json
import numpy as np

from matplotlib.patches import Rectangle
from networkx.algorithms.dag import topological_generations
from matplotlib.patches import FancyArrowPatch  # NEW
from networkx.algorithms.dag import topological_generations  # NEW

random.seed(42)

from graph_utils import *
from story_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--node_count", type=int, default=5)
    parser.add_argument("--edge_count", type=int, default=7)
    parser.add_argument("--container_count", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dag_all = generate_all_dag(args.node_count, args.edge_count)
    print(f'the number of dag ({args.node_count}, {args.edge_count}): {len(dag_all)}')

    if os.path.exists(f'{args.node_count}_node_{args.edge_count}_edge_{args.container_count}_container.jsonl'):
        os.remove(f'{args.node_count}_node_{args.edge_count}_edge_{args.container_count}_container.jsonl')

    all_cnt = 0
    dag_num = 0
    all_data = []
    valid_dag_all = []
    Nnm = 0
    for dag in dag_all:
        if dag_num >= 120:
            continue

        topo_node_list = dag.nodes()
        edge_list = dag.edges()
        parents = find_parents(topo_node_list, edge_list)

        # 根据graph结构构建人物顺序
        error, character_order = generate_character_order_for_dag(args.node_count, parents, edge_list)
        if error == 1:
            continue

        valid_dag_all.append(dag)
        if args.verbose:
            print(dag_num)
            # draw_dag(dag)
            print(f'拓扑序列：{topo_node_list}')
            print(f'有向边列表：{edge_list}')
            print(f'父亲节点：{parents}')

        character_mapping_all = generate_character_mapping_for_dag(args.node_count)
        action_all = generate_character_action_for_dag(args.container_count, args.node_count, topo_node_list)
        print(len(character_mapping_all), len(action_all))
        
        for character_num in range(len(character_mapping_all)):
            for action_num in range(len(action_all)):
                character_mapping = character_mapping_all[character_num]
                action = action_all[action_num]
                story = merge_to_story(args.node_count, character_order, character_mapping, action)

                sample = {"id": all_cnt,
                            "graph_id": dag_num,
                            "character_id": character_num,
                            "action_id": action_num,
                            "story": story,
                            "query": {}}

                for order_num in range(1, args.node_count + 1):
                    question, answer = generation_question_and_answer(order_num, parents, character_mapping, action)
                    order = f"{order_num}-order"
                    sample["query"][order] = {"question": question, "answer": answer}
                all_data.append(sample)

                all_cnt += 1

            #     if action_num > 0:
            #         break
            # if character_num > 0:
            #     break
        
        if list(topo_node_list) != [i for i in range(args.node_count)] and Nnm == 0:
            Nnm = dag_num
        dag_num += 1

    print(f"total num: {all_cnt}")
    print(f'N(n, m) = {Nnm}')

    with open(f'{args.node_count}_node_{args.edge_count}_edge_{args.container_count}_container.jsonl', 'a') as f:
        for sample in all_data:
            json_line = json.dumps(sample)
            f.write(json_line + '\n')





    def dag_layered_layout(G, layer_gap=1.4, node_gap=1.6, jitter=0.03, seed=42):
        """
        按拓扑层数纵向排布，同层横向等距；加少量随机抖动避免重合。
        layer_gap: 层间距（y轴）
        node_gap : 同层节点间距（x轴）
        jitter   : 抖动幅度（0~0.1之间即可）
        """
        random.seed(seed)
        layers = list(topological_generations(G))  # 每一层是一个可迭代的节点集合
        layers = layers[::-1]  # 让“源头”在上方，终点在下方

        pos = {}
        for y, layer in enumerate(layers):
            layer = list(layer)
            # 居中：以中心为对称排列
            offset = - (len(layer) - 1) * node_gap / 2.0
            for i, v in enumerate(layer):
                x = offset + i * node_gap + random.uniform(-jitter, jitter)
                yy = y * layer_gap + random.uniform(-jitter, jitter)
                pos[v] = (x, yy)
        return pos

    # =========================================
    # 绘制前 144 张 DAG 到一张图片（分层布局 + 深色系）
    # =========================================
    plt.rcParams["font.family"] = "Times New Roman"

    num_graphs = min(len(valid_dag_all), 120)  # 只取前120张
    # 网格更宽松：10 列（可改回 12）
    cols = 8
    rows = (num_graphs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(36, 48))  # 放大画布

    # 颜色：前 108 张蓝色，后 12 张绿色（更稳重）
    BLUE = "#4F6EDB"
    GREEN = "#3DA35A"

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]

        if i < num_graphs:
            dag = valid_dag_all[i]
            pos = dag_layered_layout(dag, layer_gap=1.4, node_gap=1.6, jitter=0.03)

            node_color = GREEN if i >= num_graphs - 8 else BLUE

            # 先画节点和边（不画标签）
            nx.draw(
                dag, pos,
                with_labels=False,
                node_size=450,
                arrowsize=10,
                width=1.2,
                node_color=node_color,
                edgecolors="#1f1f1f",   # 节点描边
                linewidths=1.2,
                ax=ax
            )

            # 再单独画标签，带白底，避免遮挡边
            labels = {n: str(n+1) for n in dag.nodes()}
            nx.draw_networkx_labels(
                dag, pos, labels=labels,
                font_size=11, font_family="Times New Roman",
                bbox=dict(boxstyle="round,pad=0.25", fc="none", ec="none", alpha=0.9),
                ax=ax
            )

            ax.set_title(f"G{i}", fontsize=12, fontfamily="Times New Roman")
        ax.axis("off")

    # 子图间距更宽
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    out_png = f"all_dags_{args.node_count}_node_{args.edge_count}_edge.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {out_png}")

