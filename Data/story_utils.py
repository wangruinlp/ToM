import random
import itertools

random.seed(42)


character_pool_all = ["Alice", "Bob", "Charles", "David", "Emily", "Frank", "George", "Henry", "Isabella", "Jack", "Kevin", "Lily"]
object_pool_all = ["book", "pen", "fork", "knife", ]
container_pool_all = ["cabinet", "sofa", "table", "chair", "basin", "bowl", "box", "basket", "drawer", "shelf"]


def generate_character_order_for_dag(node_count, parents, edge_list):
    character_order = []
    error = 0
    room_node = []
    for node, parent_list in parents.items():
        in_list = [node] if node not in room_node else []
        for n in parent_list:
            if n not in room_node:
                in_list.append(n)

        for n in in_list:
            if n not in room_node:
                room_node.append(n)

        out = node
        for n in room_node:
            if n != out and (n, node) not in edge_list:
                error = 1

        room_node.remove(out)
        character_order.append({'in': in_list, 'out': out})
    return error, character_order


def generate_character_mapping_for_dag(node_count):
    pool_split = []
    for i in range(node_count):
        pool_split.append(character_pool_all[
                              (len(character_pool_all) * i) // node_count : (len(character_pool_all) * (i + 1)) // node_count])
    character_mapping_all = list(itertools.product(*pool_split))

    return character_mapping_all


def generate_character_action_for_dag(container_num, node_count, topo_node_list):
    object_comb = list(itertools.combinations(object_pool_all, 1))
    # container_comb = list(itertools.combinations(container_pool_all, container_num))
    pool_split = []
    for i in range(container_num):
        pool_split.append(container_pool_all[
                              (len(container_pool_all) * i) // container_num : (len(container_pool_all) * (i + 1)) // container_num])
    container_comb = list(itertools.product(*pool_split))

    action_all = []
    for object_pool in object_comb:
        for container_pool in container_comb:
            action = [{} for i in range(node_count)]
            for i in topo_node_list:
                action[i] = {"object": random.choice(object_pool), \
                             "container": random.choice(container_pool)}
            action.append(container_pool)
            action_all.append(action)

            if len(action_all) >= 144:
                break
        
        if len(action_all) >= 144:
            break

    return action_all


def merge_to_story(node_count, character_order, character_mapping, action):
    init_object = action[0]["object"]
    init_container = random.choice(action[-1])
    current_container = init_container
    
    text = "There were "
    for i in range(len(action[-1])):
        if i == len(action[-1]) - 1:
            text += f"and one {action[-1][i]} in the room.\n"
        else:
            text += f"one {action[-1][i]}, "
    text += f"The {init_object} was in the {init_container}.\n"

    for step in character_order:
        for i in range(len(step['in'])):
            c = step['in'][i]
            if i == 0:
                text += (str(character_mapping[c]))
            else:
                text += (", " + str(character_mapping[c]))

        if len(step['in']) > 0:
            text += " entered the room.\n"

        out_character = character_mapping[step['out']]
        moved_object = action[int(step['out'])]["object"]
        to_container = action[int(step['out'])]["container"]

        if to_container != current_container:
            text += f"{out_character} moved the {moved_object} to {to_container}.\n"
        else:
            text += f"{out_character} made no movement.\n"

        current_container = to_container
        text += f"{out_character} exited the room.\n"

    return text


def generation_question_and_answer(order_num, parents, character_mapping, action):
    question_character_order = random.sample(character_mapping, order_num)
    random.shuffle(question_character_order)

    question_text = "Where does "
    for character in question_character_order:
        question_text += f"{character} thinks "
    query_object = action[0]["object"]
    question_text += f"the {query_object} is?"

    pointer_id = character_mapping.index(question_character_order[0])
    for i in range(len(question_character_order) - 1):
        son_id = character_mapping.index(question_character_order[i + 1])
        if pointer_id in parents[son_id]:
            pointer_id = son_id
    answer = action[pointer_id]["container"]

    return question_text, answer

