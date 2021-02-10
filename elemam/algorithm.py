# -*- coding: utf-8 -*-

import math


# Initializes the algorithm
def calculate(nodes, ac):
    min_k_nodes = [nodes[-1]]
    topnode = [nodes[-1]]

    evaluate(nodes, ac, topnode, min_k_nodes)

    return min_k_nodes


def sort(nodes):
    sorted_array = []
    part_array = {}
    for n in reversed(nodes):
        h = n.level

        try:
            part_array[h].append(n)
        except KeyError:
            part_array.update({h: [n]})

    for n in sorted(part_array):
        sorted_array.extend(part_array[n])

    return sorted_array


def evaluate(nodes, ac, topnode, min_k_nodes):
    for n in nodes:
        if n.level == math.floor((nodes[-1].level + nodes[0].level) / 2):
            if check_kanon(n, ac):
                test = createsubarray(n, topnode[-1], True, True)
                isOk = True
                for mkn in reversed(min_k_nodes):
                    if mkn in test:
                        min_k_nodes.remove(mkn)
                        continue
                    mkn_subarray = createsubarray(mkn, topnode[-1], True, True)
                    if n in mkn_subarray:
                        isOk = False
                if isOk:
                    min_k_nodes.append(n)

            if n.iskanon:
                # Returns False if every node was checked and thus stops the algorithm
                if n.set_children_kanon() is False:
                    return False
                if (nodes[-1].level - nodes[0].level) + 1 < 4:
                    for node in nodes:
                        if node.iskanon is None:
                            if evaluate([node], ac, topnode, min_k_nodes) is False:
                                return False
                    continue

                newnodes = createsubarray(nodes[0], n, False)
                if newnodes is None:
                    continue

                sorted = sort(newnodes)
                if evaluate(sorted, ac, topnode, min_k_nodes) is False:
                    return False
            else:
                # Returns False if every node was checked and thus stops the algorithm
                if n.set_parents_notkanon() is False:
                    return False
                if (nodes[-1].level - nodes[0].level) + 1 < 4:
                    for node in nodes:
                        if node.iskanon is None:
                            if evaluate([node], ac, topnode, min_k_nodes) is False:
                                return False
                    continue

                newnodes = createsubarray(n, nodes[-1], True)
                if newnodes is None:
                    continue

                sorted = sort(newnodes)
                if evaluate(sorted, ac, topnode, min_k_nodes) is False:
                    return False
    return True


def check_kanon(node, ac):
    if node.iskanon is not None:
        return False

    else:
        # Calculation if Node is k-anon
        if ac.calculate_kanon(node):
            node.iskanon = True
            return True
        else:
            node.iskanon = False
            return False


def createsubarray(bot, top, direction, only_create_subarray=False):
    """Creates a subpart of the generalization lattice

    :param bot: bot/start node from where the subpart starts
    :param top: top/end node where the subpart stops
    :param direction: Calculate from bot to top (True) or top to bot (False)
    :param only_create_subarray: Specifies if the function is used as utility or for the algorithm
    :return: list of nodes that are in the subpart of the generalization lattice
    """
    # Checks how to use this function
    if not only_create_subarray:
        # Check if subarry was already created before
        if top.id in bot.mem:
            return None
        # Mark that this subarray was already created
        bot.mem.append(top.id)

    # Initialize array with bot and top node
    subarray = [bot, top]
    # Create rest of the subarray
    iterate(subarray, bot, top, direction)

    return subarray


def iterate(list, bot, top, direction):
    if direction is True:
        attr_range = range(len(bot.attributes))
        for n in bot.childNodes:
            is_valid = True
            if n not in list:
                if n.level == top.level:
                    return True
                for num in attr_range:
                    if n.attributes[num] > top.attributes[num]:
                        is_valid = False
                        break
                if is_valid:
                    list.append(n)
                    iterate(list, n, top, direction)
        return True
    else:
        attr_range = range(len(top.attributes))
        for n in top.parentNodes:
            is_valid = True
            if n not in list:
                if n.level == bot.level:
                    return True
                for num in attr_range:
                    if n.attributes[num] < bot.attributes[num]:
                        is_valid = False
                        break
                if is_valid:
                    list.append(n)
                    iterate(list, bot, n, direction)
        return True
