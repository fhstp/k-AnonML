#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import sys

from elemam.algorithm import calculate, sort
from elemam.kanon import AnonCheck
from elemam.node import Node

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.data import transform_columns


def create_anon_data(best_nodes, raw_data, qi_index, gen_strat, kanon, res_folder):
    """Create anonymized data in 2d list format

    :param qi_index: List of QI column indexes
    :param best_node: Node that is used to anonymized the data
    :param raw_data: Raw data read from the csv file
    :param gen_strat: List containing the generalization strategies for the QI
    :return: 2d list: anon_data[row][col]
    """
    anon_data_list = {}
    for node in best_nodes:
        anon_data = [[0 for _ in range(len(raw_data))] for _ in range(len(raw_data[0]))]
        for row in range(len(raw_data[0])):
            gen_strat_iter = iter(gen_strat)
            attributes_iter = iter(node.attributes)
            for col in range(len(raw_data)):
                raw_value = raw_data[col][row]

                if col in qi_index:
                    attribute = next(attributes_iter)
                    if attribute == 0:
                        next(gen_strat_iter)
                        anon_data[row][col] = raw_value
                        continue
                    strat = next(gen_strat_iter)

                    level = attribute - 1

                    # Generate the anonymized value
                    if isinstance(strat, list):
                        args = []
                        for arg_len in range(1, len(strat)):
                            args.append(strat[arg_len])
                        vg = strat[0](raw_value, level, *tuple(args))
                    else:
                        vg = strat(raw_value, level)
                else:
                    vg = [raw_value]

                anon_data[row][col] = vg[0]

        eq_classes_dict = {}
        for row in range(len(raw_data[0])):

            key = tuple(anon_data[row][col] for col in qi_index)
            try:
                eq_classes_dict[key][0] += 1
                eq_classes_dict[key][1].append(row)
            except KeyError:
                eq_classes_dict.update({key: [1, [row]]})

        suppressed_count = 0
        suppressed_rows = []
        for vals in eq_classes_dict.values():
            if vals[0] < kanon:
                suppressed_count += vals[0]
                for row in vals[1]:
                    suppressed_rows.append(row)
                    for col in qi_index:
                        anon_data[row][col] = "*"

        print("Suppressed: " + str(suppressed_count) + " rows")
        writer = csv.writer(open(os.path.join(res_folder, "supprarray.csv"), "a+"))
        writer.writerow(str(suppressed_count))
        anon_data_list.update({tuple(node.attributes): anon_data})
    return anon_data_list


def main(raw_data, kanon, gen_strat, max_gen_level, qi_index, metric, res_folder, suppression_rate=0):
    raw_data = transform_columns(raw_data)
    qi_data = [raw_data[i] for i in qi_index]

    # Get number of QI
    quasi_ident_count = len(qi_index)

    # Define the suppression limit
    allowed_suppressed = int(len(qi_data[0]) * (suppression_rate / 100))

    # Create object for checking anonymity
    ac = AnonCheck(qi_data, max_gen_level, gen_strat, allowed_suppressed, kanon)

    node_array = []
    level_nodes = {}

    # Creating Node-structure starting with root-node
    rootnode = Node([0] * quasi_ident_count, node_array, max_gen_level, level_nodes)

    # Sorting list by height in the generalization lattice
    sorted_array = sort(node_array)

    # Evaluate generalization hierarchy with the OLA(ELEmam) algorithm
    # Returns lowest Nodes in a generalization path
    min_k = calculate(sorted_array, ac)

    # Calculation the best Node by the percision metric
    prec = 1
    lvl = 999999999
    eqcount = 9999999
    penalty = None
    loss = 999999999
    best_nodes = []

    if metric == "ent":
        test = create_anon_data(min_k, raw_data, qi_index, gen_strat, kanon, res_folder)
    for node in min_k:
        if metric == "prec":
            if node.prec < prec:
                prec = node.prec
                best_nodes = [node]
        elif metric == "gweight":
            if node.level < lvl:
                lvl = node.level
                best_nodes = [node]
        elif metric == "aecs":
            if node.eqclasses == eqcount:
                print("Bad")
            if node.eqclasses != 0 and node.eqclasses < eqcount:
                eqcount = node.eqclasses
                best_nodes = [node]
        elif metric == "dm":
            if penalty is None or node.DM_penalty < penalty:
                penalty = node.DM_penalty
                best_nodes = [node]
        elif metric == "dms":
            if node.DMs_penalty < penalty:
                penalty = node.DMs_penalty
                best_nodes = [node]
        elif metric == "ent":
            print("Metric ENT")
            new_loss = 0
            dictarray_r = []
            dictarray_g = []
            for col in range(len(qi_data)):
                d_r = {}
                d_g = {}
                for row in range(len(qi_data[0])):
                    i_r = qi_data[col][row]
                    i_g = test[tuple(node.attributes)][row][qi_index[col]]
                    if i_r in d_r:
                        d_r[i_r] += 1
                    else:
                        d_r.update({i_r: 1})
                    if i_g in d_g:
                        d_g[i_g] += 1
                    else:
                        d_g.update({i_g: 1})
                dictarray_r.append(d_r)
                dictarray_g.append(d_g)
            for k in range(len(qi_data[0])):
                for j in range(len(qi_data)):
                    r_val = qi_data[j][k]
                    g_val = test[tuple(node.attributes)][k][qi_index[j]]
                    r_count = dictarray_r[j][r_val]
                    g_count = dictarray_g[j][g_val]
                    new_loss += math.log(r_count / g_count, 2)
            if new_loss < loss:
                loss = new_loss
                best_nodes = [node]

    # Print best generalisation
    print(best_nodes[0].attributes)
    with open(os.path.join(res_folder, 'genarray.csv'), 'a+') as gen_file:
        writer = csv.writer(gen_file)
        writer.writerow(best_nodes[0].attributes)
    if metric == "none":
        best_nodes = min_k
    return create_anon_data(best_nodes, raw_data, qi_index, gen_strat, kanon, res_folder), best_nodes[0].attributes
