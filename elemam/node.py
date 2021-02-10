# -*- coding: utf-8 -*-

id = 0


class Node:
    def __init__(self, attributes: list, node_list: list, max_gen_level: list, level_nodes: dict):
        """
        :param attributes: List containing the generalization levels of the QI
        :param node_list: Global node list containing a reference to every node
        :param max_gen_level: Global list cointaining the maximum generalization levels of the QI
        :param level_nodes: Dictionary grouping the Nodes by height in the generalization lattice
        """
        self.ismarked = False
        self.ismarkedK = False
        self.attributes = attributes
        self.level = sum(attributes)
        self.iskanon = None
        self.childNodes = []
        self.parentNodes = []
        global id
        self.id = id
        self.eqclasses = 0
        self.DM_penalty = 0
        self.DMs_penalty = 0

        self.mem = []

        self.level_nodes = level_nodes

        attribute_count = len(self.attributes)

        # If this is the first node set defaults
        if len(node_list) == 0:
            id = 0
            self.id = id
            node_list.append(self)
            self.iskanon = False

        # Calculate precision of the Node
        self.prec = 0
        for a in range(attribute_count):
            self.prec += (self.attributes[a] / (max_gen_level[a]))
        self.prec /= attribute_count

        # Loop every QI level to create child nodes (generalizations of the QI)
        for x in range(attribute_count):
            # Check if QI can't be further generalized
            if not self.attributes[x] == max_gen_level[x]:
                # Create QI level values for new node
                next_attributes = self.attributes.copy()
                next_attributes[x] += 1
                next_level = self.level+1

                node = None
                # Check if height in generalzation lattice was already reached
                if next_level in level_nodes:
                    # Check if new node exists already
                    # if not:node=None; if yes:node=that node
                    node = next((n for n in level_nodes[next_level] if n.attributes == next_attributes), None)

                if node is None:
                    # Increase global ID counter
                    id += 1

                    # Create new Node
                    node = Node(next_attributes, node_list, max_gen_level, level_nodes)

                    # Put new node into the dictionary
                    try:
                        level_nodes[next_level].append(node)
                    except KeyError:
                        level_nodes.update({next_level: [node]})

                    # Add node to the global node list
                    node_list.append(node)

                # Add new node to the child nodes of this node
                self.childNodes.append(node)
                # Add this node to the parent nodes of the new node
                node.parentNodes.append(self)

    def set_children_kanon(self):
        # Only continue if the node wasn't already marked
        if self.ismarkedK is False:
            self.iskanon = True
            self.ismarkedK = True
            global id
            id -= 1
            if id < 0:
                return False

            for node in self.childNodes:
                if node.ismarkedK is True:
                    continue
                if node.set_children_kanon() is False:
                    return False
        return True

    def set_parents_notkanon(self):
        # Only continue if the node wasn't already marked
        if self.ismarked is False:
            self.iskanon = False
            self.ismarked = True
            global id
            id -= 1
            if id < 0:
                return False

            for node in self.parentNodes:
                if node.ismarked is True:
                    continue
                if node.set_parents_notkanon() is False:
                    return False
        return True
