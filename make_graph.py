import random


class Node():
    def __init__(self, level, max_level, max_node):
        self.edges = []
        self.candidate_nodes = [(i - 1) * 4 + j for i in range(level + 1, max_level + 1) for j in range(1, max_node + 1)]

        self._generate_random_edges()

    def _generate_random_edges(self):
        for _ in self.candidate_nodes:
            if random.random() > 0.5:
                self.edges.append(_)


node = Node(2, 4, 4)
print(node.edges)


def main():
    level, n_nodes = 4, 4
    dict1 = {}

    for i in range(4):
        dict1[i] = []
        for j in range(n_nodes):
            dict1[i].append(j)


if __name__ == '__main__':
    main()
