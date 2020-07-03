import random


class Graph(object):
    def __init__(self, max_level=4, per_n_nodes=4):
        self.max_level = max_level
        self.per_n_nodes = per_n_nodes

        self.graph = {}

        self.dilation = [1, 2, 4, 8]
        self.hidden_size = [
            [32, 64, 96, 128],
            [128, 256, 384, 512],
            [128, 256, 384, 512],
            [128, 256, 384, 512]
        ]

        self._make_graph()

    def _make_graph(self):
        for lv in range(1, self.max_level + 1):
            self.graph[lv] = {}
            random_out_channels = self._generate_random_channels(lv)
            for node in range(1, self.per_n_nodes + 1):
                node_num = (lv - 1) * 4 + node
                self.graph[lv][node_num] = {}
                self.graph[lv][node_num]['channels'] = random_out_channels
                self._generate_random_edges(lv, node_num)
                self._generate_random_dilation(lv, node_num)

    def _generate_random_edges(self, level, node_num):
        candidate_nodes = [(i - 1) * 4 + j for i in range(1, level + 1) for j in range(1, self.per_n_nodes + 1)]
        self.graph[level][node_num]['edges'] = []
        for _ in candidate_nodes:
            if random.random() > 0.5:
                self.graph[level][node_num]['edges'].append(_)

    def _generate_random_channels(self, level):
        random_out_channels = self.hidden_size[level - 1][random.randint(0, len(self.hidden_size[level - 1]) - 1)]
        return random_out_channels

    def _generate_random_dilation(self, level, node_num):
        random_dilation = self.dilation[random.randint(0, len(self.dilation) - 1)]
        self.graph[level][node_num]['dilation'] = random_dilation


def main():
    import pprint
    p = pprint.PrettyPrinter(indent=4)
    g = Graph()

    p.pprint(g.graph)

    pass


# if __name__ == '__main__':
#     main()
