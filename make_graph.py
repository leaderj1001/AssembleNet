import random


class Graph(object):
    def __init__(self,
                 dilation=(1, 2, 4, 8),
                 m=(1.5, 2, 3, 1.5),
                 hidden_size=(
                     [2, 4, 8, 16],
                     [8, 16, 32, 64],
                     [8, 16, 32, 64],
                     [8, 16, 32, 64],
                 ),
                 max_level=4,
                 per_n_nodes=4):
        self.max_level = max_level
        self.per_n_nodes = per_n_nodes

        self.graph = {}

        self.dilation = dilation
        assert len(self.dilation) == max_level, 'error: make the dilation length and max_level the same.'
        # self.hidden_size = [
        #     [32, 64, 96, 128],
        #     [128, 256, 384, 512],
        #     [128, 256, 384, 512],
        #     [128, 256, 384, 512]
        # ]
        self.hidden_size = hidden_size
        assert len(self.hidden_size) == max_level, 'error: make the hidden_size length and max_level the same.'
        self.m = m
        assert len(self.m) == max_level, 'error: make the m length and max_level the same.'

        self.empty_edge_nodes = []

        self._make_graph()

    def _make_graph(self):
        for lv in range(1, self.max_level + 1):
            random_out_channels = self._generate_random_channels(lv)
            for node in range(1, self.per_n_nodes + 1):
                node_num = lv * 4 + node
                self.graph[node_num] = {}
                self.graph[node_num]['channels'] = random_out_channels
                self.graph[node_num]['m'] = self.m[lv - 1]
                self.graph[node_num]['level'] = lv
                self._generate_random_edges(lv, node_num)
                self._generate_random_dilation(lv, node_num)

    def _generate_random_edges(self, level, node_num):
        candidate_nodes = [(i - 1) * 4 + j for i in range(1, level + 1) for j in range(1, self.per_n_nodes + 1)]
        self.graph[node_num]['edges'] = []
        self.graph[node_num]['others'] = []
        self.graph[node_num]['n_totals'] = len(candidate_nodes)
        for _ in candidate_nodes:
            rand_prob = random.random()
            if rand_prob > 0.5 and _ not in self.empty_edge_nodes:
                self.graph[node_num]['edges'].append(_)
            else:
                self.graph[node_num]['others'].append(_)

        # handling empty connected edges
        if len(self.graph[node_num]['edges']) == 0:
            self.empty_edge_nodes.append(node_num)

    def _generate_random_channels(self, level):
        random_out_channels = self.hidden_size[level - 1][random.randint(0, len(self.hidden_size[level - 1]) - 1)]
        return random_out_channels

    def _generate_random_dilation(self, level, node_num):
        random_dilation = self.dilation[random.randint(0, len(self.dilation) - 1)]
        self.graph[node_num]['dilation'] = random_dilation


def main():
    import pprint
    p = pprint.PrettyPrinter(width=160, indent=4)
    g = Graph()

    p.pprint(g.graph)

    pass


# if __name__ == '__main__':
#     main()
