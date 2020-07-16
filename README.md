# Implementing AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures Explain using Pytorch

## Reference
  ### This repo is not official repository.
  - [Paper Link](https://arxiv.org/abs/1905.13209)
  - Author: Michael S. Ryoo (Robotics at Google, Google Research), AJ Piergiovanni (Robotics at Google, Google Research), Mingxing Tan (Google Research), Anelia Angelova (Robotics at Google, Google Research)
  - Organization: Robotics at Google, Google Research
  
## Usage
  - Make graph
  ```python
  from make_graph import Graph
  import pprint

  p = pprint.PrettyPrinter(width=160, indent=4)
  g = Graph()
  p.pprint(g.grpah)
  ```
  - Make Network
  ```python
  from make_graph import Graph
  import pprint

  g = Graph()
  m = Model(g.graph)
  pprint.pprint(m.graph, width=160)

  x = torch.randn([2, 3, 16, 256, 256])
  print(m(x).size())

  ```
## Work In Process
