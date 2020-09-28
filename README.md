# Implementing AssembleNet: Searching for Multi-Stream Neural Connectivity in Video Architectures Explain using Pytorch (Work In Process)

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
  - Network Evolution
  ```
  from make_graph import Graph
  import pprint

  g = Graph()
  m = Model(g.graph)
  pprint.pprint(m.graph, width=160)
  m._evolution()
  pprint.pprint(m.graph, width=160)

  x = torch.randn([2, 3, 16, 256, 256])
  print(m(x).size())
  ```
## Evolution
  <img width="1080" alt="스크린샷 2020-09-28 오후 12 50 18" src="https://user-images.githubusercontent.com/22078438/94390762-92836300-018e-11eb-852b-845d7294f31e.png">

## Work In Process
  - Connection-Learning-Guided Mutation
  - Evolution
  - Training
