# CayleyPy

AI-based library to work with extremely large graphs.<br>
Supporting:  Cayley graphs, Schreier coset graphs, more to be added.

## Overview

Extremely large graphs (e.g. googol size, ~10<sup>100</sup> nodes) cannot be approached in a usual way,
it is impossible neither to create, neither to store them by standard methods.

Typically such graphs arise as state-transition graphs.
For chess, Go or any other games - nodes of the graphs are positions, edges correspond to moves between them.
For Rubik's cube - nodes are configurations, edges corresponds to configurations different by single moves. 

The most simple and clear examples of such graphs - are [Cayley graphs](https://en.wikipedia.org/wiki/Cayley_graph) in mathematics.
(and [Schreier coset graphs](https://en.wikipedia.org/wiki/Schreier_coset_graph) ). 
Initial developments will focus on these graphs, supporting other types later. 

We plan to support:

* ML/RL methods for pathfinding 
* Estimation of diameters and growths
* Embeddings
* Efficient BFS for small subgraphs
* Efficient random walks generation
* Efficient Beam Search 
* Hamiltonian paths finding
* Efficient computing on CPU, GPU, TPU (with JAX), usable on Kaggle.
* Etc. 

Mathematical applications: 
* Estimation of diameters and growths
* Approximation of the word metrics and diffusion distance
* Estimation of the mixing time for random walks of different types 
* BFS from given state (growth function, adjacency matrix, last layers).
* Library of graphs and generators (LRX, TopSpin, Rubik Cubes, wreath, globe etc.,
  see [here](https://www.kaggle.com/code/ivankolt/generation-of-incidence-mtx-pancake)).
* Library of datasets with solutions to some problems (e.g. growth functions like
  [here](https://www.kaggle.com/code/fedimser/bfs-for-binary-string-permutations)).

## Examples

See the following Kaggle notebooks for examples of library usage:

* [Basic usage](https://www.kaggle.com/code/fedimser/cayleypy-demo) - defining Cayley graphs for permutation and matrix groups, running BFS, getting explicit Networkx graphs.
* [Computing spectra](https://www.kaggle.com/code/fedimser/computing-spectra-of-cayley-graphs-using-cayleypy).
* [Library of puzzles in GAP format in CayleyPy](https://www.kaggle.com/code/fedimser/library-of-puzzles-in-gap-format-in-cayleypy).
* Path finding in Cayley Graphs:
  * [Beam search with CayleyPy](https://www.kaggle.com/code/fedimser/beam-search-with-cayleypy) - simple example of finding paths for LRX (n=12) using beam search and neural network.
  * [Finding shortest paths for LRX (n=8) using BFS](https://www.kaggle.com/code/fedimser/lrx-solution).
  * [Finding shortest paths for LRX cosets (n=16 and n=32) using BFS](https://www.kaggle.com/code/fedimser/lrx-binary-with-cayleypy-bfs-only).
  * [Beam search with neural network for LRX cosets (n=32)](https://www.kaggle.com/code/fedimser/solve-lrx-binary-with-cayleypy).
  * [Beam search for LRX, n=16](https://www.kaggle.com/code/fedimser/lrx-solution-n-16-beamsearch). 
  * [Beam search for LRX, n=32](https://www.kaggle.com/code/fedimser/lrx-solution-n-32-beamsearch)
* Growth function computations:
  * [For LX](https://www.kaggle.com/code/fedimser/growth-function-for-lx-cayley-graph).
  * [For TopSpin cosets](https://www.kaggle.com/code/fedimser/growth-functions-for-topspin-cosets).
* Benchmarks:
  * [Benchmarks versions of BFS in CayleyPy](https://www.kaggle.com/code/fedimser/benchmark-versions-of-bfs-in-cayleypy).
  * [Benchmark BFS on GPU](https://www.kaggle.com/code/fedimser/benchmark-bfs-in-cayleypy-on-gpu-p100).

## Installation

We recommend installing the latest version from GitHub:

```
pip install git+https://github.com/cayleypy/cayleypy
```

You may also install using pip, although this might be missing recently added features:

```
pip install cayleypy
```

## Documentation

Documentation (API reference) for the latest version of the library is available
[here](https://cayleypy.github.io/cayleypy-docs/api.html).

## Development

To start development, run:

```
git clone https://github.com/cayleypy/cayleypy.git
cd cayleypy
pip install -e .[lint,test,dev,docs]
```

To run all tests, including some slow running tests:

```
RUN_SLOW_TESTS=1 pytest
```

Before committing, run these checks:

```
./lint.sh
pytest 
```

To check coverage, run:

```
coverage run -m pytest && coverage html
```

To rebuild documentation locally, run:

```
./docs/build_docs.sh 
```

### Formatting

This repository uses the [Black formatter](https://github.com/psf/black).
If you are getting error saying that some files "would be reformatted", you need to format
your code using Black. There are few convenient ways to do that:
* From command line: run `black .` 
* In PyCharm: go to Setting>Tools>Black, and check "Use Black formatter": "On code reformat" 
    (then it will run on Ctrl+Alt+L), or "On save", or both.
* In Visual Studio code: install the
    [Black Formatter extension](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter),
    then use Ctrl+Shift+I to format code. 
    If you are  asked to configure default formatter, pick the Black formatter.

### Style

* In general, this repository follows [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). All
    contributors should read it.
* When writing comments, [use punctuation](https://google.github.io/styleguide/pyguide.html#386-punctuation-spelling-and-grammar).
    In particular, always put a period (".") in the end of sentences.
* We have pylint checks to enforce some style rules. You should fix pylint warnings instead of disabling the check. 

## How to add a new Cayley graph

Cayley graphs must be defined by a function that returns `CayleyGraphDef`. 
First, you need to decide where in the library to put it:
* If it's a graph generated by permutations, the function should be added to  
    `PermutationGroups` in `cayleypy/graphs_lib.py`, annotated as `@staticmethod`.
* If it's a graph generated by matrices, the function should be added to  
    `MatrixGroups` in `cayleypy/graphs_lib.py`.
* If it's a graph for a physical puzzle, the function should be added to 
    `Puzzles` in `caylepy/puzzles/puzzles.py`. If it requires non-trivial construction,
    move that to separate function(s) and put them in separate file in `cayleypy/puzzles`.
    If the puzzle is defined by hardcoded permutations, put them in `cayleypy/puzzles/moves.py`. 
* If it's a graph for a puzzle, and you have definition in GAP format, put the `.gap` file in
    `puzzles/gap_files/default`. It will become available via `cayleypy.GapPuzzles`.
* If it's a new type of graph, check with @fedimser where to put it.

Do not add new graphs to `prepare_graph`! We want new graphs to be added in different 
places to avoid merge conflicts.

Then, you need to define your graph. Definition consists of the following:
* Generators.
* Generator names (optional).
* Central state (optional, defaults to neutral element in the group, e.g. 
    identity permutation).

When you are ready do the following: 
1. Create a fork of this repository.
2. Clone your fork and make a new bracnch in your fork.
3. Add your function to an approritate place. See how other graphs are defined and follow that as an example.
4. Write a docstring for your function, describing your graph. If possible, include reference
   (e.g. to Wikipedia article, Arxiv paper or a book) where the graph is defined.
5. Add a test that creates an instance of your graph for small size and checks something about it 
     (at least check number of generators).
6. Create a pull request from your fork to the this repository. 
   Choose PR name reflecting your changes (e.g. "Update graphs_lib" is a poor choice, while "Add LRX groups to graphs_lib" is a better one). 
7. Make sure that all automated checks (like unittests and style checks) are passed.
8. Get two approvals from the reviewers team. They might make comments or request changes before
   provding an approval -- implement those. If needed, discuss how to proceed. If a commit
   is pushed after the approval, the approval has to be obtained again. 
9. Once approvals are granted -- merge the PR. 

For more details on how to make PRs with forks refer to [this guide](https://graphite.dev/guides/create-and-manage-pull-requests-from-fork).

## Predictor models

CayleyPy contains a library of machine learning models to be used as predictors in the beam search algorithm for
finding paths in Cayley graph. These models can be easily accessed using `Predictor.pretrained`
([example](https://www.kaggle.com/code/fedimser/lrx-solution-n-32-beamsearch)).

Each such model is a PyTorch neural network which consists of 3 parts: 
* Model architecture description (a subclass of `nn.Models`) - defined in `cayleypy/models/`: "MLP" and "RESMLP" in
  `models.py`, "TRANSFORMER" (which consumes states as tokens, see `GroupTokenizer`) in `transformer.py`, and "QV" (a
  Q-head and a V-head over any of the others, given by `backbone_type`) in `qv_model.py`.
* Model architecture hyperparameters (such as input size or sizes of hidden layers) - defined by `models.ModelConfig`.
* Model weights - these are stored on Kaggle.

A model has either one output - an estimate of the distance of the state it is applied to - or one output per generator
of the graph (a Q-model), which estimates distances of all children of a state in one forward pass. Beam search uses a
Q-model when called with `use_child_scores=True`, see `Predictor.score_children`.

List of currently available models is 
[here](https://github.com/cayleypy/cayleypy/blob/main/cayleypy/models/models_lib.py).

### Beam search options

Beam search has several options that make a predictor go further, all of them available with `beam_mode="simple"`
(the default) only:
* `use_child_scores=True` - score all children of the beam in one pass, which is what a Q-model is for (with a
  single-output model it changes nothing but the order of evaluations).
* `non_backtracking=True` - never apply the move that undoes the move a state was reached by. Needs generators that are
  inverse closed.
* `canonical_dedup=SymmetryGroup(...)` - keep one state per orbit of the symmetries of the graph, which is like
  multiplying the beam width by the number of symmetries (`SymmetryGroup.reflections` and
  `SymmetryGroup.rubik_cube_rotations` are ready to use, `SymmetryGroup.derive` finds them for a small graph).
* `lower_bound=BfsLowerBound(...)` together with `prune_above=k` - drop states that cannot be on a path of length at
  most `k`, using exact distances of the states near the central state.

Independently of the search, `SymmetrizedPredictor` averages a predictor over symmetries of the graph (test-time
augmentation), which reduces the error of an imperfect model at the price of one model evaluation per symmetry.

### Training models

Models can be trained with `cayleypy.train`. The snippet below is the whole way from a model config to a solved state:
train a Q-model on random walks, save it as a self-describing checkpoint, fine-tune it on Bellman targets, ensemble the
two checkpoints and search with them. Steps 1-3 are exactly how the "lrx-14" model of this library was trained (about 3
minutes on a CPU, plus about 1.5 minutes for the fine-tuning).

```python
import torch

from cayleypy import CayleyGraph, EnsemblePredictor, PermutationGroups, Predictor
from cayleypy.models import ModelConfig, load_checkpoint
from cayleypy.train import BellmanTrainer, TrainConfig, Trainer

graph = CayleyGraph(PermutationGroups.lrx(14), device="cpu", random_seed=42)

# 1. Describe the model: one output per generator (a Q-model), so beam search scores all children in one pass.
model_config = ModelConfig(
    model_type="RESMLP",
    input_size=14,
    num_classes_for_one_hot=14,
    layers_sizes=[512, 512, 512],
    n_outputs=graph.definition.n_generators,
)

# 2. Train it on random walks, mixing in a few per cent of states whose distance is known exactly.
train_config = TrainConfig(
    n_epochs=200,
    n_walks=1024,
    rw_length=92,  # Diameter of this graph is 91.
    batch_size=1024,
    lr=1e-3,
    lr_min=1e-5,
    ema_decay=0.999,
    anchors_depth=6,  # Breadth-first search of this depth gives the states with exact distances.
    anchors_fraction=0.02,
    seed=42,
    verbose=1,
)
trainer = Trainer(graph, model_config, train_config)
trainer.train()

# 3. Save a self-describing checkpoint: the config and the hash of the graph are stored next to the weights.
trainer.save("lrx_14_q_resmlp.pt")

# 4. Fine-tune on Bellman targets "1 + smallest predicted distance of a child", with a lower learning rate.
finetune_config = TrainConfig(
    n_epochs=100,
    n_walks=1024,
    rw_length=92,
    batch_size=1024,
    lr=1e-4,
    lr_min=1e-6,
    ema_decay=0.999,
    bellman_anchors_depth=6,  # Bellman targets need anchors: they are what keeps the scale from drifting.
    anchors_fraction=0.02,
    seed=42,
    verbose=1,
)
finetuner = BellmanTrainer.from_checkpoint("lrx_14_q_resmlp.pt", graph, finetune_config)
finetuner.train()
finetuner.save("lrx_14_q_resmlp_bellman.pt")

# 5. Ensemble the two checkpoints - they make different errors, so their weighted sum is a better estimate.
members = [
    # load_checkpoint returns the model and the config it was saved with.
    Predictor(graph, load_checkpoint(path, graph_def=graph.definition)[0])
    for path in ["lrx_14_q_resmlp.pt", "lrx_14_q_resmlp_bellman.pt"]
]
predictor = EnsemblePredictor(members, [0.3, 0.7])

# 6. Search: all children of the beam are scored in one pass, and moves undoing the previous move are banned.
result = graph.beam_search(
    start_state=torch.randperm(14),
    predictor=predictor,
    beam_width=100,
    use_child_scores=True,
    non_backtracking=True,
    return_path=True,
)
print("Path found:", result.path_found, "length:", result.path_length)
```

The snippet trains on random walks with BFS anchors, which is what `TrainConfig` builds by default. Other sources of
training data can be passed as `Trainer(..., data_source=...)`:

* `PathDataSource` turns paths that are already known into targets for states nothing else reaches - a beam search
  result obtained with `return_path=True`, or the published solutions of a competition
  (`PathDataSource.from_tsv` reads the format of the "cayleypy-beam-results" repository). Its `weight` is what tells the
  loss that these targets are upper bounds and deserve less trust than exact distances.
* `MixtureDataSource` combines any sources in given proportions, which is how walks and anchors are mixed above.
* `BfsAnchors` alone gives exact distances near the central state.

`TrainConfig(loss="pinball", tau=...)` trains a quantile of the target distribution instead of its mean - `tau` below
0.5 makes the model predict low, i.e. closer to a lower bound on the distance.

On 50 uniformly random permutations of 14 elements with beam width 100, this ensemble finds a path for all 50 of them
(mean path length 63.4, diameter of the graph is 91), while the fine-tuned checkpoint alone solves 49 and the checkpoint
trained on walks alone solves 47; with beam width 300 all three solve all 50. The Hamming distance heuristic solves none
of them even with beam width 1000.

### How to add a new predictor model
1. Train your model.
2. Verify that when used with beam search, it reliably finds the paths.
3. Export weights to a file - either as a bare state dict (`torch.save(model.state_dict(), path)`) or as a
    self-describing checkpoint (`models.save_checkpoint`, which is also what `train.Trainer.save` writes). Both are
    accepted by `ModelConfig.load`.
4. Upload weights as model on Kaggle, make it public and use open source license (MIT license is recommended).
5. Make sure the graph for which your model should be used has unique name (that is, `CayleyGraphDef.name`). For
    example, `PermutationGroups.lrx(16)` has name "lrx-16". Also `prepare_graph` given this name should return
    this graph (this is needed for tests).
6. Define `ModelConfig` for your model:
    * `weights_kaggle_id` is identifier of your saved model on Kaggle. This is what you would pass to 
      `kagglehub.model_download`.
    * `weights_path` is the name of file with weights.
    * `graph_hash` is hash of the graph your model was trained for (see `models.graph_hash`; a checkpoint written by
        `train.Trainer.save` already stores it). Set it, and loading your model for a graph that has the expected name
        but another definition will fail instead of silently returning nonsense.
    * If your can be exactly described by one of available model types (see the list of architectures above), use that
        model type with appropriate hyperparameters. If needed, add new hyperparameters to ModelConfig.
    * If your model architecture is very different from we already have in library, define new model type for it.
    * For example, we already have model type "MLP" (multi-layer perceptron) defined by `MlpModel` with the following
        parameters: `input_size`, `num_classes_for_one_hot`, `layers_sizes`.
7. Verify that when you define your model config, call `load` on it and then use that as predictor in beam search,
    it works.
8. Add your model to `PREDICTOR_MODELS` in `models_lib`. Use graph name as a key.
9. Run `pytest cayleypy/models/models_lib_test.py`. This will check that your model can be loaded from Kaggle and used
    for inference (i.e. has correct input and output shape), but it doesn't check quality of your model.
9. Optionally, add a test that beam search with your model successfully finds a path.

## Kaggle competitions

Our community has recently launched several Kaggle competitions to develop and benchmark our methods and make it easier for the larger audience to get involved. Most of those competitions don't require the usage of this library, however it might be handy as those tasks are the ones 


| Competition   | Graph size(s) | Description |
| --------------| ------------- | ------------|
| [4x4x4 Cube](https://www.kaggle.com/competitions/cayley-py-444-cube/)                                    | 10<sup>55</sup>                 | 4x4x4 Rubik's cube (play [here](https://alpha.twizzle.net/explore/?puzzle=4x4x4))                                                  |
| [5x5x5 Cube](https://www.kaggle.com/competitions/cayley-py-555-cube/)                                    | 10<sup>92</sup>                 | 5x5x5 Rubik's cube (play [here](https://alpha.twizzle.net/explore/?puzzle=5x5x5))                                                  |
| [6x6x6 Cube](https://www.kaggle.com/competitions/cayley-py-666-cube/)                                    | 10<sup>150</sup>                 | 6x6x6 Rubik's cube (play [here](https://alpha.twizzle.net/explore/?puzzle=6x6x6))                                                  |
| [7x7x7 Cube](https://www.kaggle.com/competitions/cayley-py-777-cube/)                                    | 10<sup>210</sup>                 | 7x7x7 Rubik's cube (play [here](https://alpha.twizzle.net/explore/?puzzle=7x7x7))                                                  |
| [Christopher's Jewel](https://www.kaggle.com/competitions/cayleypy-christophers-jewel/)                  | 10<sup>16<sup>                  | small octahedron-shaped puzzle (play [here](https://alpha.twizzle.net/explore/?puzzle=Christopher%27s+jewel))                      |
| [Megaminx](https://www.kaggle.com/competitions/cayley-py-megaminx)                                       | 10<sup>69<sup>                  | dodecahedron-shaped puzzle (play [here](https://alpha.twizzle.net/explore/?puzzle=megaminx))                                       |
| [Professor Tetraminx](https://www.kaggle.com/competitions/cayley-py-professor-tetraminx-solve-optimally) | 10<sup>32</sup>                 | tetrahedron-shaped puzzle, medium size version (play [here](https://alpha.twizzle.net/explore/?puzzle=professor+tetraminx))        |
| [IHES Supercube](https://www.kaggle.com/competitions/cayleypy-ihes-cube)                                 | 10<sup>24</sup>                 | a variant of 3×3×3 Rubik’s cube with oriented faces                                                                                |
| [RapaportM2](https://www.kaggle.com/competitions/cayleypy-rapapport-m2/)                                 | (tba)                           | pairs swaps                                                                                                                        |
| [Transposons](https://www.kaggle.com/competitions/cayleypy-transposons/)                                 | 10<sup>8</sup>-10<sup>158</sup> | transpositions of adjacent substrings                                                                                              |
| [Reversals](https://www.kaggle.com/competitions/cayleypy-reversals)                                      | 10<sup>8</sup>-10<sup>64</sup>  | substring reversals                                                                                                                |
| [Pancake sorting](https://www.kaggle.com/competitions/CayleyPy-pancake)                                  | (tba)                           | string prefix reversals                                                                                                            |
| [Glushkov's problem](https://www.kaggle.com/competitions/cayleypy-glushkov/)                             | (tba)                           | left shifts and first two elementstranspositions                                                                                   |


## Credits

The idea of the project - Alexander Chervov - see 
[arXiv:2502.18663](https://arxiv.org/abs/2502.18663), 
[arXiv:2502.13266](https://arxiv.org/abs/2502.13266),
[arxiv:2509.19162](https://arxiv.org/abs/2509.19162),
discussion group https://t.me/sberlogasci/1,
Early ideas and prototypes appeared during Kaggle challenge Santa 2023:
Prototype: https://www.kaggle.com/code/alexandervc/santa23-globe26-modeling5,
Description: https://www.kaggle.com/competitions/santa-2023/discussion/466399, 
https://www.kaggle.com/competitions/santa-2023/discussion/472594. 

The initial code developments can be found at Kaggle dataset:
https://www.kaggle.com/datasets/alexandervc/growth-in-finite-groups (see paper https://arxiv.org/abs/2502.13266 )
Other developments can be found at:
https://www.kaggle.com/competitions/lrx-oeis-a-186783-brainstorm-math-conjecture/code,
https://www.kaggle.com/datasets/alexandervc/cayleypy-development-3-growth-computations,
see also beam-search part: [ Cayleypy (Ivan Koltsov) ](https://github.com/iKolt/cayleypy),
Rubik's cube part: [Piligrim (Kirill Khoruzhii)](https://github.com/k1242).

Also, code from the following Kaggle notebooks was used:

* https://www.kaggle.com/code/ivankolt/generation-of-incidence-mtx-pancake (advanced BFS).
* https://www.kaggle.com/code/avm888/cayleypy-growth-function.
* https://www.kaggle.com/code/avm888/jax-version-cayleypy (how to use JAX).
* https://www.kaggle.com/code/fedimser/bfs-for-binary-string-permutations (bit operations).
* https://www.kaggle.com/code/ivankolt/lrx-4bit-uint64?scriptVersionId=221435319 (fast BFS)
