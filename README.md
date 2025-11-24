# Breaking the Sorting Barrier: Shortest Path Algorithm Implementation

Implementation of the groundbreaking 2024-2025 shortest path algorithm by Ran Duan et al. that breaks the 40-year-old sorting barrier, achieving O(m log^(2/3) n) time complexity for directed graphs with non-negative weights.

## Overview

This repository provides a comprehensive implementation of the algorithm described in "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" (STOC 2025 Best Paper). The algorithm surpasses Dijkstra's O(m + n log n) bound, which stood as the best known for 65 years.

## Key Features

- **Faster than Dijkstra**: O(m log^(2/3) n) vs O(m + n log n)
- **Deterministic**: No randomization required
- **General graphs**: Works on directed and undirected graphs
- **Real weights**: Handles real non-negative edge weights
- **Comprehensive testing**: Extensive verification against classical algorithms

## Algorithm Components

1. **Frontier Clustering**: Groups neighboring nodes to reduce comparisons
2. **Selective Bellman-Ford**: Limited iterations to identify influential nodes
3. **Layer Management**: Processes graph in layers without strict ordering
4. **Recursive Partitioning**: Adaptive problem division

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/shortest-paths.git
cd shortest-paths

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.core.graph import Graph
from src.algorithms.barrier_breaker import BarrierBreakerSSSP
from src.algorithms.dijkstra import DijkstraSSSP

# Create a graph
graph = Graph(directed=True)
graph.add_edge(0, 1, weight=4)
graph.add_edge(0, 2, weight=2)
graph.add_edge(1, 2, weight=1)
graph.add_edge(1, 3, weight=5)
graph.add_edge(2, 3, weight=8)

# Run the new algorithm
bb_sssp = BarrierBreakerSSSP(graph)
distances_new = bb_sssp.compute(source=0)

# Compare with Dijkstra
dijkstra = DijkstraSSSP(graph)
distances_classic = dijkstra.compute(source=0)

print(f"New Algorithm: {distances_new}")
print(f"Dijkstra: {distances_classic}")
assert distances_new == distances_classic  # Same results, faster computation
```

## Project Structure

```
shortest-paths/
├── src/
│   ├── core/           # Core data structures
│   ├── algorithms/     # Algorithm implementations
│   └── utils/          # Helper utilities
├── tests/              # Test suite
├── benchmarks/         # Performance analysis
└── examples/           # Usage examples and demos
```

## Performance Characteristics

### Theoretical Complexity
- **New Algorithm**: O(m log^(2/3) n) - proven asymptotic improvement
- **Dijkstra (binary heap)**: O(m log n)
- **Dijkstra (Fibonacci heap)**: O(m + n log n)

### Practical Performance

The algorithm is designed for **very large sparse graphs** (n > 100,000 nodes). Due to implementation overhead from clustering, layer management, and selective Bellman-Ford operations, the constants hidden in big-O notation are substantial.

**Expected behavior by graph size:**

| Graph Size | Expected Performance |
|-----------|---------------------|
| n < 10,000 | Dijkstra faster (lower overhead) |
| 10,000 ≤ n < 50,000 | Comparable performance |
| n ≥ 50,000 | Barrier Breaker shows advantages |
| n ≥ 1,000,000 | Significant speedup expected |

The algorithm adaptively disables advanced features on smaller graphs to minimize overhead while maintaining correctness. This implementation prioritizes correctness and educational value over raw performance optimization.

*Note: The theoretical breakthrough is real and significant, but practical speedups require massive graphs and further implementation optimization.*

## Testing

```bash
# Run all tests
pytest tests/

# Run correctness tests
pytest tests/test_correctness.py

# Run performance benchmarks
python benchmarks/run_benchmarks.py
```

## Documentation

- [Algorithm Analysis](ALGORITHM_ANALYSIS.md) - Comprehensive theoretical analysis
- [Implementation Details](docs/IMPLEMENTATION.md) - Technical implementation notes
- [API Reference](docs/API.md) - Complete API documentation

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting PRs.

## Citations

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  booktitle={Symposium on Theory of Computing (STOC)},
  year={2025}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, and Longhui Yin for the groundbreaking algorithm
- STOC 2025 for recognizing this work with the Best Paper Award
- The broader graph algorithms research community

## References

- [New Method Is the Fastest Way To Find the Best Routes (2025)](https://www.quantamagazine.org/new-method-is-the-fastest-way-to-find-the-best-routes-20250806/)
- [Original Paper (2025)](https://arxiv.org/abs/2504.17033)
- [Undirected Algorithm (2023)](https://arxiv.org/abs/2307.04139)
- [Related Work by Duan (2018)](https://arxiv.org/abs/1808.10658)