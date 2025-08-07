import sys
import os
import time
import random
import math
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.graph import Graph, WeightedGraph
from src.algorithms.dijkstra import DijkstraSSSP, DijkstraFibonacci
from src.algorithms.bellman_ford import BellmanFordSSSP
from src.algorithms.barrier_breaker import BarrierBreakerSSSP


@dataclass
class BenchmarkResult:
    algorithm: str
    graph_type: str
    num_nodes: int
    num_edges: int
    execution_time: float
    relaxations: int
    heap_operations: int
    nodes_visited: int
    correctness_verified: bool
    speedup_vs_dijkstra: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)


class GraphGenerator:
    @staticmethod
    def generate_sparse_graph(n: int, density: float = 2.0) -> Graph:
        m = int(n * density)
        graph = Graph(directed=True)
        
        for i in range(n - 1):
            weight = random.uniform(1, 10)
            graph.add_edge(i, i + 1, weight)
        
        edges_added = n - 1
        attempts = 0
        max_attempts = m * 10
        
        while edges_added < m and attempts < max_attempts:
            src = random.randint(0, n - 1)
            dst = random.randint(0, n - 1)
            attempts += 1
            
            if src != dst and not graph.has_edge(src, dst):
                weight = random.uniform(1, 10)
                graph.add_edge(src, dst, weight)
                edges_added += 1
        
        return graph
    
    @staticmethod
    def generate_dense_graph(n: int, edge_probability: float = 0.3) -> Graph:
        graph = Graph(directed=True)
        
        for i in range(n):
            for j in range(n):
                if i != j and random.random() < edge_probability:
                    weight = random.uniform(1, 10)
                    graph.add_edge(i, j, weight)
        
        return graph
    
    @staticmethod
    def generate_layered_graph(layers: int, nodes_per_layer: int) -> Graph:
        graph = Graph(directed=True)
        n = layers * nodes_per_layer
        
        for layer in range(layers - 1):
            for i in range(nodes_per_layer):
                src = layer * nodes_per_layer + i
                
                num_connections = random.randint(1, min(3, nodes_per_layer))
                for _ in range(num_connections):
                    dst = (layer + 1) * nodes_per_layer + random.randint(0, nodes_per_layer - 1)
                    if not graph.has_edge(src, dst):
                        weight = random.uniform(1, 10)
                        graph.add_edge(src, dst, weight)
        
        for layer in range(layers):
            for i in range(nodes_per_layer - 1):
                if random.random() < 0.3:
                    src = layer * nodes_per_layer + i
                    dst = layer * nodes_per_layer + i + 1
                    weight = random.uniform(0.5, 2)
                    graph.add_edge(src, dst, weight)
        
        return graph
    
    @staticmethod
    def generate_grid_graph(rows: int, cols: int, directed: bool = False) -> Graph:
        graph = Graph(directed=directed)
        
        for r in range(rows):
            for c in range(cols):
                node_id = r * cols + c
                
                if c < cols - 1:
                    right = r * cols + (c + 1)
                    weight = random.uniform(1, 5)
                    graph.add_edge(node_id, right, weight)
                
                if r < rows - 1:
                    down = (r + 1) * cols + c
                    weight = random.uniform(1, 5)
                    graph.add_edge(node_id, down, weight)
                
                if directed:
                    if c > 0:
                        left = r * cols + (c - 1)
                        weight = random.uniform(1, 5)
                        graph.add_edge(node_id, left, weight)
                    
                    if r > 0:
                        up = (r - 1) * cols + c
                        weight = random.uniform(1, 5)
                        graph.add_edge(node_id, up, weight)
        
        return graph


class AlgorithmBenchmark:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def run_algorithm(self, graph: Graph, algorithm_class, algorithm_name: str, 
                     source: int = 0) -> Tuple[Dict[int, float], float, Dict]:
        start_time = time.perf_counter()
        algorithm = algorithm_class(graph)
        distances = algorithm.compute(source)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        stats = algorithm.get_statistics() if hasattr(algorithm, 'get_statistics') else {}
        
        return distances, execution_time, stats
    
    def verify_correctness(self, distances1: Dict[int, float], 
                          distances2: Dict[int, float]) -> bool:
        for node in distances1:
            if abs(distances1[node] - distances2[node]) > 1e-6:
                return False
        return True
    
    def benchmark_graph(self, graph: Graph, graph_type: str) -> List[BenchmarkResult]:
        results = []
        source = 0
        
        if self.verbose:
            print(f"\nBenchmarking {graph_type} graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
        
        dijkstra_dist, dijkstra_time, dijkstra_stats = self.run_algorithm(
            graph, DijkstraSSSP, "Dijkstra (Binary Heap)", source
        )
        
        dijkstra_result = BenchmarkResult(
            algorithm="Dijkstra (Binary Heap)",
            graph_type=graph_type,
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            execution_time=dijkstra_time,
            relaxations=dijkstra_stats.get("relaxations", 0),
            heap_operations=dijkstra_stats.get("heap_operations", 0),
            nodes_visited=dijkstra_stats.get("nodes_visited", 0),
            correctness_verified=True,
            speedup_vs_dijkstra=1.0
        )
        results.append(dijkstra_result)
        
        if graph.num_nodes <= 1000:
            fib_dist, fib_time, fib_stats = self.run_algorithm(
                graph, DijkstraFibonacci, "Dijkstra (Fibonacci)", source
            )
            
            fib_result = BenchmarkResult(
                algorithm="Dijkstra (Fibonacci)",
                graph_type=graph_type,
                num_nodes=graph.num_nodes,
                num_edges=graph.num_edges,
                execution_time=fib_time,
                relaxations=fib_stats.get("relaxations", 0),
                heap_operations=fib_stats.get("heap_operations", 0),
                nodes_visited=fib_stats.get("nodes_processed", 0),
                correctness_verified=self.verify_correctness(dijkstra_dist, fib_dist),
                speedup_vs_dijkstra=dijkstra_time / fib_time if fib_time > 0 else 0
            )
            results.append(fib_result)
        
        barrier_dist, barrier_time, barrier_stats = self.run_algorithm(
            graph, BarrierBreakerSSSP, "Barrier Breaker", source
        )
        
        barrier_result = BenchmarkResult(
            algorithm="Barrier Breaker",
            graph_type=graph_type,
            num_nodes=graph.num_nodes,
            num_edges=graph.num_edges,
            execution_time=barrier_time,
            relaxations=barrier_stats.get("relaxations", 0),
            heap_operations=barrier_stats.get("heap_operations", 0),
            nodes_visited=barrier_stats.get("nodes_visited", 0),
            correctness_verified=self.verify_correctness(dijkstra_dist, barrier_dist),
            speedup_vs_dijkstra=dijkstra_time / barrier_time if barrier_time > 0 else 0
        )
        results.append(barrier_result)
        
        if self.verbose:
            print(f"  Dijkstra (Binary): {dijkstra_time:.4f}s")
            if graph.num_nodes <= 1000:
                print(f"  Dijkstra (Fibonacci): {fib_time:.4f}s (speedup: {dijkstra_time/fib_time:.2f}x)")
            print(f"  Barrier Breaker: {barrier_time:.4f}s (speedup: {dijkstra_time/barrier_time:.2f}x)")
            print(f"  Correctness verified: {barrier_result.correctness_verified}")
        
        return results
    
    def run_comprehensive_benchmark(self):
        test_configs = [
            ("Sparse", [(100, 2.0), (500, 2.0), (1000, 2.0), (5000, 2.0)]),
            ("Dense", [(50, 0.3), (100, 0.3), (200, 0.2), (500, 0.1)]),
            ("Layered", [(10, 10), (20, 20), (30, 30), (50, 50)]),
            ("Grid", [(10, 10), (20, 20), (30, 30), (50, 50)])
        ]
        
        for graph_type, configs in test_configs:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Testing {graph_type} Graphs")
                print(f"{'='*60}")
            
            for config in configs:
                if graph_type == "Sparse":
                    n, density = config
                    graph = GraphGenerator.generate_sparse_graph(n, density)
                    graph_desc = f"Sparse(n={n}, density={density})"
                elif graph_type == "Dense":
                    n, prob = config
                    graph = GraphGenerator.generate_dense_graph(n, prob)
                    graph_desc = f"Dense(n={n}, p={prob})"
                elif graph_type == "Layered":
                    layers, nodes = config
                    graph = GraphGenerator.generate_layered_graph(layers, nodes)
                    graph_desc = f"Layered({layers}x{nodes})"
                else:
                    rows, cols = config
                    graph = GraphGenerator.generate_grid_graph(rows, cols, directed=True)
                    graph_desc = f"Grid({rows}x{cols})"
                
                results = self.benchmark_graph(graph, graph_desc)
                self.results.extend(results)
    
    def generate_report(self, output_file: str = "benchmark_results.json"):
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*60}")
            print(f"Total configurations tested: {len(self.results)}")
            print(f"Results saved to: {output_file}")
            
            barrier_results = [r for r in self.results if r.algorithm == "Barrier Breaker"]
            valid_speedups = [r.speedup_vs_dijkstra for r in barrier_results if r.correctness_verified]
            
            if valid_speedups:
                print(f"\nBarrier Breaker Performance:")
                print(f"  Average speedup: {sum(valid_speedups)/len(valid_speedups):.2f}x")
                print(f"  Max speedup: {max(valid_speedups):.2f}x")
                print(f"  Min speedup: {min(valid_speedups):.2f}x")
                print(f"  Correctness rate: {len(valid_speedups)}/{len(barrier_results)} ({100*len(valid_speedups)/len(barrier_results):.1f}%)")
    
    def _generate_summary(self) -> Dict[str, Any]:
        barrier_results = [r for r in self.results if r.algorithm == "Barrier Breaker"]
        dijkstra_results = [r for r in self.results if "Dijkstra" in r.algorithm]
        
        if not barrier_results:
            return {}
        
        valid_speedups = [r.speedup_vs_dijkstra for r in barrier_results if r.correctness_verified]
        
        return {
            "total_tests": len(self.results),
            "barrier_breaker": {
                "avg_speedup": sum(valid_speedups) / len(valid_speedups) if valid_speedups else 0,
                "max_speedup": max(valid_speedups) if valid_speedups else 0,
                "min_speedup": min(valid_speedups) if valid_speedups else 0,
                "correctness_rate": len(valid_speedups) / len(barrier_results) if barrier_results else 0,
                "avg_execution_time": sum(r.execution_time for r in barrier_results) / len(barrier_results)
            },
            "dijkstra": {
                "avg_execution_time": sum(r.execution_time for r in dijkstra_results) / len(dijkstra_results) if dijkstra_results else 0
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark shortest path algorithms")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    benchmark = AlgorithmBenchmark(verbose=not args.quiet)
    benchmark.run_comprehensive_benchmark()
    benchmark.generate_report(args.output)


if __name__ == "__main__":
    main()