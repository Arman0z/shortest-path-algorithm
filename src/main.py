#!/usr/bin/env python3

import sys
import os
import argparse
import json
import time
from typing import Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.graph import Graph
from src.algorithms.dijkstra import DijkstraSSSP, DijkstraFibonacci
from src.algorithms.bellman_ford import BellmanFordSSSP
from src.algorithms.barrier_breaker import BarrierBreakerSSSP


def load_graph_from_file(filename: str) -> Graph:
    """Load a graph from a JSON file.
    
    Expected format:
    {
        "directed": true/false,
        "edges": [[source, target, weight], ...]
    }
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    graph = Graph(directed=data.get("directed", True))
    
    for edge in data["edges"]:
        if len(edge) == 3:
            source, target, weight = edge
        else:
            source, target = edge
            weight = 1.0
        graph.add_edge(source, target, weight)
    
    return graph


def save_results_to_file(filename: str, results: Dict):
    """Save algorithm results to a JSON file."""
    # Convert dict keys to strings for JSON serialization
    serializable_results = {
        k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in results.items()
    }

    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def create_example_graph() -> Graph:
    """Create a simple example graph for demonstration."""
    graph = Graph(directed=True)
    
    edges = [
        (0, 1, 4),
        (0, 2, 2),
        (1, 2, 1),
        (1, 3, 5),
        (2, 3, 8),
        (2, 4, 10),
        (3, 4, 2),
        (3, 5, 6),
        (4, 5, 3)
    ]
    
    for src, dst, weight in edges:
        graph.add_edge(src, dst, weight)
    
    return graph


def run_algorithm(graph: Graph, algorithm_name: str, source: int,
                 verbose: bool = False) -> Tuple[Dict[int, float], Dict, float]:
    """Run a specific algorithm and return results."""
    algorithms = {
        "dijkstra": DijkstraSSSP,
        "dijkstra-fib": DijkstraFibonacci,
        "bellman-ford": BellmanFordSSSP,
        "barrier-breaker": BarrierBreakerSSSP
    }

    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    if verbose:
        print(f"\nRunning {algorithm_name} algorithm...")

    algorithm = algorithms[algorithm_name](graph)

    start_time = time.perf_counter()
    distances = algorithm.compute(source)
    execution_time = time.perf_counter() - start_time

    stats = algorithm.get_statistics() if hasattr(algorithm, 'get_statistics') else {}

    if verbose:
        print(f"Execution time: {execution_time:.4f} seconds")
        if stats:
            print("Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

    return distances, stats, execution_time


def compare_algorithms(graph: Graph, source: int, verbose: bool = False):
    """Compare all algorithms on the same graph."""
    
    algorithms = ["dijkstra", "barrier-breaker"]
    
    if graph.num_nodes <= 1000:
        algorithms.append("dijkstra-fib")
    
    if graph.num_nodes <= 5000:
        algorithms.append("bellman-ford")
    
    results = {}
    base_time = None
    base_distances = None
    
    print(f"\nComparing algorithms on graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
    print("=" * 60)
    
    for algo_name in algorithms:
        distances, stats, exec_time = run_algorithm(graph, algo_name, source, verbose)
        
        if base_distances is None:
            base_distances = distances
            base_time = exec_time
        else:
            correctness = all(
                abs(distances[node] - base_distances[node]) < 1e-6 
                for node in distances
            )
            if not correctness:
                print(f"WARNING: {algo_name} produced different results!")
        
        speedup = base_time / exec_time if exec_time > 0 else 0
        
        results[algo_name] = {
            "execution_time": exec_time,
            "speedup": speedup,
            "statistics": stats
        }
        
        print(f"{algo_name:20s}: {exec_time:8.4f}s (speedup: {speedup:5.2f}x)")
    
    print("=" * 60)
    
    barrier_time = results.get("barrier-breaker", {}).get("execution_time", 0)
    dijkstra_time = results.get("dijkstra", {}).get("execution_time", 1)
    
    if barrier_time > 0:
        improvement = (1 - barrier_time / dijkstra_time) * 100
        print(f"\nBarrier Breaker is {improvement:.1f}% faster than Dijkstra")
    
    return results


def visualize_path(graph: Graph, algorithm_name: str, source: int, target: int):
    """Find and display the shortest path between two nodes."""
    
    algorithm_map = {
        "dijkstra": DijkstraSSSP,
        "dijkstra-fib": DijkstraFibonacci,
        "bellman-ford": BellmanFordSSSP,
        "barrier-breaker": BarrierBreakerSSSP
    }
    
    if algorithm_name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    algorithm = algorithm_map[algorithm_name](graph)
    path, distance = algorithm.get_shortest_path(source, target)
    
    print(f"\nShortest path from {source} to {target}:")
    print(f"Distance: {distance}")
    
    if path:
        print(f"Path: {' -> '.join(map(str, path))}")
        
        for i in range(len(path) - 1):
            edge_weight = graph.get_edge_weight(path[i], path[i + 1])
            print(f"  {path[i]} -> {path[i + 1]} (weight: {edge_weight})")
    else:
        print("No path exists")
    
    return path, distance


def main():
    parser = argparse.ArgumentParser(
        description="Breaking the Sorting Barrier - Shortest Path Algorithms"
    )
    
    parser.add_argument(
        "--algorithm", "-a",
        choices=["dijkstra", "dijkstra-fib", "bellman-ford", "barrier-breaker", "compare"],
        default="barrier-breaker",
        help="Algorithm to use (default: barrier-breaker)"
    )
    
    parser.add_argument(
        "--graph", "-g",
        help="Path to graph file (JSON format)"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=int,
        default=0,
        help="Source node (default: 0)"
    )
    
    parser.add_argument(
        "--target", "-t",
        type=int,
        help="Target node for path finding"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with example graph"
    )
    
    args = parser.parse_args()
    
    if args.demo or not args.graph:
        print("Using example graph for demonstration...")
        graph = create_example_graph()
    else:
        print(f"Loading graph from {args.graph}...")
        graph = load_graph_from_file(args.graph)
    
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    if args.algorithm == "compare":
        results = compare_algorithms(graph, args.source, args.verbose)
        if args.output:
            save_results_to_file(args.output, results)
            print(f"\nResults saved to {args.output}")
    
    elif args.target is not None:
        path, distance = visualize_path(graph, args.algorithm, args.source, args.target)
        if args.output:
            results = {
                "algorithm": args.algorithm,
                "source": args.source,
                "target": args.target,
                "path": path,
                "distance": distance
            }
            save_results_to_file(args.output, results)
    
    else:
        distances, stats, exec_time = run_algorithm(
            graph, args.algorithm, args.source, args.verbose
        )
        
        print(f"\nDistances from node {args.source}:")
        
        sorted_nodes = sorted(distances.items(), key=lambda x: x[1])[:10]
        for node, dist in sorted_nodes:
            if dist < float('inf'):
                print(f"  Node {node}: {dist:.4f}")
        
        if len(distances) > 10:
            print(f"  ... ({len(distances) - 10} more nodes)")
        
        if args.output:
            results = {
                "algorithm": args.algorithm,
                "source": args.source,
                "distances": distances,
                "statistics": stats,
                "execution_time": exec_time
            }
            save_results_to_file(args.output, results)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()