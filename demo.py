#!/usr/bin/env python3
"""
Demo script for the Breaking the Sorting Barrier shortest path implementation.
This demonstrates the usage of the new O(m log^(2/3) n) algorithm.
"""

import random
import time
from src.core.graph import Graph
from src.algorithms.dijkstra import DijkstraSSSP
from src.algorithms.barrier_breaker import BarrierBreakerSSSP


def create_demo_graph():
    """Create a sample graph for demonstration."""
    print("Creating demo graph...")
    graph = Graph(directed=True)
    
    # Add edges with weights
    edges = [
        (0, 1, 4), (0, 2, 2),
        (1, 2, 1), (1, 3, 5),
        (2, 3, 8), (2, 4, 10),
        (3, 4, 2), (3, 5, 6),
        (4, 5, 3)
    ]
    
    for src, dst, weight in edges:
        graph.add_edge(src, dst, weight)
    
    print(f"Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges\n")
    return graph


def demo_basic_usage():
    """Demonstrate basic usage of the algorithm."""
    print("=" * 60)
    print("DEMO: Basic Usage")
    print("=" * 60)
    
    graph = create_demo_graph()
    source = 0
    
    # Run the new algorithm
    print("Running Barrier Breaker algorithm...")
    bb = BarrierBreakerSSSP(graph)
    start = time.perf_counter()
    distances = bb.compute(source)
    bb_time = time.perf_counter() - start
    
    print(f"Execution time: {bb_time:.6f} seconds")
    print(f"\nShortest distances from node {source}:")
    for node in sorted(distances.keys()):
        print(f"  Node {node}: {distances[node]:.2f}")
    
    # Get path to specific target
    target = 5
    path, distance = bb.get_shortest_path(source, target)
    print(f"\nShortest path from {source} to {target}:")
    print(f"  Path: {' -> '.join(map(str, path))}")
    print(f"  Total distance: {distance:.2f}")
    
    # Show statistics
    stats = bb.get_statistics()
    print("\nAlgorithm statistics:")
    print(f"  Nodes visited: {stats['nodes_visited']}")
    print(f"  Relaxations performed: {stats['relaxations']}")
    print(f"  Bellman-Ford iterations: {stats['bf_iterations']}")
    print(f"  Clusters processed: {stats['clusters_processed']}")


def demo_comparison():
    """Compare the new algorithm with Dijkstra."""
    print("\n" + "=" * 60)
    print("DEMO: Algorithm Comparison")
    print("=" * 60)
    print("\nNOTE: This algorithm is designed for very large graphs (n > 100,000).")
    print("On smaller graphs, Dijkstra is expected to be faster due to lower overhead.")
    print("The algorithm adaptively disables advanced features on small graphs.\n")

    # Create a larger graph for meaningful comparison
    print("Creating larger graph for comparison...")
    graph = Graph(directed=True)
    n = 100  # Number of nodes
    
    # Create a connected graph
    for i in range(n - 1):
        graph.add_edge(i, i + 1, random.uniform(1, 10))
    
    # Add random edges for complexity
    for _ in range(n * 2):
        src = random.randint(0, n - 1)
        dst = random.randint(0, n - 1)
        if src != dst:
            graph.add_edge(src, dst, random.uniform(1, 10))
    
    print(f"Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges\n")
    
    source = 0
    
    # Run Dijkstra
    print("Running Dijkstra's algorithm...")
    dijkstra = DijkstraSSSP(graph)
    start = time.perf_counter()
    dijkstra_distances = dijkstra.compute(source)
    dijkstra_time = time.perf_counter() - start
    dijkstra_stats = dijkstra.get_statistics()
    
    # Run Barrier Breaker
    print("Running Barrier Breaker algorithm...")
    bb = BarrierBreakerSSSP(graph)
    start = time.perf_counter()
    bb_distances = bb.compute(source)
    bb_time = time.perf_counter() - start
    bb_stats = bb.get_statistics()
    
    # Verify correctness
    print("\nVerifying correctness...")
    all_correct = True
    for node in graph.nodes:
        if abs(dijkstra_distances[node] - bb_distances[node]) > 1e-6:
            all_correct = False
            print(f"  Mismatch at node {node}: Dijkstra={dijkstra_distances[node]}, BB={bb_distances[node]}")
    
    if all_correct:
        print("  ✓ All distances match correctly!")
    
    # Compare performance
    print("\nPerformance Comparison:")
    print(f"  Dijkstra:")
    print(f"    Time: {dijkstra_time:.6f} seconds")
    print(f"    Relaxations: {dijkstra_stats['relaxations']}")
    print(f"    Heap operations: {dijkstra_stats['heap_operations']}")
    
    print(f"  Barrier Breaker:")
    print(f"    Time: {bb_time:.6f} seconds")
    print(f"    Relaxations: {bb_stats['relaxations']}")
    print(f"    Heap operations: {bb_stats['heap_operations']}")
    
    speedup = dijkstra_time / bb_time if bb_time > 0 else 0
    print(f"\n  Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print(f"  Barrier Breaker is {(speedup - 1) * 100:.1f}% faster!")
    elif speedup < 1:
        print(f"  Dijkstra is {(1/speedup - 1) * 100:.1f}% faster (expected on small graphs)")
    else:
        print("  Both algorithms have similar performance")


def demo_scalability():
    """Demonstrate scalability on different graph sizes."""
    print("\n" + "=" * 60)
    print("DEMO: Scalability Analysis")
    print("=" * 60)
    
    sizes = [50, 100, 200, 500]
    
    print("\nTesting on graphs of increasing size (sparse graphs):")
    print("-" * 50)
    print(f"{'Size':<10} {'Dijkstra':<15} {'Barrier':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for n in sizes:
        # Create sparse graph (m = O(n))
        graph = Graph(directed=True)
        
        # Create connected path
        for i in range(n - 1):
            graph.add_edge(i, i + 1, random.uniform(1, 10))
        
        # Add some random edges
        for _ in range(n):
            src = random.randint(0, n - 1)
            dst = random.randint(0, n - 1)
            if src != dst:
                graph.add_edge(src, dst, random.uniform(1, 10))
        
        # Time Dijkstra
        dijkstra = DijkstraSSSP(graph)
        start = time.perf_counter()
        dijkstra.compute(0)
        dijkstra_time = time.perf_counter() - start
        
        # Time Barrier Breaker
        bb = BarrierBreakerSSSP(graph)
        start = time.perf_counter()
        bb.compute(0)
        bb_time = time.perf_counter() - start
        
        speedup = dijkstra_time / bb_time if bb_time > 0 else 0
        
        print(f"{n:<10} {dijkstra_time:<15.6f} {bb_time:<15.6f} {speedup:<10.2f}x")
    
    print("-" * 50)
    print("\nNote: The algorithm shows better speedup on larger, sparser graphs")
    print("      as predicted by the O(m log^(2/3) n) complexity.")


def demo_visualization():
    """Visualize the algorithm's progress."""
    print("\n" + "=" * 60)
    print("DEMO: Algorithm Progress Visualization")
    print("=" * 60)
    
    # Create a layered graph
    print("Creating layered graph...")
    graph = Graph(directed=True)
    
    layers = 5
    nodes_per_layer = 4
    
    # Create layers
    for layer in range(layers - 1):
        for i in range(nodes_per_layer):
            src = layer * nodes_per_layer + i
            for j in range(nodes_per_layer):
                dst = (layer + 1) * nodes_per_layer + j
                if random.random() < 0.5:  # 50% chance of edge
                    weight = random.uniform(1, 5)
                    graph.add_edge(src, dst, weight)
    
    print(f"Created layered graph with {graph.num_nodes} nodes and {graph.num_edges} edges\n")
    
    # Run algorithm and show progress
    bb = BarrierBreakerSSSP(graph, adaptive=True)
    distances = bb.compute(0)
    
    print(bb.visualize_progress())
    
    print("\nAlgorithm completed successfully!")
    print(f"Reached {len([d for d in distances.values() if d < float('inf')])} nodes from source")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Breaking the Sorting Barrier - Algorithm Demo")
    print("=" * 60)
    print("\nThis demo showcases the O(m log^(2/3) n) shortest path algorithm")
    print("that breaks the 40-year-old sorting barrier.\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run demos
    demo_basic_usage()
    demo_comparison()
    demo_scalability()
    demo_visualization()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()