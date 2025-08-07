import pytest
import random
import math
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.graph import Graph, WeightedGraph
from src.algorithms.dijkstra import DijkstraSSSP, DijkstraFibonacci
from src.algorithms.bellman_ford import BellmanFordSSSP
from src.algorithms.barrier_breaker import BarrierBreakerSSSP


class TestGraphGeneration:
    @staticmethod
    def create_simple_graph() -> Graph:
        g = Graph(directed=True)
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
            g.add_edge(src, dst, weight)
        return g
    
    @staticmethod
    def create_complete_graph(n: int) -> Graph:
        g = Graph(directed=True)
        for i in range(n):
            for j in range(n):
                if i != j:
                    weight = random.uniform(1, 10)
                    g.add_edge(i, j, weight)
        return g
    
    @staticmethod
    def create_sparse_graph(n: int, m: int) -> Graph:
        g = Graph(directed=True)
        edges_added = set()
        
        for i in range(n - 1):
            weight = random.uniform(1, 10)
            g.add_edge(i, i + 1, weight)
            edges_added.add((i, i + 1))
        
        while len(edges_added) < m:
            src = random.randint(0, n - 1)
            dst = random.randint(0, n - 1)
            if src != dst and (src, dst) not in edges_added:
                weight = random.uniform(1, 10)
                g.add_edge(src, dst, weight)
                edges_added.add((src, dst))
        
        return g
    
    @staticmethod
    def create_grid_graph(rows: int, cols: int) -> Graph:
        g = Graph(directed=False)
        
        for r in range(rows):
            for c in range(cols):
                node_id = r * cols + c
                
                if c < cols - 1:
                    right = r * cols + (c + 1)
                    weight = random.uniform(1, 5)
                    g.add_edge(node_id, right, weight)
                
                if r < rows - 1:
                    down = (r + 1) * cols + c
                    weight = random.uniform(1, 5)
                    g.add_edge(node_id, down, weight)
        
        return g


class TestAlgorithmCorrectness:
    
    def test_simple_graph_all_algorithms(self):
        graph = TestGraphGeneration.create_simple_graph()
        source = 0
        
        dijkstra = DijkstraSSSP(graph)
        dijkstra_fib = DijkstraFibonacci(graph)
        bellman_ford = BellmanFordSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        dist_dijkstra = dijkstra.compute(source)
        dist_dijkstra_fib = dijkstra_fib.compute(source)
        dist_bellman = bellman_ford.compute(source)
        dist_barrier = barrier_breaker.compute(source)
        
        expected = {0: 0, 1: 3, 2: 2, 3: 7, 4: 9, 5: 12}
        
        for node in expected:
            assert abs(dist_dijkstra[node] - expected[node]) < 1e-6, \
                f"Dijkstra failed for node {node}"
            assert abs(dist_dijkstra_fib[node] - expected[node]) < 1e-6, \
                f"Dijkstra Fibonacci failed for node {node}"
            assert abs(dist_bellman[node] - expected[node]) < 1e-6, \
                f"Bellman-Ford failed for node {node}"
            assert abs(dist_barrier[node] - expected[node]) < 1e-6, \
                f"Barrier Breaker failed for node {node}"
    
    def test_disconnected_graph(self):
        graph = Graph(directed=True)
        graph.add_edge(0, 1, 1)
        graph.add_edge(2, 3, 1)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        dist_dijkstra = dijkstra.compute(0)
        dist_barrier = barrier_breaker.compute(0)
        
        assert dist_dijkstra[0] == 0
        assert dist_dijkstra[1] == 1
        assert dist_dijkstra[2] == float('inf')
        assert dist_dijkstra[3] == float('inf')
        
        assert dist_barrier[0] == 0
        assert dist_barrier[1] == 1
        assert dist_barrier[2] == float('inf')
        assert dist_barrier[3] == float('inf')
    
    def test_single_node_graph(self):
        graph = Graph(directed=True)
        graph.add_node(0)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        dist_dijkstra = dijkstra.compute(0)
        dist_barrier = barrier_breaker.compute(0)
        
        assert dist_dijkstra[0] == 0
        assert dist_barrier[0] == 0
    
    def test_path_reconstruction(self):
        graph = TestGraphGeneration.create_simple_graph()
        source = 0
        target = 5
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        path_dijkstra, dist_dijkstra = dijkstra.get_shortest_path(source, target)
        path_barrier, dist_barrier = barrier_breaker.get_shortest_path(source, target)
        
        assert path_dijkstra == [0, 2, 1, 3, 4, 5] or path_dijkstra == [0, 1, 3, 4, 5]
        assert abs(dist_dijkstra - 12) < 1e-6
        assert abs(dist_barrier - 12) < 1e-6
    
    @pytest.mark.parametrize("n", [10, 20, 50])
    def test_complete_graphs(self, n):
        random.seed(42)
        graph = TestGraphGeneration.create_complete_graph(n)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        for source in range(min(5, n)):
            dist_dijkstra = dijkstra.compute(source)
            dist_barrier = barrier_breaker.compute(source)
            
            for node in graph.nodes:
                assert abs(dist_dijkstra[node] - dist_barrier[node]) < 1e-6, \
                    f"Mismatch at node {node} from source {source}"
    
    @pytest.mark.parametrize("n,m", [(50, 100), (100, 300), (200, 500)])
    def test_sparse_graphs(self, n, m):
        random.seed(42)
        graph = TestGraphGeneration.create_sparse_graph(n, m)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        source = 0
        dist_dijkstra = dijkstra.compute(source)
        dist_barrier = barrier_breaker.compute(source)
        
        for node in graph.nodes:
            if dist_dijkstra[node] < float('inf'):
                assert abs(dist_dijkstra[node] - dist_barrier[node]) < 1e-6, \
                    f"Mismatch at node {node}"
            else:
                assert dist_barrier[node] == float('inf'), \
                    f"Barrier breaker found path to unreachable node {node}"
    
    @pytest.mark.parametrize("rows,cols", [(5, 5), (10, 10), (20, 20)])
    def test_grid_graphs(self, rows, cols):
        random.seed(42)
        graph = TestGraphGeneration.create_grid_graph(rows, cols)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        source = 0
        dist_dijkstra = dijkstra.compute(source)
        dist_barrier = barrier_breaker.compute(source)
        
        for node in graph.nodes:
            assert abs(dist_dijkstra[node] - dist_barrier[node]) < 1e-6, \
                f"Mismatch at node {node} in grid graph"
    
    def test_zero_weight_edges(self):
        graph = Graph(directed=True)
        graph.add_edge(0, 1, 0)
        graph.add_edge(1, 2, 5)
        graph.add_edge(0, 2, 10)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        dist_dijkstra = dijkstra.compute(0)
        dist_barrier = barrier_breaker.compute(0)
        
        assert dist_dijkstra[0] == 0
        assert dist_dijkstra[1] == 0
        assert dist_dijkstra[2] == 5
        
        assert dist_barrier[0] == 0
        assert dist_barrier[1] == 0
        assert dist_barrier[2] == 5
    
    def test_large_weight_variation(self):
        graph = Graph(directed=True)
        graph.add_edge(0, 1, 0.001)
        graph.add_edge(1, 2, 1000)
        graph.add_edge(0, 2, 1001)
        graph.add_edge(2, 3, 0.001)
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        dist_dijkstra = dijkstra.compute(0)
        dist_barrier = barrier_breaker.compute(0)
        
        for node in graph.nodes:
            assert abs(dist_dijkstra[node] - dist_barrier[node]) < 1e-6, \
                f"Mismatch at node {node} with large weight variation"


class TestPerformanceCharacteristics:
    
    def test_algorithm_statistics(self):
        graph = TestGraphGeneration.create_sparse_graph(100, 200)
        source = 0
        
        dijkstra = DijkstraSSSP(graph)
        barrier_breaker = BarrierBreakerSSSP(graph)
        
        dijkstra.compute(source)
        barrier_breaker.compute(source)
        
        dijkstra_stats = dijkstra.get_statistics()
        barrier_stats = barrier_breaker.get_statistics()
        
        assert dijkstra_stats["nodes_visited"] > 0
        assert dijkstra_stats["relaxations"] > 0
        assert dijkstra_stats["heap_operations"] > 0
        
        assert barrier_stats["nodes_visited"] > 0
        assert barrier_stats["relaxations"] > 0
        assert barrier_stats["bf_iterations"] >= 0
        assert barrier_stats["clusters_processed"] >= 0
    
    def test_adaptive_vs_non_adaptive(self):
        random.seed(42)
        graph = TestGraphGeneration.create_sparse_graph(50, 100)
        source = 0
        
        adaptive_bb = BarrierBreakerSSSP(graph, adaptive=True)
        non_adaptive_bb = BarrierBreakerSSSP(graph, adaptive=False)
        
        dist_adaptive = adaptive_bb.compute(source)
        dist_non_adaptive = non_adaptive_bb.compute(source)
        
        for node in graph.nodes:
            assert abs(dist_adaptive[node] - dist_non_adaptive[node]) < 1e-6, \
                f"Adaptive and non-adaptive results differ at node {node}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])