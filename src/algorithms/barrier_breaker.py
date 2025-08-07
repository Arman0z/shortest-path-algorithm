import heapq
import math
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time

from ..core.graph import Graph
from ..core.clustering import FrontierClustering, Cluster
from ..core.layers import LayerManager, AdaptiveLayerManager
from .bellman_ford import SelectiveBellmanFord


@dataclass(order=True)
class ExplorationNode:
    priority: float
    node_id: int = field(compare=False)
    layer_id: int = field(compare=False)
    is_influential: bool = field(default=False, compare=False)


class BarrierBreakerSSSP:
    def __init__(self, graph: Graph, adaptive: bool = True):
        self.graph = graph
        self.adaptive = adaptive
        
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.visited: Set[int] = set()
        self.partially_processed: Set[int] = set()
        
        if adaptive:
            self.layer_manager = AdaptiveLayerManager(graph)
        else:
            self.layer_manager = LayerManager(graph)
        
        self.clustering = FrontierClustering(graph)
        self.selective_bf = SelectiveBellmanFord(graph)
        
        self.num_relaxations = 0
        self.num_heap_operations = 0
        self.num_bf_iterations = 0
        self.num_clusters_processed = 0
        
        self.performance_stats = {
            "layer_times": {},
            "cluster_times": {},
            "bf_times": [],
            "total_time": 0
        }
    
    def compute(self, source: int) -> Dict[int, float]:
        start_time = time.time()
        
        self.distances = {node: float('inf') for node in self.graph.nodes}
        self.predecessors = {node: None for node in self.graph.nodes}
        self.visited = set()
        self.partially_processed = set()
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        
        self.distances[source] = 0
        
        self.layer_manager.initialize_from_source(source)
        
        influential_nodes = self._identify_influential_nodes(source)
        self.layer_manager.mark_influential_nodes(influential_nodes, self.distances)
        
        exploration_queue = []
        heapq.heappush(exploration_queue, ExplorationNode(0, source, 0, False))
        self.num_heap_operations += 1
        
        iteration = 0
        while exploration_queue or not self._all_layers_complete():
            iteration += 1
            
            # Process the main exploration queue
            if exploration_queue:
                self._process_exploration_queue(exploration_queue)
            
            # Only use clustering and Bellman-Ford when beneficial
            if len(self.visited) < self.graph.num_nodes * 0.8:
                if iteration % 10 == 0 and self.adaptive:
                    self._adapt_strategy()
                
                if iteration % 3 == 0:
                    frontier_clusters = self._get_frontier_clusters()
                    if frontier_clusters:
                        self._process_clusters(frontier_clusters, exploration_queue)
                
                if iteration % 5 == 0:
                    self._run_selective_bellman_ford(exploration_queue)
            
            self._update_layers()
        
        self.performance_stats["total_time"] = time.time() - start_time
        return self.distances
    
    def _identify_influential_nodes(self, source: int) -> Set[int]:
        start_time = time.time()
        
        influential = self.selective_bf.identify_influential_nodes(
            source, 
            iterations=min(3, int(math.log(self.graph.num_nodes + 1))),
            threshold_percentile=0.15
        )
        
        connectivity_scores = defaultdict(int)
        for edge in self.graph.edges:
            connectivity_scores[edge.source] += 1
            connectivity_scores[edge.target] += 1
        
        sorted_nodes = sorted(connectivity_scores.items(), key=lambda x: x[1], reverse=True)
        num_high_connectivity = max(1, int(len(sorted_nodes) * 0.1))
        high_connectivity_nodes = {node for node, _ in sorted_nodes[:num_high_connectivity]}
        
        influential.update(high_connectivity_nodes)
        
        self.performance_stats["bf_times"].append(time.time() - start_time)
        return influential
    
    def _process_exploration_queue(self, exploration_queue: List[ExplorationNode]):
        batch_size = max(1, len(exploration_queue))  # Process all available nodes
        
        for _ in range(batch_size):
            if not exploration_queue:
                break
            
            node = heapq.heappop(exploration_queue)
            self.num_heap_operations += 1
            
            if node.node_id in self.visited:
                continue
            
            if node.priority > self.distances[node.node_id]:
                continue
            
            self.visited.add(node.node_id)
            
            for neighbor, weight in self.graph.get_neighbors(node.node_id):
                self.num_relaxations += 1
                new_dist = self.distances[node.node_id] + weight
                
                if new_dist < self.distances[neighbor]:
                    old_dist = self.distances[neighbor]
                    self.distances[neighbor] = new_dist
                    self.predecessors[neighbor] = node.node_id
                    
                    old_layer, new_layer = self.layer_manager.update_node_layer(neighbor, new_dist)
                    
                    is_influential = neighbor in self.layer_manager.get_influential_nodes()
                    
                    # Use actual distance as priority to maintain correctness
                    priority = new_dist
                    
                    if neighbor not in self.visited:
                        heapq.heappush(
                            exploration_queue, 
                            ExplorationNode(priority, neighbor, new_layer, is_influential)
                        )
                        self.num_heap_operations += 1
    
    def _get_frontier_clusters(self) -> List[Cluster]:
        frontier_nodes = set()
        
        for node in self.graph.nodes:
            if node not in self.visited:
                for neighbor, _ in self.graph.get_neighbors(node):
                    if neighbor in self.visited:
                        frontier_nodes.add(node)
                        break
        
        if not frontier_nodes:
            return []
        
        clusters = self.clustering.create_geometric_clusters(frontier_nodes, self.distances)
        
        clusters.sort(key=lambda c: c.avg_distance)
        
        return clusters[:max(1, int(math.sqrt(len(clusters))))]
    
    def _process_clusters(self, clusters: List[Cluster], exploration_queue: List[ExplorationNode]):
        start_time = time.time()
        
        for cluster in clusters:
            self.num_clusters_processed += 1
            
            # Process all nodes in cluster, not just representative
            for node in cluster.nodes:
                if node not in self.visited:
                    layer_id = self.layer_manager.node_to_layer.get(node, 0)
                    is_influential = node in self.layer_manager.get_influential_nodes()
                    
                    priority = self.distances.get(node, float('inf'))
                    
                    if priority < float('inf'):
                        heapq.heappush(
                            exploration_queue,
                            ExplorationNode(priority, node, layer_id, is_influential)
                        )
                        self.num_heap_operations += 1
        
        self.performance_stats["cluster_times"] = time.time() - start_time
    
    def _run_selective_bellman_ford(self, exploration_queue: List[ExplorationNode]):
        start_time = time.time()
        self.num_bf_iterations += 1
        
        influential = self.layer_manager.get_influential_nodes(max_layers=3)
        
        if not influential:
            return
        
        edges_to_relax = []
        for edge in self.graph.edges:
            if edge.source in influential or edge.target in influential:
                edges_to_relax.append(edge)
        
        updated_nodes = set()
        
        for edge in edges_to_relax:
            if self.distances[edge.source] != float('inf'):
                self.num_relaxations += 1
                new_dist = self.distances[edge.source] + edge.weight
                if new_dist < self.distances[edge.target]:
                    self.distances[edge.target] = new_dist
                    self.predecessors[edge.target] = edge.source
                    updated_nodes.add(edge.target)
        
        for node in updated_nodes:
            if node not in self.visited:
                layer_id = self.layer_manager.node_to_layer.get(node, 0)
                is_influential = node in influential
                
                priority = self.distances[node]
                if is_influential:
                    priority *= 0.9
                
                heapq.heappush(
                    exploration_queue,
                    ExplorationNode(priority, node, layer_id, is_influential)
                )
                self.num_heap_operations += 1
        
        self.performance_stats["bf_times"].append(time.time() - start_time)
    
    def _update_layers(self):
        processed_fraction = len(self.visited) / max(1, self.graph.num_nodes)
        
        if processed_fraction > 0.3:
            self.layer_manager.rebalance_layers(self.distances)
    
    def _all_layers_complete(self) -> bool:
        for node in self.graph.nodes:
            if self.distances[node] < float('inf') and node not in self.visited:
                return False
        return True
    
    def _adapt_strategy(self):
        if not self.adaptive:
            return
        
        visited_ratio = len(self.visited) / max(1, self.graph.num_nodes)
        
        if visited_ratio < 0.3:
            self.clustering.target_cluster_size = max(1, int(math.sqrt(self.graph.num_nodes)))
        elif visited_ratio < 0.7:
            self.clustering.target_cluster_size = max(1, int(math.log(self.graph.num_nodes + 1)))
        else:
            self.clustering.target_cluster_size = 1
        
        if hasattr(self.layer_manager, 'adapt_layer_width'):
            self.layer_manager.adapt_layer_width(self.performance_stats.get("layer_times", {}))
    
    def get_shortest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        if source not in self.distances:
            self.compute(source)
        
        if target not in self.graph.nodes:
            raise ValueError(f"Target node {target} not in graph")
        
        if self.distances[target] == float('inf'):
            return [], float('inf')
        
        path = []
        current = target
        visited_in_path = set()
        
        while current is not None:
            if current in visited_in_path:
                raise ValueError("Cycle detected in path reconstruction")
            visited_in_path.add(current)
            path.append(current)
            current = self.predecessors[current]
        
        path.reverse()
        return path, self.distances[target]
    
    def get_statistics(self) -> Dict[str, any]:
        return {
            "nodes_visited": len(self.visited),
            "nodes_partially_processed": len(self.partially_processed),
            "relaxations": self.num_relaxations,
            "heap_operations": self.num_heap_operations,
            "bf_iterations": self.num_bf_iterations,
            "clusters_processed": self.num_clusters_processed,
            "layers_created": len(self.layer_manager.layers),
            "influential_nodes": len(self.layer_manager.get_influential_nodes()),
            "total_time": self.performance_stats.get("total_time", 0),
            "bf_time": sum(self.performance_stats.get("bf_times", [])),
            "cluster_time": self.performance_stats.get("cluster_times", 0)
        }
    
    def visualize_progress(self) -> str:
        lines = ["Algorithm Progress:"]
        lines.append("=" * 60)
        
        visited_pct = len(self.visited) / max(1, self.graph.num_nodes) * 100
        partial_pct = len(self.partially_processed) / max(1, self.graph.num_nodes) * 100
        
        lines.append(f"Nodes visited: {len(self.visited)}/{self.graph.num_nodes} ({visited_pct:.1f}%)")
        lines.append(f"Nodes partially processed: {len(self.partially_processed)} ({partial_pct:.1f}%)")
        lines.append(f"Total relaxations: {self.num_relaxations}")
        lines.append(f"Bellman-Ford iterations: {self.num_bf_iterations}")
        lines.append(f"Clusters processed: {self.num_clusters_processed}")
        
        lines.append("\n" + self.layer_manager.visualize_layers())
        
        return "\n".join(lines)