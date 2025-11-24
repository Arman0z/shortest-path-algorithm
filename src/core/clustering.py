from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import math
from .graph import Graph


@dataclass
class Cluster:
    id: int
    nodes: Set[int] = field(default_factory=set)
    representative: Optional[int] = None
    frontier_nodes: Set[int] = field(default_factory=set)
    internal_edges: int = 0
    boundary_edges: int = 0
    avg_distance: float = float('inf')
    
    def add_node(self, node_id: int):
        self.nodes.add(node_id)
        if self.representative is None:
            self.representative = node_id
    
    def merge_with(self, other: 'Cluster'):
        self.nodes.update(other.nodes)
        self.frontier_nodes.update(other.frontier_nodes)
        self.internal_edges += other.internal_edges
        self.boundary_edges += other.boundary_edges
        
        if other.avg_distance < self.avg_distance:
            self.representative = other.representative
            self.avg_distance = other.avg_distance
    
    def size(self) -> int:
        return len(self.nodes)
    
    def is_frontier_node(self, node_id: int) -> bool:
        return node_id in self.frontier_nodes
    
    def update_representative(self, distances: Dict[int, float]):
        if not self.nodes:
            return
        
        min_dist = float('inf')
        best_node = None
        
        for node in self.nodes:
            if node in distances and distances[node] < min_dist:
                min_dist = distances[node]
                best_node = node
        
        if best_node is not None:
            self.representative = best_node
            self.avg_distance = min_dist


class FrontierClustering:
    """
    Manages clustering of frontier nodes to reduce comparison operations.
    Groups neighboring nodes to process representatives instead of all nodes.
    """

    def __init__(self, graph: Graph, cluster_size_hint: Optional[int] = None):
        self.graph = graph
        self.clusters: Dict[int, Cluster] = {}
        self.node_to_cluster: Dict[int, int] = {}
        self.next_cluster_id = 0

        if cluster_size_hint is None:
            self.target_cluster_size = max(1, int(math.sqrt(len(graph.nodes))))
        else:
            self.target_cluster_size = cluster_size_hint
    
    def create_distance_based_clusters(self, distances: Dict[int, float], 
                                      visited: Set[int]) -> List[Cluster]:
        self.clusters = {}
        self.node_to_cluster = {}
        self.next_cluster_id = 0
        
        frontier_nodes = set()
        for node in self.graph.nodes:
            if node not in visited:
                for neighbor, _ in self.graph.get_neighbors(node):
                    if neighbor in visited:
                        frontier_nodes.add(node)
                        break
        
        if not frontier_nodes:
            return []
        
        distance_buckets = defaultdict(list)
        for node in frontier_nodes:
            if node in distances:
                bucket = int(distances[node] * 10) / 10.0
                distance_buckets[bucket].append(node)
        
        for bucket_dist, nodes in sorted(distance_buckets.items()):
            self._cluster_nodes_in_bucket(nodes, bucket_dist)
        
        return list(self.clusters.values())
    
    def create_geometric_clusters(self, frontier_nodes: Set[int], 
                                 distances: Dict[int, float]) -> List[Cluster]:
        self.clusters = {}
        self.node_to_cluster = {}
        self.next_cluster_id = 0
        
        if not frontier_nodes:
            return []
        
        unassigned = set(frontier_nodes)
        
        while unassigned:
            seed = min(unassigned, key=lambda n: distances.get(n, float('inf')))
            cluster = self._grow_cluster_from_seed(seed, unassigned, distances)
            unassigned -= cluster.nodes
        
        return list(self.clusters.values())
    
    def _cluster_nodes_in_bucket(self, nodes: List[int], distance: float):
        node_groups = []
        current_group = []
        
        for node in nodes:
            current_group.append(node)
            if len(current_group) >= self.target_cluster_size:
                node_groups.append(current_group)
                current_group = []
        
        if current_group:
            node_groups.append(current_group)
        
        for group in node_groups:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            
            cluster = Cluster(id=cluster_id)
            cluster.avg_distance = distance
            
            for node in group:
                cluster.add_node(node)
                self.node_to_cluster[node] = cluster_id
            
            self._analyze_cluster_connectivity(cluster)
            self.clusters[cluster_id] = cluster
    
    def _grow_cluster_from_seed(self, seed: int, available_nodes: Set[int], 
                               distances: Dict[int, float]) -> Cluster:
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        cluster = Cluster(id=cluster_id)
        cluster.add_node(seed)
        cluster.avg_distance = distances.get(seed, float('inf'))
        
        self.node_to_cluster[seed] = cluster_id
        
        candidates = []
        for neighbor, weight in self.graph.get_neighbors(seed):
            if neighbor in available_nodes:
                candidates.append((neighbor, weight))
        
        candidates.sort(key=lambda x: x[1])
        
        for neighbor, _ in candidates:
            if cluster.size() >= self.target_cluster_size:
                break
            if neighbor in available_nodes:
                cluster.add_node(neighbor)
                self.node_to_cluster[neighbor] = cluster_id
        
        self._analyze_cluster_connectivity(cluster)
        self.clusters[cluster_id] = cluster
        return cluster
    
    def _analyze_cluster_connectivity(self, cluster: Cluster):
        cluster.internal_edges = 0
        cluster.boundary_edges = 0
        cluster.frontier_nodes.clear()
        
        for node in cluster.nodes:
            has_external_edge = False
            for neighbor, _ in self.graph.get_neighbors(node):
                if neighbor in cluster.nodes:
                    cluster.internal_edges += 1
                else:
                    cluster.boundary_edges += 1
                    has_external_edge = True
            
            if has_external_edge:
                cluster.frontier_nodes.add(node)
    
    def get_cluster_representatives(self) -> List[int]:
        representatives = []
        for cluster in self.clusters.values():
            if cluster.representative is not None:
                representatives.append(cluster.representative)
        return representatives
    
    def get_inter_cluster_edges(self) -> List[Tuple[int, int, float]]:
        inter_cluster_edges = []
        
        for edge in self.graph.edges:
            src_cluster = self.node_to_cluster.get(edge.source)
            tgt_cluster = self.node_to_cluster.get(edge.target)
            
            if src_cluster is not None and tgt_cluster is not None:
                if src_cluster != tgt_cluster:
                    inter_cluster_edges.append((
                        self.clusters[src_cluster].representative,
                        self.clusters[tgt_cluster].representative,
                        edge.weight
                    ))
        
        return inter_cluster_edges
    
    def refine_clusters(self, distances: Dict[int, float], iterations: int = 2):
        for _ in range(iterations):
            moves = []
            
            for node_id, cluster_id in self.node_to_cluster.items():
                current_cluster = self.clusters[cluster_id]
                
                if current_cluster.size() <= 1:
                    continue
                
                best_cluster = cluster_id
                best_score = self._compute_node_cluster_affinity(node_id, cluster_id, distances)
                
                for neighbor, _ in self.graph.get_neighbors(node_id):
                    if neighbor in self.node_to_cluster:
                        neighbor_cluster = self.node_to_cluster[neighbor]
                        if neighbor_cluster != cluster_id:
                            score = self._compute_node_cluster_affinity(node_id, neighbor_cluster, distances)
                            if score > best_score:
                                best_score = score
                                best_cluster = neighbor_cluster
                
                if best_cluster != cluster_id:
                    moves.append((node_id, cluster_id, best_cluster))
            
            for node_id, old_cluster, new_cluster in moves:
                self.clusters[old_cluster].nodes.discard(node_id)
                self.clusters[new_cluster].add_node(node_id)
                self.node_to_cluster[node_id] = new_cluster
            
            for cluster in self.clusters.values():
                self._analyze_cluster_connectivity(cluster)
                cluster.update_representative(distances)
    
    def _compute_node_cluster_affinity(self, node_id: int, cluster_id: int, 
                                      distances: Dict[int, float]) -> float:
        if cluster_id not in self.clusters:
            return -float('inf')
        
        cluster = self.clusters[cluster_id]
        
        internal_edges = 0
        for neighbor, weight in self.graph.get_neighbors(node_id):
            if neighbor in cluster.nodes:
                internal_edges += 1
        
        node_dist = distances.get(node_id, float('inf'))
        cluster_dist = cluster.avg_distance
        
        distance_similarity = 1.0 / (1.0 + abs(node_dist - cluster_dist))
        
        connectivity_score = internal_edges / max(1, len(cluster.nodes))
        
        return connectivity_score * 0.7 + distance_similarity * 0.3
    
    def get_statistics(self) -> Dict[str, Any]:
        total_nodes = sum(c.size() for c in self.clusters.values())
        avg_size = total_nodes / max(1, len(self.clusters))
        
        return {
            "num_clusters": len(self.clusters),
            "total_nodes": total_nodes,
            "avg_cluster_size": avg_size,
            "min_cluster_size": min(c.size() for c in self.clusters.values()) if self.clusters else 0,
            "max_cluster_size": max(c.size() for c in self.clusters.values()) if self.clusters else 0,
            "total_internal_edges": sum(c.internal_edges for c in self.clusters.values()),
            "total_boundary_edges": sum(c.boundary_edges for c in self.clusters.values())
        }