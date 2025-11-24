from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
from .graph import Graph


@dataclass
class Layer:
    id: int
    distance_range: Tuple[float, float]
    nodes: Set[int] = field(default_factory=set)
    processed_nodes: Set[int] = field(default_factory=set)
    frontier_nodes: Set[int] = field(default_factory=set)
    influential_nodes: Set[int] = field(default_factory=set)
    
    def add_node(self, node_id: int, is_frontier: bool = False, is_influential: bool = False):
        self.nodes.add(node_id)
        if is_frontier:
            self.frontier_nodes.add(node_id)
        if is_influential:
            self.influential_nodes.add(node_id)
    
    def mark_processed(self, node_id: int):
        self.processed_nodes.add(node_id)
        self.frontier_nodes.discard(node_id)
    
    def is_complete(self) -> bool:
        return len(self.frontier_nodes) == 0
    
    def get_unprocessed_nodes(self) -> Set[int]:
        return self.nodes - self.processed_nodes
    
    def size(self) -> int:
        return len(self.nodes)
    
    def contains_distance(self, distance: float) -> bool:
        return self.distance_range[0] <= distance < self.distance_range[1]


class LayerManager:
    """
    Manages graph layering by distance ranges for efficient exploration.
    Enables processing nodes in layers without strict distance ordering.
    """

    def __init__(self, graph: Graph, layer_granularity: Optional[float] = None):
        self.graph = graph
        self.layers: List[Layer] = []
        self.node_to_layer: Dict[int, int] = {}
        self.current_layer_id = 0

        if layer_granularity is None:
            self.layer_width = self._compute_adaptive_width()
        else:
            self.layer_width = layer_granularity

        self.max_distance = 0.0
        
    def _compute_adaptive_width(self) -> float:
        if self.graph.num_edges == 0:
            return 1.0
        
        total_weight = sum(edge.weight for edge in self.graph.edges)
        avg_weight = total_weight / self.graph.num_edges
        
        return avg_weight * math.log(max(2, self.graph.num_nodes))
    
    def initialize_from_source(self, source: int, initial_distances: Optional[Dict[int, float]] = None):
        self.layers = []
        self.node_to_layer = {}
        self.current_layer_id = 0
        
        if initial_distances is None:
            initial_distances = {source: 0.0}
        
        first_layer = Layer(
            id=0,
            distance_range=(0.0, self.layer_width)
        )
        first_layer.add_node(source, is_frontier=True)
        self.layers.append(first_layer)
        self.node_to_layer[source] = 0
        
        if initial_distances:
            self.max_distance = max(d for d in initial_distances.values() if d < float('inf'))
            num_layers = int(self.max_distance / self.layer_width) + 1
            
            for i in range(1, num_layers + 1):
                layer = Layer(
                    id=i,
                    distance_range=(i * self.layer_width, (i + 1) * self.layer_width)
                )
                self.layers.append(layer)
    
    def assign_node_to_layer(self, node_id: int, distance: float, 
                            is_frontier: bool = False, is_influential: bool = False) -> int:
        if distance == float('inf'):
            return -1
        
        layer_id = int(distance / self.layer_width)
        
        while layer_id >= len(self.layers):
            new_layer = Layer(
                id=len(self.layers),
                distance_range=(len(self.layers) * self.layer_width, 
                              (len(self.layers) + 1) * self.layer_width)
            )
            self.layers.append(new_layer)
        
        self.layers[layer_id].add_node(node_id, is_frontier, is_influential)
        self.node_to_layer[node_id] = layer_id
        
        self.max_distance = max(self.max_distance, distance)
        
        return layer_id
    
    def update_node_layer(self, node_id: int, new_distance: float) -> Tuple[int, int]:
        old_layer_id = self.node_to_layer.get(node_id, -1)
        new_layer_id = int(new_distance / self.layer_width)
        
        if old_layer_id == new_layer_id:
            return old_layer_id, new_layer_id
        
        if old_layer_id >= 0 and old_layer_id < len(self.layers):
            old_layer = self.layers[old_layer_id]
            old_layer.nodes.discard(node_id)
            old_layer.frontier_nodes.discard(node_id)
            old_layer.influential_nodes.discard(node_id)
            old_layer.processed_nodes.discard(node_id)
        
        self.assign_node_to_layer(node_id, new_distance)
        
        return old_layer_id, new_layer_id
    
    def get_current_frontier(self) -> Set[int]:
        frontier = set()
        for layer in self.layers:
            frontier.update(layer.frontier_nodes)
        return frontier
    
    def get_layer_frontier(self, layer_id: int) -> Set[int]:
        if 0 <= layer_id < len(self.layers):
            return self.layers[layer_id].frontier_nodes.copy()
        return set()
    
    def get_influential_nodes(self, max_layers: Optional[int] = None) -> Set[int]:
        influential = set()
        limit = min(max_layers, len(self.layers)) if max_layers else len(self.layers)
        
        for i in range(limit):
            influential.update(self.layers[i].influential_nodes)
        
        return influential
    
    def mark_influential_nodes(self, nodes: Set[int], distances: Dict[int, float]):
        for node_id in nodes:
            if node_id in distances:
                layer_id = self.node_to_layer.get(node_id)
                if layer_id is not None and 0 <= layer_id < len(self.layers):
                    self.layers[layer_id].influential_nodes.add(node_id)
    
    def get_next_unprocessed_layer(self) -> Optional[Layer]:
        for layer in self.layers:
            if not layer.is_complete():
                return layer
        return None
    
    def get_layer_by_distance(self, distance: float) -> Optional[Layer]:
        if distance == float('inf'):
            return None
        
        layer_id = int(distance / self.layer_width)
        if 0 <= layer_id < len(self.layers):
            return self.layers[layer_id]
        return None
    
    def compute_layer_connectivity(self) -> Dict[int, Dict[str, int]]:
        connectivity = {}
        
        for layer in self.layers:
            intra_edges = 0
            forward_edges = 0
            backward_edges = 0
            
            for node in layer.nodes:
                for neighbor, _ in self.graph.get_neighbors(node):
                    neighbor_layer = self.node_to_layer.get(neighbor, -1)
                    
                    if neighbor_layer == layer.id:
                        intra_edges += 1
                    elif neighbor_layer > layer.id:
                        forward_edges += 1
                    elif neighbor_layer >= 0:
                        backward_edges += 1
            
            connectivity[layer.id] = {
                "intra_edges": intra_edges,
                "forward_edges": forward_edges,
                "backward_edges": backward_edges,
                "total_edges": intra_edges + forward_edges + backward_edges
            }
        
        return connectivity
    
    def rebalance_layers(self, distances: Dict[int, float]):
        nodes_to_reassign = []
        
        for node_id, layer_id in self.node_to_layer.items():
            if node_id in distances:
                expected_layer = int(distances[node_id] / self.layer_width)
                if expected_layer != layer_id:
                    nodes_to_reassign.append((node_id, distances[node_id]))
        
        for node_id, distance in nodes_to_reassign:
            self.update_node_layer(node_id, distance)
    
    def get_layer_statistics(self) -> List[Dict[str, Any]]:
        stats = []
        
        for layer in self.layers:
            stats.append({
                "layer_id": layer.id,
                "distance_range": layer.distance_range,
                "total_nodes": layer.size(),
                "processed_nodes": len(layer.processed_nodes),
                "frontier_nodes": len(layer.frontier_nodes),
                "influential_nodes": len(layer.influential_nodes),
                "completion": len(layer.processed_nodes) / max(1, layer.size())
            })
        
        return stats
    
    def visualize_layers(self) -> str:
        lines = ["Layer Structure:"]
        lines.append("-" * 60)
        
        for layer in self.layers:
            bar_length = int(layer.size() / max(1, self.graph.num_nodes) * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            lines.append(
                f"Layer {layer.id:2d} [{layer.distance_range[0]:6.2f}, {layer.distance_range[1]:6.2f}): "
                f"{bar} {layer.size():4d} nodes"
            )
            
            if layer.influential_nodes:
                lines.append(f"         Influential: {len(layer.influential_nodes)} nodes")
            if layer.frontier_nodes:
                lines.append(f"         Frontier: {len(layer.frontier_nodes)} nodes")
        
        return "\n".join(lines)


class AdaptiveLayerManager(LayerManager):
    def __init__(self, graph: Graph):
        super().__init__(graph)
        self.layer_densities: Dict[int, float] = {}
        self.layer_processing_times: Dict[int, float] = {}
        
    def adapt_layer_width(self, performance_data: Dict[int, float]):
        if not performance_data:
            return
        
        avg_time = sum(performance_data.values()) / len(performance_data)
        
        for layer_id, proc_time in performance_data.items():
            if proc_time > avg_time * 1.5:
                self._split_layer(layer_id)
            elif proc_time < avg_time * 0.5 and layer_id < len(self.layers) - 1:
                self._merge_layers(layer_id, layer_id + 1)
    
    def _split_layer(self, layer_id: int):
        if layer_id >= len(self.layers):
            return
        
        layer = self.layers[layer_id]
        mid_distance = (layer.distance_range[0] + layer.distance_range[1]) / 2
        
        new_layer = Layer(
            id=len(self.layers),
            distance_range=(mid_distance, layer.distance_range[1])
        )
        
        layer.distance_range = (layer.distance_range[0], mid_distance)
        
        nodes_to_move = []
        for node_id in layer.nodes:
            if node_id in self.node_to_layer:
                nodes_to_move.append(node_id)
        
        self.layers.insert(layer_id + 1, new_layer)
        
        for i in range(layer_id + 2, len(self.layers)):
            self.layers[i].id = i
    
    def _merge_layers(self, layer1_id: int, layer2_id: int):
        if layer1_id >= len(self.layers) or layer2_id >= len(self.layers):
            return
        
        layer1 = self.layers[layer1_id]
        layer2 = self.layers[layer2_id]
        
        layer1.distance_range = (layer1.distance_range[0], layer2.distance_range[1])
        layer1.nodes.update(layer2.nodes)
        layer1.frontier_nodes.update(layer2.frontier_nodes)
        layer1.influential_nodes.update(layer2.influential_nodes)
        layer1.processed_nodes.update(layer2.processed_nodes)
        
        for node_id in layer2.nodes:
            self.node_to_layer[node_id] = layer1_id
        
        self.layers.pop(layer2_id)
        
        for i in range(layer2_id, len(self.layers)):
            self.layers[i].id = i