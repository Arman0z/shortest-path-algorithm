from typing import Dict, List, Tuple, Optional, Set, Any
from ..core.graph import Graph


class BellmanFordSSSP:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.num_relaxations = 0
        self.iterations_performed = 0
        self.early_termination = False
        
    def compute(self, source: int, detect_negative_cycles: bool = True) -> Dict[int, float]:
        self.distances = {node: float('inf') for node in self.graph.nodes}
        self.predecessors = {node: None for node in self.graph.nodes}
        self.num_relaxations = 0
        self.iterations_performed = 0
        self.early_termination = False
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        
        self.distances[source] = 0
        
        num_nodes = len(self.graph.nodes)
        
        for i in range(num_nodes - 1):
            self.iterations_performed += 1
            updated = False
            
            for edge in self.graph.edges:
                self.num_relaxations += 1
                if self.distances[edge.source] != float('inf'):
                    new_dist = self.distances[edge.source] + edge.weight
                    if new_dist < self.distances[edge.target]:
                        self.distances[edge.target] = new_dist
                        self.predecessors[edge.target] = edge.source
                        updated = True
            
            if not updated:
                self.early_termination = True
                break
        
        if detect_negative_cycles:
            for edge in self.graph.edges:
                if self.distances[edge.source] != float('inf'):
                    if self.distances[edge.source] + edge.weight < self.distances[edge.target]:
                        raise ValueError("Graph contains negative weight cycle")
        
        return self.distances
    
    def compute_limited(self, source: int, max_iterations: int) -> Dict[int, float]:
        self.distances = {node: float('inf') for node in self.graph.nodes}
        self.predecessors = {node: None for node in self.graph.nodes}
        self.num_relaxations = 0
        self.iterations_performed = 0
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        
        self.distances[source] = 0
        
        for i in range(max_iterations):
            self.iterations_performed += 1
            updated = False
            
            for edge in self.graph.edges:
                self.num_relaxations += 1
                if self.distances[edge.source] != float('inf'):
                    new_dist = self.distances[edge.source] + edge.weight
                    if new_dist < self.distances[edge.target]:
                        self.distances[edge.target] = new_dist
                        self.predecessors[edge.target] = edge.source
                        updated = True
            
            if not updated:
                break
        
        return self.distances
    
    def get_shortest_path(self, source: int, target: int) -> Tuple[List[int], float]:
        if source not in self.distances:
            self.compute(source)
        
        if target not in self.graph.nodes:
            raise ValueError(f"Target node {target} not in graph")
        
        if self.distances[target] == float('inf'):
            return [], float('inf')
        
        path = []
        current = target
        visited = set()
        
        while current is not None:
            if current in visited:
                raise ValueError("Negative cycle detected in path reconstruction")
            visited.add(current)
            path.append(current)
            current = self.predecessors[current]
        
        path.reverse()
        return path, self.distances[target]
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations_performed,
            "relaxations": self.num_relaxations,
            "early_termination": self.early_termination,
            "nodes_reached": len([d for d in self.distances.values() if d < float('inf')])
        }


class SelectiveBellmanFord:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.influential_nodes: Set[int] = set()
        self.num_relaxations = 0
        
    def identify_influential_nodes(self, source: int, iterations: int = 3, 
                                  threshold_percentile: float = 0.2) -> Set[int]:
        temp_distances = {node: float('inf') for node in self.graph.nodes}
        temp_distances[source] = 0
        
        node_improvements = {node: 0 for node in self.graph.nodes}
        
        for _ in range(iterations):
            for edge in self.graph.edges:
                if temp_distances[edge.source] != float('inf'):
                    new_dist = temp_distances[edge.source] + edge.weight
                    if new_dist < temp_distances[edge.target]:
                        improvement = temp_distances[edge.target] - new_dist
                        temp_distances[edge.target] = new_dist
                        node_improvements[edge.source] += improvement
        
        sorted_nodes = sorted(node_improvements.items(), key=lambda x: x[1], reverse=True)
        num_influential = max(1, int(len(sorted_nodes) * threshold_percentile))
        
        self.influential_nodes = {node for node, _ in sorted_nodes[:num_influential]}
        return self.influential_nodes
    
    def compute_from_influential(self, source: int, max_iterations: int = 5) -> Dict[int, float]:
        self.distances = {node: float('inf') for node in self.graph.nodes}
        self.predecessors = {node: None for node in self.graph.nodes}
        self.num_relaxations = 0
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        
        self.influential_nodes = self.identify_influential_nodes(source, max_iterations)
        
        self.distances[source] = 0
        
        for _ in range(max_iterations):
            updated = False
            
            for edge in self.graph.edges:
                if edge.source in self.influential_nodes or edge.target in self.influential_nodes:
                    self.num_relaxations += 1
                    if self.distances[edge.source] != float('inf'):
                        new_dist = self.distances[edge.source] + edge.weight
                        if new_dist < self.distances[edge.target]:
                            self.distances[edge.target] = new_dist
                            self.predecessors[edge.target] = edge.source
                            updated = True
            
            if not updated:
                break
        
        for _ in range(2):
            for edge in self.graph.edges:
                self.num_relaxations += 1
                if self.distances[edge.source] != float('inf'):
                    new_dist = self.distances[edge.source] + edge.weight
                    if new_dist < self.distances[edge.target]:
                        self.distances[edge.target] = new_dist
                        self.predecessors[edge.target] = edge.source
        
        return self.distances
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "influential_nodes": len(self.influential_nodes),
            "relaxations": self.num_relaxations,
            "nodes_reached": len([d for d in self.distances.values() if d < float('inf')])
        }