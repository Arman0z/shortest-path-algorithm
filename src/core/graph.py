from typing import Dict, List, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
import heapq
from collections import defaultdict, deque


@dataclass
class Edge:
    source: int
    target: int
    weight: float
    
    def __repr__(self):
        return f"Edge({self.source} -> {self.target}, w={self.weight})"


@dataclass
class Node:
    id: int
    edges_out: List[Edge] = field(default_factory=list)
    edges_in: List[Edge] = field(default_factory=list)
    
    def add_outgoing_edge(self, edge: Edge):
        self.edges_out.append(edge)
    
    def add_incoming_edge(self, edge: Edge):
        self.edges_in.append(edge)
    
    def get_neighbors(self, directed: bool = True) -> List[Tuple[int, float]]:
        neighbors = [(e.target, e.weight) for e in self.edges_out]
        if not directed:
            neighbors.extend([(e.source, e.weight) for e in self.edges_in])
        return list(set(neighbors))


class Graph:
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.nodes: Dict[int, Node] = {}
        self.edges: List[Edge] = []
        self.num_nodes = 0
        self.num_edges = 0
    
    def add_node(self, node_id: int) -> Node:
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)
            self.num_nodes += 1
        return self.nodes[node_id]
    
    def add_edge(self, source: int, target: int, weight: float = 1.0):
        src_node = self.add_node(source)
        tgt_node = self.add_node(target)
        
        edge = Edge(source, target, weight)
        src_node.add_outgoing_edge(edge)
        tgt_node.add_incoming_edge(edge)
        self.edges.append(edge)
        
        if not self.directed:
            reverse_edge = Edge(target, source, weight)
            tgt_node.add_outgoing_edge(reverse_edge)
            src_node.add_incoming_edge(reverse_edge)
            self.edges.append(reverse_edge)
            self.num_edges += 2
        else:
            self.num_edges += 1
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        if node_id not in self.nodes:
            return []
        return self.nodes[node_id].get_neighbors(self.directed)
    
    def get_adjacency_list(self) -> Dict[int, List[Tuple[int, float]]]:
        adj_list = {}
        for node_id, node in self.nodes.items():
            adj_list[node_id] = node.get_neighbors(self.directed)
        return adj_list
    
    def get_edge_list(self) -> List[Tuple[int, int, float]]:
        return [(e.source, e.target, e.weight) for e in self.edges]
    
    def get_node_ids(self) -> List[int]:
        return list(self.nodes.keys())
    
    def has_node(self, node_id: int) -> bool:
        return node_id in self.nodes
    
    def has_edge(self, source: int, target: int) -> bool:
        if source not in self.nodes:
            return False
        for edge in self.nodes[source].edges_out:
            if edge.target == target:
                return True
        return False
    
    def get_edge_weight(self, source: int, target: int) -> Optional[float]:
        if source not in self.nodes:
            return None
        for edge in self.nodes[source].edges_out:
            if edge.target == target:
                return edge.weight
        return None
    
    def get_strongly_connected_components(self) -> List[Set[int]]:
        if not self.directed:
            visited = set()
            components = []
            
            for node_id in self.nodes:
                if node_id not in visited:
                    component = set()
                    queue = deque([node_id])
                    
                    while queue:
                        current = queue.popleft()
                        if current not in visited:
                            visited.add(current)
                            component.add(current)
                            for neighbor, _ in self.get_neighbors(current):
                                if neighbor not in visited:
                                    queue.append(neighbor)
                    
                    components.append(component)
            return components
        
        def tarjan_scc():
            index_counter = [0]
            stack = []
            lowlink = {}
            index = {}
            on_stack = defaultdict(bool)
            components = []
            
            def strongconnect(node):
                index[node] = index_counter[0]
                lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
                
                for neighbor, _ in self.get_neighbors(node):
                    if neighbor not in index:
                        strongconnect(neighbor)
                        lowlink[node] = min(lowlink[node], lowlink[neighbor])
                    elif on_stack[neighbor]:
                        lowlink[node] = min(lowlink[node], index[neighbor])
                
                if lowlink[node] == index[node]:
                    component = set()
                    while True:
                        w = stack.pop()
                        on_stack[w] = False
                        component.add(w)
                        if w == node:
                            break
                    components.append(component)
            
            for node in self.nodes:
                if node not in index:
                    strongconnect(node)
            
            return components
        
        return tarjan_scc()
    
    def get_reverse_graph(self) -> 'Graph':
        if not self.directed:
            return self
        
        reverse_graph = Graph(directed=True)
        for edge in self.edges:
            reverse_graph.add_edge(edge.target, edge.source, edge.weight)
        return reverse_graph
    
    def to_adjacency_matrix(self) -> Tuple[List[List[float]], Dict[int, int]]:
        node_ids = sorted(self.nodes.keys())
        id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
        n = len(node_ids)
        
        matrix = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 0
        
        for edge in self.edges:
            src_idx = id_to_index[edge.source]
            tgt_idx = id_to_index[edge.target]
            matrix[src_idx][tgt_idx] = min(matrix[src_idx][tgt_idx], edge.weight)
        
        return matrix, id_to_index
    
    def __repr__(self):
        return f"Graph(directed={self.directed}, nodes={self.num_nodes}, edges={self.num_edges})"
    
    def __str__(self):
        lines = [f"Graph with {self.num_nodes} nodes and {self.num_edges} edges:"]
        for node_id in sorted(self.nodes.keys()):
            neighbors = self.get_neighbors(node_id)
            if neighbors:
                neighbor_str = ", ".join([f"{n}(w={w})" for n, w in neighbors])
                lines.append(f"  {node_id} -> {neighbor_str}")
        return "\n".join(lines)


class WeightedGraph(Graph):
    def __init__(self, directed: bool = True):
        super().__init__(directed)
        self.min_weight = float('inf')
        self.max_weight = float('-inf')
    
    def add_edge(self, source: int, target: int, weight: float = 1.0):
        super().add_edge(source, target, weight)
        self.min_weight = min(self.min_weight, weight)
        self.max_weight = max(self.max_weight, weight)
    
    def get_weight_range(self) -> Tuple[float, float]:
        return (self.min_weight, self.max_weight)
    
    def normalize_weights(self) -> 'WeightedGraph':
        normalized = WeightedGraph(self.directed)
        weight_range = self.max_weight - self.min_weight
        
        if weight_range == 0:
            for edge in self.edges:
                normalized.add_edge(edge.source, edge.target, 1.0)
        else:
            for edge in self.edges:
                norm_weight = (edge.weight - self.min_weight) / weight_range
                normalized.add_edge(edge.source, edge.target, norm_weight)
        
        return normalized