import heapq
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import math
from ..core.graph import Graph


@dataclass(order=True)
class PriorityItem:
    priority: float
    node_id: int = field(compare=False)


class DijkstraSSSP:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.visited: Set[int] = set()
        self.num_relaxations = 0
        self.num_heap_operations = 0
        
    def compute(self, source: int) -> Dict[int, float]:
        self.distances = {node: float('inf') for node in self.graph.nodes}
        self.predecessors = {node: None for node in self.graph.nodes}
        self.visited = set()
        self.num_relaxations = 0
        self.num_heap_operations = 0
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        
        self.distances[source] = 0
        
        min_heap = [PriorityItem(0, source)]
        self.num_heap_operations += 1
        
        while min_heap:
            current_item = heapq.heappop(min_heap)
            self.num_heap_operations += 1
            current = current_item.node_id
            current_dist = current_item.priority
            
            if current in self.visited:
                continue
                
            self.visited.add(current)
            
            if current_dist > self.distances[current]:
                continue
            
            for neighbor, weight in self.graph.get_neighbors(current):
                self.num_relaxations += 1
                new_dist = self.distances[current] + weight
                
                if new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.predecessors[neighbor] = current
                    heapq.heappush(min_heap, PriorityItem(new_dist, neighbor))
                    self.num_heap_operations += 1
        
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
        while current is not None:
            path.append(current)
            current = self.predecessors[current]
        
        path.reverse()
        return path, self.distances[target]
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            "nodes_visited": len(self.visited),
            "relaxations": self.num_relaxations,
            "heap_operations": self.num_heap_operations
        }


class FibonacciNode:
    def __init__(self, key: float, value: int):
        self.key = key
        self.value = value
        self.degree = 0
        self.marked = False
        self.parent = None
        self.child = None
        self.left = self
        self.right = self


class FibonacciHeap:
    def __init__(self):
        self.min_node = None
        self.num_nodes = 0
        self.node_map: Dict[int, FibonacciNode] = {}
    
    def is_empty(self) -> bool:
        return self.min_node is None
    
    def insert(self, key: float, value: int) -> FibonacciNode:
        node = FibonacciNode(key, value)
        self.node_map[value] = node
        
        if self.min_node is None:
            self.min_node = node
        else:
            self._add_to_root_list(node)
            if node.key < self.min_node.key:
                self.min_node = node
        
        self.num_nodes += 1
        return node
    
    def extract_min(self) -> Optional[Tuple[float, int]]:
        if self.min_node is None:
            return None
        
        min_node = self.min_node
        result = (min_node.key, min_node.value)
        
        if min_node.value in self.node_map:
            del self.node_map[min_node.value]
        
        if min_node.child:
            child = min_node.child
            while True:
                next_child = child.right
                self._add_to_root_list(child)
                child.parent = None
                child = next_child
                if child == min_node.child:
                    break
        
        self._remove_from_root_list(min_node)
        
        if min_node == min_node.right:
            self.min_node = None
        else:
            self.min_node = min_node.right
            self._consolidate()
        
        self.num_nodes -= 1
        return result
    
    def decrease_key(self, value: int, new_key: float):
        if value not in self.node_map:
            return
        
        node = self.node_map[value]
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        
        node.key = new_key
        parent = node.parent
        
        if parent and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        
        if node.key < self.min_node.key:
            self.min_node = node
    
    def _add_to_root_list(self, node: FibonacciNode):
        if self.min_node is None:
            self.min_node = node
            node.left = node
            node.right = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node
    
    def _remove_from_root_list(self, node: FibonacciNode):
        if node == node.right:
            return
        node.left.right = node.right
        node.right.left = node.left
    
    def _consolidate(self):
        max_degree = int(math.log2(self.num_nodes)) + 1 if self.num_nodes > 0 else 1
        degree_array = [None] * (max_degree + 1)
        
        nodes = []
        current = self.min_node
        while True:
            nodes.append(current)
            current = current.right
            if current == self.min_node:
                break
        
        for node in nodes:
            degree = node.degree
            while degree_array[degree] is not None:
                other = degree_array[degree]
                if node.key > other.key:
                    node, other = other, node
                self._link(other, node)
                degree_array[degree] = None
                degree += 1
            degree_array[degree] = node
        
        self.min_node = None
        for node in degree_array:
            if node is not None:
                if self.min_node is None:
                    self.min_node = node
                    node.left = node
                    node.right = node
                else:
                    self._add_to_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node
    
    def _link(self, child: FibonacciNode, parent: FibonacciNode):
        self._remove_from_root_list(child)
        child.parent = parent
        
        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child.left = parent.child
            child.right = parent.child.right
            parent.child.right.left = child
            parent.child.right = child
        
        parent.degree += 1
        child.marked = False
    
    def _cut(self, child: FibonacciNode, parent: FibonacciNode):
        if child == child.right:
            parent.child = None
        else:
            parent.child = child.right if parent.child == child else parent.child
            child.left.right = child.right
            child.right.left = child.left
        
        parent.degree -= 1
        self._add_to_root_list(child)
        child.parent = None
        child.marked = False
    
    def _cascading_cut(self, node: FibonacciNode):
        parent = node.parent
        if parent:
            if not node.marked:
                node.marked = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)


class DijkstraFibonacci:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.distances: Dict[int, float] = {}
        self.predecessors: Dict[int, Optional[int]] = {}
        self.num_relaxations = 0
        self.num_heap_operations = 0
    
    def compute(self, source: int) -> Dict[int, float]:
        self.distances = {node: float('inf') for node in self.graph.nodes}
        self.predecessors = {node: None for node in self.graph.nodes}
        self.num_relaxations = 0
        self.num_heap_operations = 0
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node {source} not in graph")
        
        self.distances[source] = 0
        
        fib_heap = FibonacciHeap()
        in_heap = set()
        
        for node in self.graph.nodes:
            if node == source:
                fib_heap.insert(0, node)
            else:
                fib_heap.insert(float('inf'), node)
            in_heap.add(node)
            self.num_heap_operations += 1
        
        while not fib_heap.is_empty():
            min_item = fib_heap.extract_min()
            self.num_heap_operations += 1
            
            if min_item is None:
                break
            
            current_dist, current = min_item
            in_heap.discard(current)
            
            if current_dist == float('inf'):
                break
            
            for neighbor, weight in self.graph.get_neighbors(current):
                if neighbor in in_heap:
                    self.num_relaxations += 1
                    new_dist = self.distances[current] + weight
                    
                    if new_dist < self.distances[neighbor]:
                        self.distances[neighbor] = new_dist
                        self.predecessors[neighbor] = current
                        fib_heap.decrease_key(neighbor, new_dist)
                        self.num_heap_operations += 1
        
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
        while current is not None:
            path.append(current)
            current = self.predecessors[current]
        
        path.reverse()
        return path, self.distances[target]
    
    def get_statistics(self) -> Dict[str, int]:
        return {
            "nodes_processed": len([d for d in self.distances.values() if d < float('inf')]),
            "relaxations": self.num_relaxations,
            "heap_operations": self.num_heap_operations
        }