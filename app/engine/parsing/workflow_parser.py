from collections import deque
from typing import Dict, List, Set

from app.common.enums import EdgeField


class WorkflowParser:
    """
    Utility to extract graph structures from a React Flow-style workflow definition.

    Input format (from frontend):
    {
        "nodes": [
            { "id": "A", ... },
            { "id": "B", ... }
        ],
        "edges": [
            { "source": "A", "target": "B" }
        ]
    }

    This class can compute:
    - dependency graph (node -> list of parents) : get_dependency_graph
    - forward graph (node -> list of children) : get_forward_graph
    - root/start nodes : get_start_nodes
    - topological order of execution : topological_sort

    Usage:
     parser = WorkflowParser(definition)

     try:
         parser.validate()
         print("Workflow is valid ✅")
     except ValueError as e:
         print(f"Validation error ❌: {e}")

    """

    def __init__(self, definition: dict):
        self.definition = definition
        self.nodes = definition.get("nodes", [])
        self.edges = definition.get("edges", [])
        self.node_ids = {node["id"] for node in self.nodes}

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Builds a reverse-DAG (dependencies): node_id → list of parent node_ids.

        Example output:
        {
            "A": [],
            "B": ["A"],
            "C": ["A", "B"]
        }
        """
        graph = {node["id"]: [] for node in self.nodes}
        for edge in self.edges:
            graph[edge[EdgeField.TARGET]].append(edge[EdgeField.SOURCE])
        return graph

    def get_forward_graph(self) -> Dict[str, List[str]]:
        """
        Builds a forward-DAG (children): node_id → list of child node_ids.

        Example output:
        {
            "A": ["B", "C"],
            "B": ["C"],
            "C": []
        }
        """
        graph = {node["id"]: [] for node in self.nodes}
        for edge in self.edges:
            graph[edge[EdgeField.SOURCE]].append(edge[EdgeField.TARGET])
        return graph

    def get_start_nodes(self) -> List[str]:
        """
        Returns nodes that have no dependencies (no incoming edges).

        Example output:
        ["A"]  # assuming A is the only root node
        """
        dependency_graph = self.get_dependency_graph()
        return [node_id for node_id, deps in dependency_graph.items() if not deps]

    def topological_sort(self) -> List[str]:
        """
        Returns a list of node IDs sorted topologically (execution order).
        Detects cycles and raises ValueError if one exists.

        Example output:
        ["A", "B", "C"]

        Raises:
            ValueError if a cycle exists in the DAG.
        """
        dep_graph = self.get_dependency_graph()
        in_degree = {node: len(parents) for node, parents in dep_graph.items()}
        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        sorted_nodes = []

        while queue:
            current = queue.popleft()
            sorted_nodes.append(current)

            for edge in self.edges:
                if edge[EdgeField.SOURCE] == current:
                    child = edge[EdgeField.TARGET]
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in workflow graph")

        return sorted_nodes

    def validate(self) -> None:
        """
        Validates the workflow DAG by checking:
        - all nodes exist
        - no orphaned nodes (not reachable from root)
        - graph is acyclic

        Raises:
            ValueError with description if any validation fails.
        """
        # 1. Validate edge_model.py integrity
        for edge in self.edges:
            if edge[EdgeField.SOURCE] not in self.node_ids:
                raise ValueError(
                    f"Edge source '{edge[EdgeField.SOURCE]}' not in node list"
                )
            if edge[EdgeField.TARGET] not in self.node_ids:
                raise ValueError(
                    f"Edge target '{edge[EdgeField.TARGET]}' not in node list"
                )

        # 2. Detect unreachable nodes
        reachable = self.get_all_reachable_nodes()
        unreachable = self.node_ids - reachable
        if unreachable:
            raise ValueError(f"Unreachable nodes detected: {list(unreachable)}")

        # 3. Check for cycles
        self.topological_sort()  # will raise if cycle is found

    def get_all_reachable_nodes(self) -> Set[str]:
        """
        Returns set of all node_ids reachable from root nodes using BFS
        """
        forward_graph = self.get_forward_graph()
        visited = set()
        queue = deque(self.get_start_nodes())

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(forward_graph[node])

        return visited

    def generate_execution_plan(self) -> List[Dict]:
        """
        Groups nodes into execution phases based on level (topo distance from root).
        Each phase contains nodes that can be run in parallel.

        Returns:
            List of dicts like: [{ phase: 1, nodes: [{id: "A"}] }, ...]
        """
        dep_graph = self.get_dependency_graph()
        in_degree = {node: len(parents) for node, parents in dep_graph.items()}
        forward_graph = self.get_forward_graph()

        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        node_to_phase = {}
        phase = 1
        phases = []

        while queue:
            current_phase_nodes = list(queue)
            phases.append(
                {
                    "phase": phase,
                    "nodes": [{"id": node_id} for node_id in current_phase_nodes],
                }
            )

            next_queue = deque()
            for current in current_phase_nodes:
                for child in forward_graph[current]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
                node_to_phase[current] = phase

            queue = next_queue
            phase += 1

        if len(node_to_phase) != len(self.nodes):
            raise ValueError("Cycle detected or unreachable nodes in workflow graph")

        return phases
