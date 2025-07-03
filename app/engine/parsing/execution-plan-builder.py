# from typing import List, Dict, Set, Union
# from collections import deque
# from enums import Enum
#
#
# class ExecutionPlanValidationError(str, Enum):
#     NO_ENTRY_POINT = "NO_ENTRY_POINT"
#     INVALID_INPUTS = "INVALID_INPUTS"
#
#
# class ExecutionPlanBuilder:
#     def __init__(self, nodes: List[Node], edges: List[Edge]):
#         self.nodes = nodes
#         self.edges = edges
#         self.planned: Set[str] = set()
#         self.execution_plan: List[Dict] = []
#         self.inputs_with_errors: List[Dict] = []
#
#     def generate(self) -> Union[Dict[str, List[Dict]], Dict[str, Union[str, List[Dict]]]]:
#         entry_points = [n for n in self.nodes if TaskRegistry.get(n.type).is_entry_point]
#
#         if not entry_points:
#             return {"error": {"type": ExecutionPlanValidationError.NO_ENTRY_POINT}}
#
#         phase = 1
#         self.execution_plan.append({
#             "phase": phase,
#             "nodes": [n.dict() for n in entry_points]
#         })
#
#         for entry in entry_points:
#             self.planned.add(entry.id)
#
#         while len(self.planned) < len(self.nodes):
#             phase += 1
#             phase_nodes = []
#
#             for node in self.nodes:
#                 if node.id in self.planned:
#                     continue
#
#                 invalid_inputs = self.get_invalid_inputs(node)
#                 if invalid_inputs:
#                     if self.incomers_planned(node):
#                         self.inputs_with_errors.append({
#                             "nodeId": node.id,
#                             "inputs": invalid_inputs
#                         })
#                     continue
#
#                 phase_nodes.append(node)
#
#             if not phase_nodes:
#                 break  # nothing to plan, stop
#
#             for node in phase_nodes:
#                 self.planned.add(node.id)
#
#             self.execution_plan.append({
#                 "phase": phase,
#                 "nodes": [n.dict() for n in phase_nodes]
#             })
#
#         if self.inputs_with_errors:
#             return {
#                 "error": {
#                     "type": ExecutionPlanValidationError.INVALID_INPUTS,
#                     "invalidElements": self.inputs_with_errors
#                 }
#             }
#
#         return {"executionPlan": self.execution_plan}
#
#     def get_invalid_inputs(self, node: AppNode) -> List[str]:
#         task_def = TaskRegistry.get(node.type)
#         invalid_inputs = []
#
#         for param in task_def.inputs:
#             val = node.inputs.get(param.name)
#             has_value = val is not None and str(val).strip() != ""
#
#             if has_value:
#                 continue
#
#             incoming = [
#                 edge for edge in self.edges
#                 if edge.target == node.id and edge.target_handle == param.name
#             ]
#             if not param.required:
#                 if incoming and all(e.source in self.planned for e in incoming):
#                     continue
#                 if not incoming:
#                     continue  # optional and unused is okay
#
#             if incoming and incoming[0].source in self.planned:
#                 continue
#
#             invalid_inputs.append(param.name)
#
#         return invalid_inputs
#
#     def incomers_planned(self, node: AppNode) -> bool:
#         incomers = [
#             edge.source for edge in self.edges if edge.target == node.id
#         ]
#         return all(n in self.planned for n in incomers)
