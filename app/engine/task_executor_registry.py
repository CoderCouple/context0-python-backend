# from typing import Dict, Type
# from app.executors.base import TaskExecutor
#
#
# class TaskExecutorRegistry:
#     _registry: Dict[str, TaskExecutor] = {}
#
#     @classmethod
#     def register(cls, node_type: str, executor: TaskExecutor):
#         cls._registry[node_type] = executor
#
#     @classmethod
#     def get(cls, node_type: str) -> TaskExecutor:
#         if node_type not in cls._registry:
#             raise ValueError(f"No executor registered for type '{node_type}'")
#         return cls._registry[node_type]
