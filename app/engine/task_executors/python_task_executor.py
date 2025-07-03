import importlib

from app.core.execution_context import ExecutionContext
from app.executors.registry import TaskExecutorRegistry
from app.executors.task_executor import TaskExecutor
from app.models.node import Node
from app.models.node_result import NodeResult


class PythonTaskExecutor(TaskExecutor):
    def execute(self, node: Node, context: ExecutionContext) -> NodeResult:
        fn_path = node.config["function"]
        module_name, func_name = fn_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, func_name)
            result = fn(**context.variables)

            return NodeResult(success=True, output=result, logs=f"Executed {fn_path}")
        except Exception as e:
            return NodeResult(
                success=False, output={}, logs=f"Error executing {fn_path}: {str(e)}"
            )


# Register on import
TaskExecutorRegistry.register("python", PythonTaskExecutor())
