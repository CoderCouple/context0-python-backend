import requests

from app.core.execution_context import ExecutionContext
from app.executors.task_executor import TaskExecutor
from app.models.node import Node
from app.models.node_result import NodeResult


class HttpTaskExecutor(TaskExecutor):
    def execute(self, node: Node, context: ExecutionContext) -> NodeResult:
        url = node.config["url"]
        method = node.config.get("method", "POST").upper()
        headers = node.config.get("headers", {})
        body = context.to_json()

        try:
            if method == "POST":
                response = requests.post(url, json=body, headers=headers)
            elif method == "GET":
                response = requests.get(url, params=body, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()

            return NodeResult(
                success=True, output=response.json(), logs=f"Called {method} {url}"
            )

        except Exception as e:
            return NodeResult(
                success=False, output={}, logs=f"HTTP call failed: {str(e)}"
            )


TaskExecutorRegistry.register("http", HttpTaskExecutor())
