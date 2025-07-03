# # --- Core Engine ---
#
# class WorkflowExecutionEngine:
#     def __init__(self, db: WorkflowRepository, queue: QueueService,
#                  executor_registry: Dict[str, TaskExecutor], logger: EventLogger,
#                  condition_evaluator: ConditionEvaluator):
#         self.db = db
#         self.queue = queue
#         self.executor_registry = executor_registry
#         self.logger = logger
#         self.condition_evaluator = condition_evaluator
#
#     def execute_workflow(self, workflow_id: UUID):
#         workflow = self.db.get(workflow_id)
#         ready_nodes = [n for n in workflow.nodes if n.can_execute(ExecutionContext(workflow_id, n, {}, workflow.owner_id, workflow.owner_id))]
#         for node in ready_nodes:
#             self.queue.enqueue(workflow_id, node.id)
#
#     def schedule_ready_nodes(self, workflow: Workflow):
#         for node in workflow.nodes:
#             if node.can_execute(ExecutionContext(workflow.id, node, {}, workflow.owner_id, workflow.owner_id)):
#                 self.queue.enqueue(workflow.id, node.id)
#
#     def handle_node_result(self, node: Node, result: NodeResult):
#         self.db.update_node(node)
#         self.db.log_node_result(node.id, result)
#         if not result.success:
#             if node.retries < RetryPolicy(3, 5, 'exponential').max_retries:
#                 self.retry_node(node)
#             else:
#                 node.status = "FAILED"
#                 self.logger.log_error(node.id, node.id, result.logs)
#         else:
#             node.status = "SUCCESS"
#
#     def retry_node(self, node: Node):
#         self.queue.enqueue(node.workflow_id, node.id)
#
#     def mark_workflow_complete(self, workflow: Workflow):
#         workflow.status = "COMPLETED"
#         self.db.save(workflow)
#         self.logger.log_event(workflow.id, "Workflow completed")
