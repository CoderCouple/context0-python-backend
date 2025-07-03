class ExecutionContext:
    def __init__(
        self,
        workflow_id: str,
        current_node: Node,
        variables: Dict[str, any],
        tenant_id: UUID,
        user_id: UUID,
    ):
        self.workflow_id = workflow_id
        self.current_node = current_node
        self.variables = variables
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.node_status = {}

    def get_variable(self, name: str):
        return self.variables.get(name)

    def set_variable(self, name: str, value: any):
        self.variables[name] = value

    def to_json(self):
        return {
            "workflow_id": str(self.workflow_id),
            "variables": self.variables,
            "tenant_id": str(self.tenant_id),
            "user_id": str(self.user_id),
        }
