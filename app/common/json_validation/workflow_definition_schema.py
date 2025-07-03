workflow_definition_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Workflow Definition",
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "position", "data"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "drag_handle": {"type": "string"},
                    "position": {
                        "type": "object",
                        "required": ["x", "y"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                        },
                    },
                    "measured": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "number"},
                            "height": {"type": "number"},
                        },
                    },
                    "data": {
                        "type": "object",
                        "required": ["type", "inputs"],
                        "properties": {
                            "type": {"type": "string"},
                            "inputs": {"type": "object", "additionalProperties": True},
                        },
                        "additionalProperties": True,
                    },
                    "selected": {"type": "boolean"},
                    "dragging": {"type": "boolean"},
                },
                "additionalProperties": True,
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "source", "target"],
                "properties": {
                    "id": {"type": "string"},
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "source_handle": {"type": "string"},
                    "target_handle": {"type": "string"},
                    "animated": {"type": "boolean"},
                },
                "additionalProperties": True,
            },
        },
        "viewport": {
            "type": "object",
            "required": ["x", "y", "zoom"],
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "zoom": {"type": "number"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["nodes", "edges", "viewport"],
    "additionalProperties": False,
}
