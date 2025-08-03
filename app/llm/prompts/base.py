"""Base prompt template system with variable injection and JSON validation"""
import re
import json
from typing import List, Dict, Any, Optional
from jinja2 import Template, Environment, meta, StrictUndefined
import jsonschema
from pydantic import BaseModel

from app.llm.types import ValidationResult, PromptVariable


class PromptTemplate(BaseModel):
    """Base class for all prompt templates with variable injection"""

    name: str
    template: str
    variables: List[PromptVariable]
    output_schema: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    tags: List[str] = []

    # Jinja2 environment is not a pydantic field
    _env: Environment = None

    class Config:
        # Allow arbitrary attributes for internal use
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize template and validate variables"""
        super().__init__(**data)

        # Set up Jinja2 environment with strict undefined
        self._env = Environment(undefined=StrictUndefined)

        # Validate template variables match declared variables
        self._validate_template_variables()

    def _validate_template_variables(self):
        """Validate that template variables match declared variables"""
        # Parse template to find variables
        ast = self._env.parse(self.template)
        template_vars = meta.find_undeclared_variables(ast)

        # Get declared variable names
        declared_vars = {var.name for var in self.variables}

        # Check for undeclared variables
        undeclared = template_vars - declared_vars
        if undeclared:
            raise ValueError(f"Template contains undeclared variables: {undeclared}")

        # Check for unused declared variables
        unused = declared_vars - template_vars
        if unused:
            # This is a warning, not an error
            print(f"Warning: Declared variables not used in template: {unused}")

    def render(self, **kwargs) -> str:
        """
        Render template with variables

        Args:
            **kwargs: Variable values to inject

        Returns:
            Rendered template string

        Raises:
            ValueError: If required variables are missing or validation fails
        """
        # Validate required variables
        missing_required = []
        for var in self.variables:
            if var.required and var.name not in kwargs:
                if var.default_value is not None:
                    kwargs[var.name] = var.default_value
                else:
                    missing_required.append(var.name)

        if missing_required:
            raise ValueError(f"Missing required variables: {missing_required}")

        # Validate variable values
        for var in self.variables:
            if var.name in kwargs and var.validation_regex:
                value = str(kwargs[var.name])
                if not re.match(var.validation_regex, value):
                    raise ValueError(
                        f"Variable '{var.name}' with value '{value}' "
                        f"does not match pattern: {var.validation_regex}"
                    )

        # Render template
        template = self._env.from_string(self.template)
        return template.render(**kwargs)

    def validate_output(self, output: str) -> ValidationResult:
        """
        Validate JSON output against schema

        Args:
            output: JSON string to validate

        Returns:
            ValidationResult with parsed data or errors
        """
        # Try to parse JSON
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return ValidationResult(is_valid=False, errors=[f"Invalid JSON: {str(e)}"])

        # If no schema, just return parsed data
        if not self.output_schema:
            return ValidationResult(is_valid=True, data=data)

        # Validate against schema
        try:
            jsonschema.validate(data, self.output_schema)
            return ValidationResult(is_valid=True, data=data)
        except jsonschema.ValidationError as e:
            return ValidationResult(
                is_valid=False,
                data=data,
                errors=[
                    f"Schema validation failed: {e.message} at {'.'.join(str(p) for p in e.path)}"
                ],
            )
        except jsonschema.SchemaError as e:
            return ValidationResult(
                is_valid=False, errors=[f"Invalid schema: {str(e)}"]
            )

    def extract_and_validate(self, text: str) -> ValidationResult:
        """
        Extract JSON from text (handling markdown blocks) and validate

        Args:
            text: Text potentially containing JSON

        Returns:
            ValidationResult
        """
        # Try to extract JSON from markdown code blocks
        json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(json_pattern, text)

        if matches:
            # Try the first JSON block found
            json_text = matches[0]
        else:
            # Assume the whole text is JSON
            json_text = text.strip()

        return self.validate_output(json_text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary for storage"""
        return {
            "name": self.name,
            "template": self.template,
            "variables": [var.dict() for var in self.variables],
            "output_schema": self.output_schema,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create template from dictionary"""
        # Convert variable dicts to PromptVariable objects
        if "variables" in data:
            data["variables"] = [
                PromptVariable(**var) if isinstance(var, dict) else var
                for var in data["variables"]
            ]
        return cls(**data)
