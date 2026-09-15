from typing import Dict, Any, Callable
from loguru import logger


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: Dict, handler: Callable):
        self.tools[name] = {
            "description": description,
            "parameters": parameters,
            "handler": handler,
        }
        logger.info(f"Tool registered: {name}")

    def get_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t["description"],
                    "parameters": t["parameters"],
                }
            }
            for name, t in self.tools.items()
        ]

    def execute(self, name: str, **kwargs) -> str:
        if name not in self.tools:
            return f"Unknown tool: {name}"
        try:
            return str(self.tools[name]["handler"](**kwargs))
        except Exception as e:
            return f"Tool error: {e}"
