"""
LLM via AICredits (OpenAI-compatible).
"""
import json
from typing import List, Dict, Any, Optional, Callable
from loguru import logger
from src.aicredits_client import get_client


class OpenAILLM:
    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.6,
        max_tokens: int = 300,
    ):
        self.client = get_client()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools: List[Dict] = []
        self.tool_handlers: Dict[str, Callable] = {}

    def register_tool(self, name: str, description: str, parameters: Dict, handler: Callable):
        self.tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        })
        self.tool_handlers[name] = handler
        logger.info(f"Registered tool: {name}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            result = {
                "content": message.content or "",
                "tool_calls": [],
                "raw": response,
            }

            if message.tool_calls:
                for tc in message.tool_calls:
                    result["tool_calls"].append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments or "{}"),
                    })
                    logger.info(f"Tool call requested: {tc.function.name}")

            return result
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {
                "content": "I'm having trouble thinking right now. Could you repeat that?",
                "tool_calls": [],
                "raw": None,
            }

    def execute_tools(self, tool_calls: List[Dict]) -> List[Dict]:
        results = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            handler = self.tool_handlers.get(name)
            if handler:
                try:
                    output = handler(**args)
                    results.append({
                        "tool_call_id": tc["id"],
                        "role": "tool",
                        "name": name,
                        "content": str(output),
                    })
                    logger.info(f"Tool {name} result: {output}")
                except Exception as e:
                    results.append({
                        "tool_call_id": tc["id"],
                        "role": "tool",
                        "name": name,
                        "content": f"Error: {str(e)}",
                    })
            else:
                results.append({
                    "tool_call_id": tc["id"],
                    "role": "tool",
                    "name": name,
                    "content": f"Tool {name} not found",
                })
        return results
