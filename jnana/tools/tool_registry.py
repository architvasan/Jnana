"""
Tool registry for Jnana agents.
Manages registration and execution of tools available to LLM agents.
"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for tools available to Jnana agents.
    
    Tools can be registered and then made available to LLM agents
    for function calling during hypothesis generation or evaluation.
    """
    
    def __init__(self):
        """Initialize empty tool registry."""
        self.tools: Dict[str, Any] = {}
        logger.info("Tool registry initialized")
        
    def register_tool(self, tool):
        """
        Register a tool for use by agents.
        
        Args:
            tool: Tool instance with get_tool_schema() and execute() methods
        """
        if not hasattr(tool, 'tool_name'):
            raise ValueError("Tool must have a 'tool_name' attribute")
        
        if not hasattr(tool, 'get_tool_schema'):
            raise ValueError("Tool must have a 'get_tool_schema()' method")
            
        if not hasattr(tool, 'execute'):
            raise ValueError("Tool must have an 'execute()' method")
        
        tool_name = tool.tool_name
        self.tools[tool_name] = tool
        logger.info(f"✓ Registered tool: {tool_name}")
        
    def unregister_tool(self, tool_name: str):
        """
        Unregister a tool.
        
        Args:
            tool_name: Name of the tool to unregister
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
        
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas for LLM function calling.
        
        Returns:
            List of tool schemas in OpenAI function calling format
        """
        schemas = []
        for tool in self.tools.values():
            try:
                schema = tool.get_tool_schema()
                schemas.append(schema)
            except Exception as e:
                logger.error(f"Error getting schema for tool {tool.tool_name}: {e}")
        
        return schemas
    
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution results
        """
        if tool_name not in self.tools:
            logger.error(f"Tool '{tool_name}' not found in registry")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found"
            }
        
        tool = self.tools[tool_name]
        
        try:
            logger.info(f"Executing tool: {tool_name}")
            result = await tool.execute(**kwargs)
            logger.info(f"Tool {tool_name} execution completed")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def has_tool(self, tool_name: str) -> bool:
        """
        Check if a tool is registered.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if tool is registered, False otherwise
        """
        return tool_name in self.tools
    
    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a tool instance by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)

