"""
BindCraft computational design tool for Jnana.
Allows LLM agents to invoke binder design directly during hypothesis generation.
"""
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class BindCraftTool:
    """
    Tool wrapper for BindCraft binder design agent.
    
    This allows Jnana's LLM agents to invoke computational binder design
    as a tool call during hypothesis generation or evaluation.
    
    The tool is OPTIONAL - if not called, the existing workflow continues unchanged.
    """
    
    def __init__(self, bindcraft_agent=None):
        """
        Initialize BindCraft tool.
        
        Args:
            bindcraft_agent: Instance of BindCraftAgent from StructBioReasoner
        """
        self.bindcraft_agent = bindcraft_agent
        self.tool_name = "bindcraft_design"
        self.description = (
            "Design peptide binders for a target protein using BindCraft computational design. "
            "Performs inverse folding with ProteinMPNN to generate novel binder sequences. "
            "Returns optimized sequences, predicted structures, and binding metrics. "
            "Use this when you want to computationally design binders rather than just proposing them from literature."
        )
        
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for LLM function calling.
        
        Returns:
            Tool schema in OpenAI function calling format
        """
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_sequence": {
                            "type": "string",
                            "description": "Amino acid sequence of the target protein to bind"
                        },
                        "binder_sequence": {
                            "type": "string",
                            "description": "Initial binder sequence (optional, can be empty string for de novo design)"
                        },
                        "num_sequences": {
                            "type": "integer",
                            "description": "Number of binder sequences to generate per round (default: 10)",
                            "default": 10
                        },
                        "num_rounds": {
                            "type": "integer",
                            "description": "Number of design-fold-filter rounds (default: 1)",
                            "default": 1
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature for ProteinMPNN (0.0-1.0, default: 0.1)",
                            "default": 0.1
                        }
                    },
                    "required": ["target_sequence"]
                }
            }
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute BindCraft design.
        
        Args:
            **kwargs: Tool parameters (target_sequence, binder_sequence, etc.)
            
        Returns:
            Design results with sequences, structures, and metrics
        """
        if not self.bindcraft_agent:
            logger.warning("BindCraft agent not initialized, tool call will be skipped")
            return {
                "success": False,
                "error": "BindCraft agent not initialized",
                "message": "Tool is not available. Please propose binders from literature instead."
            }
        
        try:
            logger.info(f"🔧 LLM called bindcraft_design tool with parameters: {list(kwargs.keys())}")
            
            # Prepare task parameters for BindCraft
            task_params = {
                "target_sequence": kwargs.get("target_sequence"),
                "binder_sequence": kwargs.get("binder_sequence", ""),
                "num_sequences": kwargs.get("num_sequences", 10),
                "num_rounds": kwargs.get("num_rounds", 1),
                "sampling_temp": kwargs.get("temperature", 0.1),
                "cwd": str(Path.cwd() / "bindcraft_output")
            }
            
            logger.info(f"  - Target sequence: {task_params['target_sequence'][:50]}... ({len(task_params['target_sequence'])} residues)")
            logger.info(f"  - Num sequences: {task_params['num_sequences']}")
            logger.info(f"  - Num rounds: {task_params['num_rounds']}")
            
            # Call BindCraft agent's run_design_cycle method
            result = await self.bindcraft_agent.run_design_cycle(task_params)
            
            # Format results for LLM consumption
            formatted_result = self._format_results(result)
            
            logger.info(f"✓ BindCraft tool execution completed: {formatted_result.get('num_sequences_generated', 0)} sequences generated")
            return formatted_result
            
        except Exception as e:
            logger.error(f"BindCraft tool execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "message": "Tool execution failed. Please propose binders from literature instead."
            }
    
    def _format_results(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format BindCraft results for LLM interpretation.
        
        Args:
            raw_result: Raw results from BindCraft agent
            
        Returns:
            Formatted results with key metrics and sequences
        """
        # Extract top sequences from all cycles
        top_sequences = []
        all_cycles = raw_result.get("all_cycles", [])
        
        for cycle in all_cycles:
            for seq_data in cycle.get("sequences", [])[:3]:  # Top 3 per cycle
                top_sequences.append({
                    "sequence": seq_data.get("sequence"),
                    "plddt": seq_data.get("plddt"),
                    "pae": seq_data.get("pae"),
                    "energy": seq_data.get("energy")
                })
        
        # Sort by energy and take top 5 overall
        top_sequences.sort(key=lambda x: x.get("energy", float('inf')))
        top_sequences = top_sequences[:5]
        
        return {
            "success": raw_result.get("success", False),
            "num_sequences_generated": raw_result.get("total_sequences_generated", 0),
            "num_passing_filters": raw_result.get("total_sequences_filtered", 0),
            "best_energy": raw_result.get("best_energy", None),
            "rounds_completed": raw_result.get("rounds_completed", 0),
            "top_sequences": top_sequences,
            "summary": self._generate_summary(raw_result),
            "message": "Use these computationally designed sequences in your binder_data.proposed_peptides"
        }
    
    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable summary for LLM."""
        if not result.get("success"):
            return "Design failed: " + result.get("error_message", "Unknown error")
        
        num_generated = result.get("total_sequences_generated", 0)
        num_passing = result.get("total_sequences_filtered", 0)
        success_rate = (num_passing / num_generated * 100) if num_generated > 0 else 0
        best_energy = result.get("best_energy", None)
        
        summary = (
            f"BindCraft generated {num_generated} binder sequences across {result.get('rounds_completed', 0)} rounds. "
            f"{num_passing} sequences passed quality filters ({success_rate:.1f}% success rate). "
        )
        
        if best_energy and best_energy != float('inf'):
            summary += f"Best binding energy: {best_energy:.2f} kcal/mol. "
        
        summary += "Top sequences are provided in the results."
        
        return summary

