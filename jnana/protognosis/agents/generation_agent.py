"""
Generation Agent for ProtoGnosis.

This agent is responsible for generating initial research hypotheses using
various strategies like literature exploration, scientific debate, etc.
"""

import json
import time
import random
from typing import Dict, List, Optional, Any
import logging
import os

from ..core.agent_core import Agent, Task, ResearchHypothesis, ContextMemory
from ..core.llm_interface import LLMInterface

# Import prompt templates
try:
    from ..prompts.generation_agent_prompts import (
        create_scientific_debate_prompt,
        create_literature_exploration_prompt,
        create_assumptions_identification_prompt,
        create_research_expansion_prompt
    )
    EXTERNAL_PROMPTS = True
except ImportError:
    EXTERNAL_PROMPTS = False
    logging.warning("External prompt templates not found. Using built-in templates.")


class GenerationAgent(Agent):
    """
    Agent responsible for generating initial research hypotheses.

    This agent uses various strategies to generate novel hypotheses:
    - Literature exploration via web search
    - Simulated scientific debates
    - Iterative assumptions identification
    - Research expansion

    Can optionally use tools (e.g., BindCraft) during hypothesis generation.
    """

    def __init__(self, agent_id: str, llm: LLMInterface, memory: ContextMemory, tool_registry=None):
        """
        Initialize the generation agent.

        Args:
            agent_id: Unique identifier for this agent
            llm: LLM interface for generation
            memory: Context memory for storing hypotheses
            tool_registry: Optional tool registry for function calling
        """
        super().__init__(agent_id, "generation", llm, memory)
        self.tool_registry = tool_registry
    
    async def execute_task(self, task: Task) -> Dict:
        """Execute a task to generate a hypothesis."""
        task_type = task.task_type

        if task_type == "generate_hypothesis":
            return await self._generate_hypothesis(task)
        elif task_type == "simulate_debate":
            return await self._simulate_debate(task)
        elif task_type == "improve_hypothesis":
            return await self._improve_hypothesis(task)
        else:
            raise ValueError(f"Unsupported task type for GenerationAgent: {task_type}")
    
    async def _generate_hypothesis(self, task: Task) -> Dict:
        """Generate a new research hypothesis."""
        self.logger.info(f"Generating hypothesis for task {task.task_id}")

        # Get research goal from memory
        research_goal = self.memory.metadata.get("research_goal", "")
        plan_config = self.memory.metadata.get("research_plan_config", {})

        if not research_goal:
            raise ValueError("No research goal found in memory")

        # Detect if this is a binder design task
        is_binder_design = ('target_sequence' in plan_config or
                           'binder_sequence' in plan_config)
        is_binder_design = True

        # Determine generation strategy based on task parameters or randomly select one
        strategy = task.params.get("strategy")
        if not strategy:
            strategies = ["literature_exploration", "scientific_debate", "assumptions_identification", "research_expansion", "binder_gen"]
            strategy = random.choice(strategies)

        # Use the appropriate prompt template based on the strategy
        if EXTERNAL_PROMPTS:
            if strategy == "literature_exploration":
                prompt = create_literature_exploration_prompt(research_goal, plan_config)
            elif strategy == "scientific_debate":
                prompt = create_scientific_debate_prompt(research_goal, plan_config)
            elif strategy == "assumptions_identification":
                prompt = create_assumptions_identification_prompt(research_goal, plan_config)
            elif strategy == "research_expansion":
                # Get top-ranked hypotheses to build upon
                top_hypotheses = self.memory.get_top_hypotheses(3)
                top_summaries = "\n".join([f"- {h.summary}" for h in top_hypotheses]) if top_hypotheses else "No existing hypotheses yet."
                prompt = create_research_expansion_prompt(research_goal, plan_config, top_summaries)
            else:
                raise ValueError(f"Unknown generation strategy: {strategy}")
        else:
            # Use built-in prompt templates
            if strategy == "literature_exploration":
                prompt = self._create_literature_exploration_prompt(research_goal, plan_config, is_binder_design)
            elif strategy == "scientific_debate":
                prompt = self._create_scientific_debate_prompt(research_goal, plan_config, is_binder_design)
            elif strategy == "assumptions_identification":
                prompt = self._create_assumptions_identification_prompt(research_goal, plan_config, is_binder_design)
            elif strategy == "research_expansion":
                # Get top-ranked hypotheses to build upon
                top_hypotheses = self.memory.get_top_hypotheses(3)
                top_summaries = "\n".join([f"- {h.summary}" for h in top_hypotheses]) if top_hypotheses else "No existing hypotheses yet."
                prompt = self._create_research_expansion_prompt(research_goal, plan_config, top_summaries, is_binder_design)
            elif strategy == "binder_gen":
                prompt = self._create_literature_exploration_prompt(research_goal, plan_config, is_binder_design=True)
            else:
                raise ValueError(f"Unknown generation strategy: {strategy}")

        # Generate hypothesis using the LLM
        system_prompt = self.fill_prompt_template("system",
                                                agent_type="generation",
                                                role="generate novel research hypotheses")

        # Define the expected output schema based on hypothesis type
        if is_binder_design:
            schema = {
                "hypothesis": {
                    "title": "string",
                    "content": "string",
                    "summary": "string",
                    "key_novelty_aspects": ["string"],
                    "testable_predictions": ["string"]
                },
                "binder_data": {
                    "target_name": "string",
                    "target_sequence": "string",
                    "proposed_peptides": [
                        {
                            "sequence": "string",
                            "source": "string",
                            #"rationale": "string",
                            "peptide_id": "string"
                        }
                    ],
                    "literature_references": ["string"],
                    "binding_affinity_goal": "string",
                    "clinical_context": "string"
                },
                "explanation": "string",
                "generation_strategy": "string"
            }
        else:
            schema = {
                "hypothesis": {
                    "title": "string",
                    "content": "string",
                    "summary": "string",
                    "key_novelty_aspects": ["string"],
                    "testable_predictions": ["string"]
                },
                "explanation": "string",
                "generation_strategy": "string"
            }
        
        try:
            # Check if tools are available
            tools = None
            is_binder_design = True
            if self.tool_registry and is_binder_design:
                tool_schemas = self.tool_registry.get_tool_schemas()
                if tool_schemas:
                    tools = tool_schemas
                    self.logger.info(f"Tools available for LLM: {[t['function']['name'] for t in tools]}")

            # STEP 1: Generate the initial hypothesis (structured JSON, no tool calling)
            response_data = self.llm.generate_with_json_output(
                prompt,
                schema,
                system_prompt=system_prompt,
                tools=tools  # Pass tools (logged but not used in json_object mode)
            )

            # Unpack the response data
            if isinstance(response_data, tuple) and len(response_data) == 3:
                response, prompt_tokens, completion_tokens = response_data

                # Update token counts
                self.total_calls += 1
                self.total_prompt_tokens += int(prompt_tokens)
                self.total_completion_tokens += int(completion_tokens)
            else:
                # Handle the case where the response is not a tuple
                response = response_data
                self.total_calls += 1
            
            # DEBUG: Log the response structure
            self.logger.info(f"LLM response keys: {list(response.keys())}")
            if "binder_data" in response:
                self.logger.info(f"binder_data keys: {list(response['binder_data'].keys())}")
                self.logger.info(f"target_sequence in binder_data: {response['binder_data'].get('target_sequence', 'NOT FOUND')[:50]}...")

            # Create a new hypothesis object
            metadata = {
                "title": response["hypothesis"]["title"],
                "key_novelty_aspects": response["hypothesis"]["key_novelty_aspects"],
                "testable_predictions": response["hypothesis"]["testable_predictions"],
                "generation_strategy": response["generation_strategy"],
                "explanation": response["explanation"]
            }

            # Add binder data if present
            if "binder_data" in response:
                metadata["binder_data"] = response["binder_data"]
                self.logger.info(f"Binder data included in hypothesis: {len(response['binder_data'].get('proposed_peptides', []))} peptides proposed")
                self.logger.info(f"Stored binder_data in metadata with keys: {list(metadata['binder_data'].keys())}")

            # STEP 2: Ask LLM if it wants to use tools (two-step approach for tool calling)
            if tools and is_binder_design and hasattr(self.llm, 'generate_with_tools'):
                try:
                    tool_results = await self._call_tools_if_needed(response, tools, research_goal)
                    if tool_results:
                        # Incorporate tool results into binder_data
                        if "binder_data" not in metadata:
                            metadata["binder_data"] = {
                                "target_sequence": response.get("binder_data", {}).get("target_sequence", ""),
                                "proposed_peptides": [],
                                "literature_references": []
                            }

                        # Add tool-generated peptides
                        for result in tool_results:
                            if result.get("success") and result.get("top_sequences"):
                                for seq_data in result["top_sequences"]:
                                    metadata["binder_data"]["proposed_peptides"].append({
                                        "sequence": seq_data["sequence"],
                                        "source": "computational:bindcraft",
                                        "rationale": f"Generated by BindCraft with pLDDT={seq_data.get('plddt', 'N/A')}, pAE={seq_data.get('pae', 'N/A')}",
                                        "confidence_score": seq_data.get("plddt", 0.0) / 100.0,
                                        "tool_metadata": {
                                            "plddt": seq_data.get("plddt"),
                                            "pae": seq_data.get("pae"),
                                            "ptm": seq_data.get("ptm"),
                                            "i_ptm": seq_data.get("i_ptm")
                                        }
                                    })

                        self.logger.info(f"✓ Tool calling successful! Added {len(tool_results)} tool-generated sequences")
                        metadata["tool_calls_made"] = True
                        metadata["tool_call_count"] = len(tool_results)
                except Exception as e:
                    self.logger.warning(f"Tool calling failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue without tool results

            hypothesis = ResearchHypothesis(
                content=response["hypothesis"]["content"],
                summary=response["hypothesis"]["summary"],
                agent_id=self.agent_id,
                metadata=metadata
            )
            
            # Add the hypothesis to memory
            self.memory.add_hypothesis(hypothesis)

            # Update agent state
            try:
                self.logger.info(f"Updating agent state for {self.agent_id}")
                agent_state = self.memory.get_agent_state(self.agent_id) or {}
                self.logger.info(f"Current agent state: {agent_state}")
                agent_state.update({
                    "last_activity": time.time(),
                    "hypotheses_generated": agent_state.get("hypotheses_generated", 0) + 1,
                    "last_strategy": strategy,
                    "last_hypothesis_id": hypothesis.hypothesis_id,
                    "total_tasks_completed": agent_state.get("total_tasks_completed", 0) + 1
                })
                self.logger.info(f"Updated agent state: {agent_state}")
                self.memory.set_agent_state(self.agent_id, agent_state)
                self.logger.info(f"Agent state saved for {self.agent_id}")
            except Exception as e:
                self.logger.error(f"Error updating agent state: {e}")
                # Don't raise - this shouldn't fail the task

            # Create dataset for this generation task
            try:
                self.logger.info(f"Creating dataset for task {task.task_id}")
                dataset = {
                    "task_id": task.task_id,
                    "agent_id": self.agent_id,
                    "strategy": strategy,
                    "research_goal": research_goal,
                    "hypothesis_generated": hypothesis.hypothesis_id,
                    "generation_time": time.time(),
                    "input_parameters": task.params,
                    "output_quality_metrics": {
                        "content_length": len(hypothesis.content),
                        "summary_length": len(hypothesis.summary),
                        "strategy_alignment": 1.0,  # Could be computed based on strategy adherence
                        "novelty_aspects_count": len(response["hypothesis"].get("key_novelty_aspects", [])),
                        "testable_predictions_count": len(response["hypothesis"].get("testable_predictions", []))
                    }
                }
                self.memory.set_dataset(task.task_id, dataset)
                self.logger.info(f"Dataset created and saved for task {task.task_id}")
            except Exception as e:
                self.logger.error(f"Error creating dataset: {e}")
                # Don't raise - this shouldn't fail the task

            return {
                "hypothesis_id": hypothesis.hypothesis_id,
                "summary": hypothesis.summary,
                "strategy": strategy
            }
            
        except Exception as e:
            self.logger.error(f"Error generating hypothesis: {str(e)}")
            raise

    async def _call_tools_if_needed(self, hypothesis_response: Dict, tools: List[Dict], research_goal: str) -> List[Dict]:
        """
        Two-step approach: Ask LLM if it wants to call tools, then execute them.

        Args:
            hypothesis_response: The initial hypothesis response from LLM
            tools: List of available tool schemas
            research_goal: The research goal

        Returns:
            List of tool execution results (empty if no tools called)
        """
        self.logger.info("🔧 Step 2: Asking LLM if it wants to use computational tools...")

        # Extract target sequence from hypothesis
        target_sequence = hypothesis_response.get("binder_data", {}).get("target_sequence", "")
        if not target_sequence:
            self.logger.info("No target sequence found, skipping tool calling")
            return []

        # Create a prompt asking if LLM wants to use tools
        tool_decision_prompt = f"""
You have generated a hypothesis for: {research_goal}

Target protein sequence: {target_sequence[:100]}...

You proposed {len(hypothesis_response.get('binder_data', {}).get('proposed_peptides', []))} peptide sequences from literature/homology/de-novo design.

**Question**: Would you like to use computational design tools to generate additional binder sequences?

Available tools:
- bindcraft_design: Uses ProteinMPNN to computationally design binders for the target protein

If you want to use computational design, call the bindcraft_design tool with appropriate parameters.
If you're satisfied with the literature-based sequences, you don't need to call any tools.
"""

        try:
            # Call LLM with tool calling support
            tool_response = self.llm.generate_with_tools(
                prompt=tool_decision_prompt,
                tools=tools,
                system_prompt="You are a computational biologist deciding whether to use computational tools.",
                temperature=0.3,
                max_tokens=500
            )

            self.logger.info(f"Tool decision response: finish_reason={tool_response.get('finish_reason')}")

            # Check if LLM made tool calls
            tool_calls = tool_response.get('tool_calls', [])
            if not tool_calls:
                self.logger.info("❌ LLM decided NOT to use computational tools")
                # Log the LLM's reasoning if available
                if tool_response.get('content'):
                    self.logger.debug(f"LLM reasoning: {tool_response.get('content')}")
                return []

            self.logger.info(f"✅ LLM DECIDED TO USE COMPUTATIONAL TOOLS! Requested {len(tool_calls)} tool call(s)")

            # Log details about each tool call decision
            for i, tool_call in enumerate(tool_calls, 1):
                self.logger.info(f"  Tool call {i}: {tool_call.get('name', 'unknown')}")
                self.logger.debug(f"  Arguments: {tool_call.get('arguments', {})}")

            # Execute each tool call
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['arguments']

                self.logger.info(f"🔧 Executing tool: {tool_name} with args: {tool_args}")

                # Execute the tool via tool registry
                result = await self.tool_registry.execute_tool(tool_name, **tool_args)
                results.append(result)

                if result.get('success'):
                    num_sequences = len(result.get('top_sequences', []))
                    self.logger.info(f"✓ Tool {tool_name} executed successfully! Generated {num_sequences} sequences")
                else:
                    self.logger.error(f"✗ Tool {tool_name} failed: {result.get('error', 'Unknown error')}")

            # Summary
            successful_results = [r for r in results if r.get('success')]
            if successful_results:
                total_sequences = sum(len(r.get('top_sequences', [])) for r in successful_results)
                self.logger.info(f"🎉 Tool calling complete! Generated {total_sequences} computational sequences from {len(successful_results)} tool(s)")

            return results

        except NotImplementedError:
            self.logger.warning("LLM provider doesn't support tool calling (generate_with_tools not implemented)")
            return []
        except Exception as e:
            self.logger.error(f"Error during tool calling: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def _simulate_debate(self, task: Task) -> Dict:
        """Simulate a scientific debate to refine a hypothesis."""
        self.logger.info(f"Simulating debate for task {task.task_id}")
        
        # Get research goal and hypothesis
        research_goal = self.memory.metadata.get("research_goal", "")
        hypothesis_id = task.params.get("hypothesis_id")
        
        if not hypothesis_id:
            raise ValueError("No hypothesis_id provided for debate task")
        
        hypothesis = self.memory.get_hypothesis(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found in memory")
        
        # Create debate prompt
        prompt = f"""
        You are simulating a scientific debate among experts to refine and improve a research hypothesis.
        
        Research goal: {research_goal}
        
        Original hypothesis:
        {hypothesis.content}
        
        Please simulate a debate among multiple scientific experts (at least 3) with diverse perspectives 
        on this hypothesis. Each expert should provide constructive criticism, suggest improvements, 
        and identify potential flaws or limitations.
        
        After the debate, synthesize the key insights and generate an improved version of the hypothesis 
        that addresses the main concerns raised.
        """
        
        system_prompt = "You are an AI co-scientist that specializes in simulating scientific debates to improve research hypotheses."
        
        schema = {
            "debate": {
                "participants": [{"name": "string", "expertise": "string", "perspective": "string"}],
                "rounds": [{"speaker": "string", "argument": "string"}]
            },
            "key_insights": ["string"],
            "improved_hypothesis": {
                "content": "string",
                "summary": "string"
            }
        }
        
        try:
            response, prompt_tokens, completion_tokens = self.llm.generate_with_json_output(prompt, schema, system_prompt=system_prompt)
            
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            # Create a new hypothesis based on the improved version
            improved_hypothesis = ResearchHypothesis(
                content=response["improved_hypothesis"]["content"],
                summary=response["improved_hypothesis"]["summary"],
                agent_id=self.agent_id,
                metadata={
                    "original_hypothesis_id": hypothesis_id,
                    "debate": response["debate"],
                    "key_insights": response["key_insights"],
                    "generation_strategy": "scientific_debate"
                }
            )
            
            # Add the improved hypothesis to memory
            self.memory.add_hypothesis(improved_hypothesis)
            
            return {
                "original_hypothesis_id": hypothesis_id,
                "improved_hypothesis_id": improved_hypothesis.hypothesis_id,
                "key_insights": response["key_insights"]
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating debate: {str(e)}")
            raise

    async def _improve_hypothesis(self, task: Task) -> Dict:
        """
        Improve a hypothesis based on experimental results from BindCraft and MD simulations.

        This method evaluates experimental results, decides whether to continue optimization
        or complete the process, and suggests new parameters for the next round if continuing.

        Args:
            task: Task containing hypothesis_id and experimental_results

        Returns:
            Dictionary with decision status, reasoning, and new parameters (if continuing)
        """
        self.logger.info(f"Improving hypothesis for task {task.task_id}")

        # Get hypothesis and experimental results from task params
        hypothesis_id = task.params.get("hypothesis_id")
        experimental_results = task.params.get("experimental_results", {})

        if not hypothesis_id:
            raise ValueError("No hypothesis_id provided for improve_hypothesis task")

        hypothesis = self.memory.get_hypothesis(hypothesis_id)
        if not hypothesis:
            raise ValueError(f"Hypothesis {hypothesis_id} not found in memory")

        # Get research goal and plan config
        research_goal = self.memory.metadata.get("research_goal", "")
        plan_config = self.memory.metadata.get("research_plan_config", {})

        # Extract binder data from hypothesis metadata
        binder_data = hypothesis.metadata.get("binder_data", {})

        # Format experimental results for the prompt
        bindcraft_results = experimental_results.get("bindcraft", {})
        md_results = experimental_results.get("md", {})

        # Create the improvement prompt
        prompt = self._create_improvement_prompt(
            research_goal=research_goal,
            hypothesis_content=hypothesis.content,
            binder_data=binder_data,
            bindcraft_results=bindcraft_results,
            md_results=md_results
        )

        system_prompt = (
            "You are an AI co-scientist specializing in iterative optimization of binder design experiments. "
            "Your role is to evaluate experimental results, decide whether optimization goals have been met, "
            "and suggest parameter adjustments for the next round if needed."
        )

        # Define the expected output schema
        schema = {
            "evaluation": {
                "success_rate": "float",
                "meets_threshold": "boolean",
                "key_findings": ["string"],
                "strengths": ["string"],
                "weaknesses": ["string"]
            },
            "decision": {
                "status": "string",  # "continue" or "complete"
                "reasoning": "string",
                "confidence": "float"
            },
            "new_parameters": {
                "num_seqs": "integer",
                "sampling_temp": "float",
                "qc_filters": {
                    "multiplicity": "float",
                    "diversity": "float",
                    "repeat": "float",
                    "charge_ratio": "float",
                    "check_bad_motifs": "boolean",
                    "net_charge": "float",
                    "bad_terminus": "boolean",
                    "hydrophobicity": "float",
                    "passing": "float"
                },
                "structure_filters": {
                    "energy": "float",
                    "rmsd": "float",
                    "rmsf": "float",
                    "passing": "float"
                },
                "simulation_time": "float"
            },
            "parameter_reasoning": {
                "num_seqs_reason": "string",
                "sampling_temp_reason": "string",
                "qc_filters_reason": "string",
                "structure_filters_reason": "string",
                "simulation_time_reason": "string"
            }
        }

        try:
            # Generate the improvement decision with the LLM
            response_data = self.llm.generate_with_json_output(prompt, schema, system_prompt=system_prompt)

            # Unpack the response data
            if isinstance(response_data, tuple) and len(response_data) == 3:
                response, prompt_tokens, completion_tokens = response_data

                # Update token counts
                self.total_calls += 1
                self.total_prompt_tokens += int(prompt_tokens)
                self.total_completion_tokens += int(completion_tokens)
            else:
                response = response_data
                self.total_calls += 1

            # Validate and adjust parameters to be within bounds
            validated_params = self._validate_parameters(response.get("new_parameters", {}))
            response["new_parameters"] = validated_params

            # Log the decision
            decision_status = response["decision"]["status"]
            self.logger.info(f"Improvement decision: {decision_status} (confidence: {response['decision']['confidence']})")
            self.logger.info(f"Reasoning: {response['decision']['reasoning']}")

            # Update agent state
            try:
                agent_state = self.memory.get_agent_state(self.agent_id) or {}
                agent_state.update({
                    "last_activity": time.time(),
                    "improvements_completed": agent_state.get("improvements_completed", 0) + 1,
                    "last_improvement_decision": decision_status,
                    "total_tasks_completed": agent_state.get("total_tasks_completed", 0) + 1
                })
                self.memory.set_agent_state(self.agent_id, agent_state)
            except Exception as e:
                self.logger.error(f"Error updating agent state: {e}")

            return {
                "hypothesis_id": hypothesis_id,
                "decision": response["decision"],
                "evaluation": response["evaluation"],
                "new_parameters": response["new_parameters"],
                "parameter_reasoning": response["parameter_reasoning"]
            }

        except Exception as e:
            self.logger.error(f"Error improving hypothesis: {str(e)}")
            raise

    def _create_literature_exploration_prompt(self, research_goal: str, plan_config: Dict, is_binder_design: bool = False) -> str:
        """Create a prompt for literature exploration-based hypothesis generation."""
        constraints = ', '.join(plan_config.get('constraints', []))
        preferences = ', '.join(plan_config.get('preferences', []))

        base_prompt = (
            f"You are an AI co-scientist specializing in generating novel research hypotheses based on literature exploration.\n\n"
            f"Research goal:\n{research_goal}\n\n"
        )

        if is_binder_design:
            target_seq = plan_config.get('target_sequence', 'UNKNOWN')
            binder_seq = plan_config.get('binder_sequence', '')

            base_prompt += (
                f"This is a BINDER DESIGN task. Please recommend starting peptides for the workflow.\n Only use custom peptides if real clinical information is available in a real literature. Do not hallucinate! Otherwise use either:\n\n"
                "**Default peptides**: \n\n"
                " - affibody: VDNKFNKELSVAGREIVTLPNLNDPQKKAFIFSLWDDPSQSANLLAEAKKLNDAQAPK"
                "- nanobody: AQVQLQESGGGLVQAGGSLRLSCAASERTFSTYAMGWFRQAPGREREFLAQINWSGTTTYYAESVKDRTTISRDNAKNTVYLEMNNLNADDTGIYFCAAHPQRGWGSTLGWTYWGQGTQVTVSSGGGGSGGGKPIPNPLLGLDSTRTGHHHHHH"
                "- affitin: MRGSHHHHHHGSVKVKFVSSGEEKEVDTSKIKKVWRNLTKYGTIVQFTYDDNGKTGRGYVRELDAPKELLDMLARAEGKLN"
                "Your hypothesis must include:\n\n"
                f"1. **Target Information:**\n"
                f"   - Target protein name\n"
                f"   - Target sequence: {target_seq}\n\n"
                f"2. **Proposed Peptide Binders (3-5 candidates):**\n"
                f"   For each peptide, provide:\n"
                f"   - Amino acid sequence (single-letter code)\n"
                f"   - Source (e.g., 'literature:PMID12345', 'homology:ProteinX', 'de_novo')\n"
                f"   - Rationale for why this peptide might bind the target\n"
                f"   - Unique peptide ID including whether this is affibody, nanobody or affitin (e.g., 'pep_001', 'pep_002')\n\n"
                f"3. **Literature Support:**\n"
                f"   - List of relevant literature references (PubMed IDs, DOIs, or citations)\n"
                f"   - Key findings from literature that support your peptide choices\n\n"
                f"4. **Design Goals:**\n"
                f"   - Desired binding affinity (e.g., 'nanomolar', 'sub-micromolar')\n"
                f"   - Clinical context or application (if applicable)\n\n"
            )

            if binder_seq:
                base_prompt += f"   - Known binder sequence for reference: {binder_seq}\n\n"

            # Add note about tools if available
            if self.tool_registry and self.tool_registry.has_tool('bindcraft_design'):
                base_prompt += (
                    f"**OPTIONAL:** You have access to computational design tools:\n"
                    f"   - bindcraft_design: Generate binders computationally using ProteinMPNN\n"
                    f"   You can call this tool if you want to supplement literature-based sequences\n"
                    f"   with computationally designed ones. However, you can also propose sequences\n"
                    f"   purely from literature if you prefer.\n\n"
                )

            base_prompt += (
                f"Follow these steps:\n"
                f"1. Conduct a thorough literature review on peptide binders for this target\n"
                f"2. Identify known binding motifs, interaction patterns, and successful binder designs\n"
                f"3. Propose 3-5 novel peptide sequences based on literature insights\n"
                f"4. Provide strong rationale for each peptide based on structural/functional knowledge\n"
                f"5. Include all information in a 'binder_data' section of your response\n\n"
            )
        else:
            base_prompt += (
                f"Your task is to generate a novel research hypothesis that addresses this goal.\n\n"
                f"Follow these steps:\n"
                f"1. Imagine you have conducted a thorough literature review in this research area\n"
                f"2. Identify key findings, methods, and theories from the literature\n"
                f"3. Look for gaps, contradictions, or unexplored connections in existing research\n"
                f"4. Develop a novel hypothesis that addresses these gaps or connects disparate findings\n"
                f"5. Ensure the hypothesis is specific, testable, and explain its significance\n\n"
            )

        base_prompt += (
            f"Constraints to consider:\n{constraints}\n\n"
            f"Preferences to incorporate:\n{preferences}\n\n"
            f"The final hypothesis should be well-grounded in existing literature while proposing "
            f"a novel direction that advances understanding in this research area."
        )

        return base_prompt
    
    def _create_scientific_debate_prompt(self, research_goal: str, plan_config: Dict, is_binder_design: bool = False) -> str:
        """Create a prompt for scientific debate-based hypothesis generation."""
        constraints = ', '.join(plan_config.get('constraints', []))
        preferences = ', '.join(plan_config.get('preferences', []))

        base_prompt = (
            f"You are an AI co-scientist specializing in generating novel research hypotheses through simulated scientific debates.\n\n"
            f"Research goal:\n{research_goal}\n\n"
        )

        if is_binder_design:
            target_seq = plan_config.get('target_sequence', 'UNKNOWN')

            base_prompt += (
                f"This is a BINDER DESIGN task. The debate should focus on designing peptide binders.\n\n"
                f"Target sequence: {target_seq}\n\n"
                f"Your task is to simulate a debate among experts (structural biologist, peptide chemist, "
                f"computational biologist, medicinal chemist) to propose 3-5 peptide binder candidates.\n\n"
                f"Follow these steps:\n"
                f"1. Create expert personas with different perspectives on binder design\n"
                f"2. Have each expert propose peptide sequences with rationale\n"
                f"3. Simulate debate where experts critique and refine each other's proposals\n"
                f"4. Synthesize the best 3-5 peptide candidates from the debate\n"
                f"5. For each peptide, include: sequence, source, rationale, peptide_id\n"
                f"6. Include literature references and binding affinity goals\n"
                f"7. Ensure all information is in a 'binder_data' section\n\n"
            )
        else:
            base_prompt += (
                f"Your task is to simulate a scientific debate among experts with different perspectives to "
                f"generate a novel research hypothesis that addresses this goal.\n\n"
                f"Follow these steps:\n"
                f"1. Create 3-5 expert personas with different backgrounds and perspectives relevant to this research area\n"
                f"2. Simulate a scientific debate where each expert proposes initial ideas and critiques others' proposals\n"
                f"3. Allow the debate to evolve through multiple rounds, refining ideas and addressing criticisms\n"
                f"4. Synthesize the most promising ideas from the debate into a coherent hypothesis\n"
                f"5. Ensure the final hypothesis is specific, testable, and explains its significance\n\n"
            )

        base_prompt += (
            f"Constraints to consider:\n{constraints}\n\n"
            f"Preferences to incorporate:\n{preferences}\n\n"
            f"The final hypothesis should represent a consensus emerging from diverse scientific perspectives, "
            f"addressing potential criticisms and limitations while maintaining novelty and testability."
        )

        return base_prompt
    
    def _create_assumptions_identification_prompt(self, research_goal: str, plan_config: Dict, is_binder_design: bool = False) -> str:
        """Create a prompt for assumptions identification-based hypothesis generation."""
        constraints = ', '.join(plan_config.get('constraints', []))
        preferences = ', '.join(plan_config.get('preferences', []))

        base_prompt = (
            f"You are an AI co-scientist specializing in generating novel research hypotheses through identification of key assumptions.\n\n"
            f"Research goal:\n{research_goal}\n\n"
        )

        if is_binder_design:
            target_seq = plan_config.get('target_sequence', 'UNKNOWN')

            base_prompt += (
                f"This is a BINDER DESIGN task. Challenge assumptions about peptide binder design.\n\n"
                f"Target sequence: {target_seq}\n\n"
                f"Your task is to identify and challenge assumptions in binder design to propose novel peptides.\n\n"
                f"Follow these steps:\n"
                f"1. Identify 3-5 key assumptions in peptide binder design (e.g., size limits, charge requirements, binding motifs)\n"
                f"2. Analyze which assumptions could be challenged based on recent literature\n"
                f"3. Propose 3-5 peptide sequences that challenge these assumptions\n"
                f"4. For each peptide, provide: sequence, source, rationale, peptide_id\n"
                f"5. Include literature references supporting your unconventional approach\n"
                f"6. Specify binding affinity goals and clinical context\n"
                f"7. Ensure all information is in a 'binder_data' section\n\n"
            )
        else:
            base_prompt += (
                f"Your task is to generate a novel research hypothesis by identifying and challenging key assumptions "
                f"in the current understanding of this research area.\n\n"
                f"Follow these steps:\n"
                f"1. Identify 3-5 key assumptions that underlie current thinking in this research area\n"
                f"2. For each assumption, analyze its validity and evidence supporting or contradicting it\n"
                f"3. Select one or more assumptions that could be productively challenged\n"
                f"4. Develop a novel hypothesis that challenges or reframes these assumptions\n"
                f"5. Ensure the hypothesis is specific, testable, and explain its significance\n\n"
            )

        base_prompt += (
            f"Constraints to consider:\n{constraints}\n\n"
            f"Preferences to incorporate:\n{preferences}\n\n"
            f"The final hypothesis should represent a meaningful challenge to existing assumptions, "
            f"opening new avenues for research while remaining scientifically plausible."
        )

        return base_prompt
    
    def _create_research_expansion_prompt(self, research_goal: str, plan_config: Dict, top_summaries: str = "", is_binder_design: bool = False) -> str:
        """Create a prompt for research expansion-based hypothesis generation."""
        # If top_summaries is not provided, get top-ranked hypotheses to build upon
        if not top_summaries:
            top_hypotheses = self.memory.get_top_hypotheses(3)
            top_summaries = "\n".join([f"- {h.summary}" for h in top_hypotheses]) if top_hypotheses else "No existing hypotheses yet."

        constraints = ', '.join(plan_config.get('constraints', []))
        preferences = ', '.join(plan_config.get('preferences', []))

        if is_binder_design:
            target_seq = plan_config.get('target_sequence', 'UNKNOWN')

            prompt = f"""
        You are generating a new binder design hypothesis that builds upon and expands existing research directions.

        Research goal:
        {research_goal}

        Target sequence: {target_seq}

        Top existing hypotheses:
        {top_summaries}

        Research constraints:
        {constraints}

        Research preferences:
        {preferences}

        This is a BINDER DESIGN task. Please generate a novel hypothesis that:
        1. Builds upon the existing hypotheses
        2. Proposes 3-5 new peptide binder candidates
        3. For each peptide, includes: sequence, source, rationale, peptide_id
        4. Includes literature references
        5. Specifies binding affinity goals and clinical context
        6. Ensures all information is in a 'binder_data' section

        Your hypothesis should expand the research in a promising direction while maintaining scientific rigor.
        """
        else:
            prompt = f"""
        You are generating a new research hypothesis that builds upon and expands existing research directions.

        Research goal:
        {research_goal}

        Top existing hypotheses:
        {top_summaries}

        Research constraints:
        {constraints}

        Research preferences:
        {preferences}

        Please generate a novel research hypothesis that builds upon the existing hypotheses while addressing the research goal.
        Your hypothesis should be specific, testable, and expand the research in a promising direction.
        """

        return prompt

    def _create_improvement_prompt(self, research_goal: str, hypothesis_content: str,
                                   binder_data: Dict, bindcraft_results: Dict,
                                   md_results: Dict) -> str:
        """
        Create a prompt for improving a hypothesis based on experimental results.

        Args:
            research_goal: The original research goal
            hypothesis_content: The hypothesis content
            binder_data: Binder-specific data from hypothesis metadata
            bindcraft_results: Results from BindCraft optimization
            md_results: Results from MD simulations

        Returns:
            Formatted prompt string
        """
        # Extract key metrics
        bc_num_rounds = bindcraft_results.get('num_rounds', 0)
        bc_total_seqs = bindcraft_results.get('total_sequences', 0)
        bc_passing_seqs = bindcraft_results.get('passing_sequences', 0)
        bc_success_rate = bindcraft_results.get('success_rate', 0.0)
        bc_seqs_per_round = bindcraft_results.get('sequences_per_round', [])
        bc_passing_per_round = bindcraft_results.get('passing_per_round', [])
        bc_params = bindcraft_results.get('parameters_used', {})

        md_stable = md_results.get('stable_complexes', 0)
        md_total = md_results.get('total_simulations', 0)
        md_avg_rmsd = md_results.get('avg_rmsd', 0.0)
        md_avg_energy = md_results.get('avg_binding_energy', 0.0)
        md_sim_time = md_results.get('simulation_time', 0)

        # Format round-by-round results
        round_summary = ""
        for i, (total, passing) in enumerate(zip(bc_seqs_per_round, bc_passing_per_round), 1):
            rate = (passing / total * 100) if total > 0 else 0
            round_summary += f"  Round {i}: {passing}/{total} sequences passed ({rate:.1f}%)\n"

        prompt = f"""
You are evaluating the results of a binder design optimization experiment and deciding whether to continue or complete the process.

**Research Goal:**
{research_goal}

**Original Hypothesis:**
{hypothesis_content}

**Target Information:**
- Target: {binder_data.get('target_name', 'Unknown')}
- Target Sequence: {binder_data.get('target_sequence', 'N/A')[:50]}...
- Proposed Peptides: {len(binder_data.get('proposed_peptides', []))} candidates

**BindCraft Optimization Results:**
- Total Rounds: {bc_num_rounds}
- Total Sequences Generated: {bc_total_seqs}
- Sequences Passing QC Filters: {bc_passing_seqs}
- Overall Success Rate: {bc_success_rate:.1%}

Round-by-Round Performance:
{round_summary}

Current Parameters:
- num_seqs: {bc_params.get('num_seqs', 'N/A')}
- sampling_temp: {bc_params.get('sampling_temp', 'N/A')}
- QC filters: {bc_params.get('qc_filters', 'N/A')}
- Structure filters: {bc_params.get('structure_filters', 'N/A')}

**MD Simulation Results:**
- Stable Complexes: {md_stable}/{md_total} ({md_stable/md_total*100 if md_total > 0 else 0:.1f}%)
- Average RMSD: {md_avg_rmsd:.2f} Å
- Average Binding Energy: {md_avg_energy:.2f} kcal/mol
- Simulation Time: {md_sim_time} ns

**Success Criteria:**
- Target: ≥5% of generated sequences should produce stable MD complexes
- Current Achievement: {md_stable/bc_total_seqs*100 if bc_total_seqs > 0 else 0:.2f}%

**Your Task:**

1. **Evaluate** the experimental results:
   - Calculate the actual success rate (stable complexes / total sequences)
   - Determine if the 5% threshold has been met
   - Identify key findings, strengths, and weaknesses

2. **Decide** whether to continue or complete:
   - Status: "continue" or "complete"
   - If success rate ≥ 5%, recommend "complete"
   - If success rate < 5% but showing improvement, recommend "continue"
   - If success rate < 5% and plateauing, recommend "continue" with significant parameter changes
   - Provide clear reasoning and confidence level (0.0-1.0)

3. **Suggest new parameters** (if continuing):
   - **num_seqs** (10-250): Number of sequences to generate per round
     - Increase if need more diversity
     - Decrease if quality is more important than quantity

   - **sampling_temp** (0.1-0.3): Temperature for sequence sampling
     - Increase for more diversity (but potentially lower quality)
     - Decrease for more conservative, higher-quality sequences

   - **qc_filters**: Quality control thresholds
     - multiplicity: Filter for sequence multiplicity
     - diversity: Minimum diversity threshold
     - repeat: Maximum repeat content
     - charge_ratio: Charge distribution ratio
     - check_bad_motifs: Boolean to check for problematic motifs
     - net_charge: Net charge threshold
     - bad_terminus: Boolean to check terminus quality
     - hydrophobicity: Hydrophobicity threshold
     - passing: Overall passing threshold

   - **structure_filters**: Structural quality thresholds
     - energy: Energy threshold (populated by BindCraft)
     - rmsd: RMSD threshold (populated by MDAgent)
     - rmsf: RMSF threshold (populated by MDAgent)
     - passing: Overall passing threshold

   - **simulation_time** (1-100 ns): MD simulation duration
     - Increase for more thorough equilibration
     - Decrease for faster iteration

4. **Provide reasoning** for each parameter change in a concise format.

**Important:**
- Be data-driven in your evaluation
- Consider trends across rounds
- Balance exploration (diversity) vs exploitation (quality)
- Provide specific, actionable parameter recommendations
- Keep reasoning concise but informative
"""

        return prompt

    def _validate_parameters(self, params: Dict) -> Dict:
        """
        Validate and adjust parameters to be within acceptable bounds.

        Args:
            params: Dictionary of parameters to validate

        Returns:
            Dictionary of validated parameters
        """
        validated = params.copy()

        # Validate num_seqs (10-250)
        if 'num_seqs' in validated:
            validated['num_seqs'] = max(10, min(250, int(validated['num_seqs'])))

        # Validate sampling_temp (0.1-0.3)
        if 'sampling_temp' in validated:
            validated['sampling_temp'] = max(0.1, min(0.3, float(validated['sampling_temp'])))

        # Validate simulation_time (1-100 ns)
        if 'simulation_time' in validated:
            validated['simulation_time'] = max(1.0, min(100.0, float(validated['simulation_time'])))

        # Validate qc_filters (ensure all required keys exist and are within reasonable bounds)
        if 'qc_filters' in validated:
            qc = validated['qc_filters']
            # Ensure boolean fields are actually boolean
            if 'check_bad_motifs' in qc:
                qc['check_bad_motifs'] = bool(qc['check_bad_motifs'])
            if 'bad_terminus' in qc:
                qc['bad_terminus'] = bool(qc['bad_terminus'])

            # Ensure numeric fields are within 0-1 range (assuming they're thresholds)
            for key in ['multiplicity', 'diversity', 'repeat', 'charge_ratio', 'net_charge', 'hydrophobicity', 'passing']:
                if key in qc:
                    qc[key] = max(0.0, min(1.0, float(qc[key])))

        # Validate structure_filters (ensure all required keys exist and are within reasonable bounds)
        if 'structure_filters' in validated:
            sf = validated['structure_filters']
            # Ensure numeric fields are reasonable
            for key in ['energy', 'rmsd', 'rmsf', 'passing']:
                if key in sf:
                    sf[key] = float(sf[key])
                    # Energy can be negative, but let's ensure it's reasonable
                    if key == 'energy':
                        sf[key] = max(-1000.0, min(1000.0, sf[key]))
                    # RMSD and RMSF should be positive
                    elif key in ['rmsd', 'rmsf']:
                        sf[key] = max(0.0, min(100.0, sf[key]))
                    # Passing should be 0-1
                    elif key == 'passing':
                        sf[key] = max(0.0, min(1.0, sf[key]))

        self.logger.info(f"Validated parameters: num_seqs={validated.get('num_seqs')}, "
                        f"sampling_temp={validated.get('sampling_temp')}, "
                        f"simulation_time={validated.get('simulation_time')}")

        return validated

