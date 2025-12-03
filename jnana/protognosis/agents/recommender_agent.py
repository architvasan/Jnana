"""
Implementation of the specialized agents in the AI Co-scientist system.
This includes the Generation, Reflection, Ranking, Evolution, Proximity, and Meta-review agents.
"""
import json
import time
import asyncio
import random
from typing import Dict, List, Optional, Any, Union
import logging
import os

from ..core.agent_core import Agent, Task, ResearchHypothesis, ResearchRecommendation, ContextMemory
from ..core.llm_interface import LLMInterface

class RecommenderAgent(Agent):
    """
    Agent responsible for generating recommendations for runs.
    This agent takes in a conclusion from a previously completed run
    and makes multiple recommendations for what to run next (bindcraft, mdsimulation).
    The agent also decides if the same parameters should be used for the next run or
    what the new parameters should be here.

    Can optionally use tools (e.g., BindCraft) it recommends.
    """

    def __init__(self, agent_id: str, llm: LLMInterface, memory: ContextMemory, tool_registry=None):
        """
        Initialize the recommendation agent.

        Args:
            agent_id: Unique identifier for this agent
            llm: LLM interface for generation
            memory: Context memory for storing hypotheses
            tool_registry: Optional tool registry for function calling
        """
        super().__init__(agent_id, "recommender", llm, memory)
        self.tool_registry = tool_registry
    
    async def execute_task(self, task: Task) -> Dict:
        """Execute a task to generate a hypothesis."""
        task_type = task.task_type

        if task_type == "recommend_run":
            return await self._recommend_run(task)
        elif task_type == "plan_run":
            return await self._plan_run(task)
        elif task_type == "execute_run":
            return await self._execute_run(task)
        else:
            raise ValueError(f"Unsupported task type for RecommenderAgent: {task_type}")
    
    async def _recommend_run(self, task: Task) -> Dict:
        """Generate a new research hypothesis."""
        self.logger.info(f"Generating hypothesis for task {task.task_id}")

        # Get research goal from memory
        research_goal = self.memory.metadata.get("research_goal", "")
        plan_config = self.memory.metadata.get("research_plan_config", {})

        if not research_goal:
            raise ValueError("No research goal found in memory")

        # Check whether previous run was binder design or simulation
        previous_run = task.params.get('previous_run')

        # Check what conclusion was reached in previous run
        previous_conclusion = task.params.get('previous_conclusion')


        prompt = self._create_recommendation_prompt(research_goal,
                    previous_run,
                    previous_conclusion)

        # Generate hypothesis using the LLM
        system_prompt = self.fill_prompt_template("system",
                                                agent_type="recommender",
                                                role="Recommend next runs to make")

        # Define the expected output schema based on hypothesis type
        
        schema = {
                "recommendation": {
                    "next_task": "string",
                    "change_parameters": "boolean",
                    "rationale": "string",
                },
            }
        
        # STEP 1: Generate the initial hypothesis (structured JSON, no tool calling)
        response_data = self.llm.generate_with_json_output(
            prompt,
            schema,
            system_prompt=system_prompt,
            tools=None  # Pass tools (logged but not used in json_object mode)
        )
        self.logger.info(f"Recommender response: {response_data}")
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

        # Create a new hypothesis object
        metadata = {
            "next_task": response["recommendation"]["next_task"],
            "change_parameters": response["recommendation"]["change_parameters"],
            "rationale": response["recommendation"]["rationale"],
        }

        recommendation_hyp = ResearchRecommendation(
            content=response["recommendation"]["next_task"],
            summary=response["recommendation"]["rationale"],
            agent_id=self.agent_id,
            metadata=metadata
        )
        
        # Add the hypothesis to memory
        self.memory.add_recommendation(recommendation_hyp)

        # Update agent state
        try:
            self.logger.info(f"Updating agent state for {self.agent_id}")
            agent_state = self.memory.get_agent_state(self.agent_id) or {}
            self.logger.info(f"Current agent state: {agent_state}")
            agent_state.update({
                "recommendations_generated": agent_state.get("recommendations_generated", 0) + 1,
                "previous_run": previous_run,
                "last_recommendation_id": recommendation_hyp.recommendation_id,
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
                "research_goal": research_goal,
                "recommendation_generated": recommendation_hyp.recommendation_id,
                "generation_time": time.time(),
                "input_parameters": task.params,
                "output_quality_metrics": {
                    "content_length": len(recommendation_hyp.content),
                    "summary_length": len(recommendation_hyp.summary),
                    "strategy_alignment": 1.0,  # Could be computed based on strategy adherence
                }
            }
            self.memory.set_dataset(task.task_id, dataset)
            self.logger.info(f"Dataset created and saved for task {task.task_id}")
        except Exception as e:
            self.logger.error(f"Error creating dataset: {e}")
            # Don't raise - this shouldn't fail the task

        return {
            "recomendation_id": recommendation_hyp.recommendation_id,
            "summary": recommendation_hyp.summary,
        }

    async def _plan_run(self, task: Task) -> Dict:
        """Generate a new research plan."""
        """task should include information about:
                new_run_type
                old_run_type
                recommendation
                old_config
        """
        self.logger.info(f"Generating run plan for task {task.task_id}")

        # Get research goal from memory
        research_goal = self.memory.metadata.get("research_goal", "")
        #plan_config = self.memory.metadata.get("research_plan_config", {})

        if not research_goal:
            raise ValueError("No research goal found in memory")

        # Check whether previous run was binder design or simulation
        previous_run = task.params.get('previous_run')
        recommendation = task.params.get('recommendation')
        #old_run_type = previous_run['run_type']
        #old_config = previous_run['config']

        prompt = self._create_research_planning_prompt(research_goal,
                    previous_run,
                    recommendation)

        # Generate hypothesis using the LLM
        system_prompt = self.fill_prompt_template("system",
                                                agent_type="recommender",
                                                role="Recommend next runs to make")

        # Define the expected output schema based on hypothesis type
        
        schema = {
                "next_task": "string",
                "new_config": self._create_schema_from_config(previous_run["config"]),
                "rationale": "string"
            }
            
        # STEP 1: Generate the initial hypothesis (structured JSON, no tool calling)
        response_data = self.llm.generate_with_json_output(
            prompt,
            schema,
            system_prompt=system_prompt,
            tools=None  # Pass tools (logged but not used in json_object mode)
        )
        self.logger.info(f"Recommender response: {response_data}")
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

        # Create a new hypothesis object
        metadata = {
            "next_task": response["next_task"],
            "new_config": response["new_config"],
            "rationale": response["rationale"]
        }

        recommendation_hyp = ResearchRecommendation(
            content=response["rationale"],
            summary=response["rationale"],
            agent_id=self.agent_id,
            metadata=metadata
        )
        
        # Add the hypothesis to memory
        self.memory.add_recommendation(recommendation_hyp)

        # Update agent state
        try:
            self.logger.info(f"Updating agent state for {self.agent_id}")
            agent_state = self.memory.get_agent_state(self.agent_id) or {}
            self.logger.info(f"Current agent state: {agent_state}")
            agent_state.update({
                "recommendations_generated": agent_state.get("recommendations_generated", 0) + 1,
                "previous_run": previous_run,
                "last_recommendation_id": recommendation_hyp.recommendation_id,
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
                "research_goal": research_goal,
                "recommendation_generated": recommendation_hyp.recommendation_id,
                "generation_time": time.time(),
                "input_parameters": task.params,
                "output_quality_metrics": {
                    "content_length": len(recommendation_hyp.content),
                    "summary_length": len(recommendation_hyp.summary),
                    "strategy_alignment": 1.0,  # Could be computed based on strategy adherence
                }
            }
            self.memory.set_dataset(task.task_id, dataset)
            self.logger.info(f"Dataset created and saved for task {task.task_id}")
        except Exception as e:
            self.logger.error(f"Error creating dataset: {e}")
            # Don't raise - this shouldn't fail the task

        return {
            "recomendation_id": recommendation_hyp.recommendation_id,
            "summary": recommendation_hyp.summary,
        }

    def _create_recommendation_prompt(self, research_goal: str, previous_run: str, previous_conclusion: str) -> str:
        """Create a prompt for recommending the next run to perform."""
        #constraints = ', '.join(plan_config.get('constraints', []))
        #preferences = ', '.join(plan_config.get('preferences', []))

        base_prompt = (
            f"You are an AI co-scientist specializing in recommending the next run to perform based on previous conclusions.\n\n"
            f"Research goal:\n{research_goal}\n\n"
        )

        base_prompt += (
                f"This is a BINDER DESIGN task."
                f"Please base this recommendation on previous run + conclusions.\n\n"
                f"Previous run type: {previous_run}\n\n"
                f"Previous run conclusion: {previous_conclusion}\n\n"
                "Your recommendation must include:\n\n"
                f"1. **Next run suggestion:**\n"
                f"   - Choose either: bindcraft, simulation (in those exact words)\n"
                f"2. **Next run parameters:**\n"
                f"   - Should paramters: change, not change (in those exact words)\n"
                f"2. **Rationale for the run choice:**\n"
                f"   - Why should I make this run choice and change or not change params"
            )

        return base_prompt


    def _create_research_planning_prompt(self, research_goal: str, previous_run: str, recommendation: str) -> str:
        """Create a prompt for recommending the next run to perform."""
        #constraints = ', '.join(plan_config.get('constraints', []))
        #preferences = ', '.join(plan_config.get('preferences', []))

        base_prompt = (
            f"You are an AI co-scientist specializing in creating the next run config based on previous config and recommendation.\n\n"
            f"Research goal:\n{research_goal}\n\n"
        )

        new_task = recommendation[0]["metadata"]["next_task"]
        change_parameters = recommendation[0]["metadata"]["change_parameters"]
        if new_task == 'bindcraft':
            if change_parameters:
                old_parameters = previous_run["config"]
                previous_runtype = previous_run["run_type"]
                #json.dumps(recent, indent=2)
                base_prompt += (
                        f"Please base this research plan on previous run config + recommendations.\n\n"
                        f"Previous run type: {previous_runtype}\n\n"
                        f"Recommended next run: {new_task}\n\n"
                        f"Recommended to change the parameters. Old config: {json.dumps(old_parameters, indent=2)}"
                        f"Rationale for next run and changing parameters: {recommendation[0]['metadata']['rationale']}"
                        f"Your output must include:"
                        f"1. ** New config with same format as old config ** \n\n"
                        f"2. ** Rationale for this config ** \n\n"
                    )

        return base_prompt

    @staticmethod
    def _create_schema_from_config(config):
        """
        Recursively create a schema from a config dictionary,
        preserving the structure and inferring types.
        """
        schema = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                # Recursive case: nested dictionary
                schema[key] = self._create_schema_from_config(value)
            else:
                # Base case: infer type from value
                type_name = type(value).__name__
                
                # Map Python types to JSON-friendly type strings
                type_mapping = {
                    'str': 'string',
                    'int': 'integer',
                    'float': 'number',
                    'bool': 'boolean',
                    'list': 'array',
                    'dict': 'object',
                    'NoneType': 'null',
                }
                
                schema[key] = type_mapping.get(type_name, 'string')
        
        return schema
