"""
Jnana-ProtoGnosis Adapter.

This module provides the main adapter class for integrating ProtoGnosis
functionality into the Jnana system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..core.coscientist import CoScientist
from ..core.multi_llm_config import LLMConfig, AgentLLMConfig
from .data_converter import ProtoGnosisDataConverter
from ...data.unified_hypothesis import UnifiedHypothesis
from ...core.model_manager import UnifiedModelManager


class JnanaProtoGnosisAdapter:
    """
    Adapter class for integrating ProtoGnosis with Jnana.
    
    This class provides a bridge between the Jnana system and ProtoGnosis,
    handling data conversion, configuration mapping, and workflow coordination.
    """
    
    def __init__(self, model_manager: UnifiedModelManager, 
                 storage_path: Optional[str] = None,
                 max_workers: int = 4):
        """
        Initialize the Jnana-ProtoGnosis adapter.
        
        Args:
            model_manager: Jnana unified model manager
            storage_path: Path for ProtoGnosis storage
            max_workers: Maximum number of worker threads
        """
        self.model_manager = model_manager
        self.storage_path = storage_path
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # ProtoGnosis instance
        self.coscientist: Optional[CoScientist] = None
        self.is_initialized = False
        
        # Data converter
        self.converter = ProtoGnosisDataConverter()
    
    async def initialize(self) -> bool:
        """
        Initialize the ProtoGnosis system with Jnana configuration.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing ProtoGnosis with Jnana configuration...")
            
            # Convert Jnana model configuration to ProtoGnosis format
            protognosis_config = self._convert_model_config()
            
            # Initialize CoScientist
            self.coscientist = CoScientist(
                llm_config=protognosis_config,
                storage_path=self.storage_path,
                max_workers=self.max_workers
            )

            # Start the CoScientist system (this starts the worker threads!)
            self.coscientist.start()

            self.is_initialized = True
            self.logger.info("ProtoGnosis initialized and started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ProtoGnosis: {e}")
            self.is_initialized = False
            return False

    async def shutdown(self) -> None:
        """
        Shutdown the ProtoGnosis system and clean up resources.
        """
        if self.coscientist and self.is_initialized:
            try:
                self.logger.info("Shutting down ProtoGnosis system...")
                self.coscientist.stop()
                self.is_initialized = False
                self.logger.info("ProtoGnosis system stopped successfully")
            except Exception as e:
                self.logger.error(f"Error shutting down ProtoGnosis: {e}")
    
    async def generate_hypotheses(self, research_goal: str, count: int = 5,
                                 strategies: Optional[List[str]] = None) -> List[UnifiedHypothesis]:
        """
        Generate hypotheses using ProtoGnosis and convert to Jnana format.
        
        Args:
            research_goal: The research goal/question
            count: Number of hypotheses to generate
            strategies: Generation strategies to use
            
        Returns:
            List of UnifiedHypothesis objects
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.coscientist:
            raise RuntimeError("ProtoGnosis not initialized")
        
        try:
            self.logger.info(f"Generating {count} hypotheses for: {research_goal[:100]}...")
            
            # Set research goal in ProtoGnosis
            self.coscientist.set_research_goal(research_goal)
            
            # Generate hypotheses
            hypothesis_ids = self.coscientist.generate_hypotheses(
                count=count,
                strategies=strategies or ["literature_exploration", "scientific_debate"]
            )
            
            # Wait for completion
            self.coscientist.wait_for_completion()
            
            # Get generated hypotheses
            pg_hypotheses = self.coscientist.get_all_hypotheses()
            
            # Convert to Jnana format
            unified_hypotheses = self.converter.batch_protognosis_to_unified(pg_hypotheses)
            
            self.logger.info(f"Successfully generated and converted {len(unified_hypotheses)} hypotheses")
            return unified_hypotheses
            
        except Exception as e:
            self.logger.error(f"Error generating hypotheses: {e}")
            return []
    
    async def generate_recommendations(self, research_goal: str, count: int = 5,
                                 results: dict = None) -> List[UnifiedHypothesis]:
        """
        Generate recommendations using ProtoGnosis and convert to Jnana format.
        
        Args:
            research_goal: The research goal/question
            count: Number of hypotheses to generate
            results: Results from previous run 
            
        Returns:
            List of UnifiedHypothesis objects
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.coscientist:
            raise RuntimeError("ProtoGnosis not initialized")
        
        try:
            self.logger.info(f"Generating {count} hypotheses for: {research_goal[:100]}...")
            
            # Set research goal in ProtoGnosis
            self.coscientist.set_research_goal(research_goal)
            
            # Generate hypotheses
            recommendation_ids = self.coscientist.recommend_next_run(
                count=count,
                results=results
            )
            
            # Wait for completion
            self.coscientist.wait_for_completion()
            
            # Get generated hypotheses
            pg_recommendations = self.coscientist.get_all_recommendations()
            self.logger.info(f"Generated recommendations: {pg_recommendations}")
            # Convert to Jnana format
            #unified_recommendations = self.converter.batch_protognosis_to_unified(pg_recommendations)
            
            self.logger.info(f"Successfully generated and converted {len(pg_recommendations)} recommendations")
            return pg_recommendations #unified_recommendations 
            
        except Exception as e:
            self.logger.error(f"Error generating hypotheses: {e}")
            return []

    async def generate_config_recommendations(self, research_goal: str, count: int = 5,
                                 previous_run: dict = None, recommendation: object=None) -> List[UnifiedHypothesis]:
        """
        Generate recommendations using ProtoGnosis and convert to Jnana format.
        
        Args:
            research_goal: The research goal/question
            count: Number of hypotheses to generate
            previous_run: a dictionary including both previous run_type and old config
            recommendation: an object created via the recommend_run task
        Returns:
            List of UnifiedHypothesis objects
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.coscientist:
            raise RuntimeError("ProtoGnosis not initialized")
        
        try:
            self.logger.info(f"Generating {count} recommended configs for: {research_goal[:100]}...")
            
            # Set research goal in ProtoGnosis
            if not hasattr(self.coscientist, 'research_goal'):
                self.coscientist.set_research_goal(research_goal)
            
            # Generate hypotheses
            recommendation_ids = self.coscientist.recommend_next_config(
                count=count,
                previous_run = previous_run,
                recommendation=recommendation
            )
            
            # Wait for completion
            self.coscientist.wait_for_completion()
            
            # Get generated hypotheses
            pg_recommend_config = self.coscientist.get_all_recommendations()
            self.logger.info(f"Generated recommendations: {pg_recommendations}")
            # Convert to Jnana format
            #unified_recommendations = self.converter.batch_protognosis_to_unified(pg_recommendations)
            
            self.logger.info(f"Successfully generated and converted {len(pg_recommendations)} recommendations")
            return pg_recommendations #unified_recommendations 
            
        except Exception as e:
            self.logger.error(f"Error generating hypotheses: {e}")
            return []


    async def run_tournament(self, hypotheses: List[UnifiedHypothesis], 
                           match_count: int = 25) -> List[UnifiedHypothesis]:
        """
        Run tournament evaluation on hypotheses using ProtoGnosis.
        
        Args:
            hypotheses: List of hypotheses to evaluate
            match_count: Number of tournament matches
            
        Returns:
            List of hypotheses with updated tournament records
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.coscientist:
            raise RuntimeError("ProtoGnosis not initialized")
        
        try:
            self.logger.info(f"Running tournament with {match_count} matches on {len(hypotheses)} hypotheses")
            
            # Convert hypotheses to ProtoGnosis format
            pg_hypotheses = self.converter.batch_unified_to_protognosis(hypotheses)
            
            # Add hypotheses to ProtoGnosis memory
            for pg_hyp in pg_hypotheses:
                self.coscientist.memory.add_hypothesis(pg_hyp)
            
            # Run tournament
            self.coscientist.run_tournament(match_count=match_count)
            self.coscientist.wait_for_completion()
            
            # Get updated hypotheses
            updated_pg_hypotheses = self.coscientist.get_all_hypotheses()
            
            # Convert back to Jnana format
            updated_unified_hypotheses = self.converter.batch_protognosis_to_unified(updated_pg_hypotheses)
            
            self.logger.info(f"Tournament completed successfully")
            return updated_unified_hypotheses
            
        except Exception as e:
            self.logger.error(f"Error running tournament: {e}")
            return hypotheses  # Return original hypotheses if tournament fails
    
    async def evolve_hypothesis(self, hypothesis: UnifiedHypothesis, 
                               feedback: str = "") -> UnifiedHypothesis:
        """
        Evolve a hypothesis using ProtoGnosis evolution agent.
        
        Args:
            hypothesis: Hypothesis to evolve
            feedback: Feedback for evolution
            
        Returns:
            Evolved hypothesis
        """
        if not self.is_initialized:
            await self.initialize()
        
        if not self.coscientist:
            raise RuntimeError("ProtoGnosis not initialized")
        
        try:
            # Convert to ProtoGnosis format
            pg_hypothesis = self.converter.unified_to_protognosis(hypothesis)
            
            # Add to ProtoGnosis memory
            self.coscientist.memory.add_hypothesis(pg_hypothesis)
            
            # Evolve hypothesis
            evolution_result = self.coscientist.evolve_hypothesis(
                pg_hypothesis.hypothesis_id, feedback
            )

            # Get the evolved hypothesis from memory
            evolved_pg_hypothesis = self.coscientist.memory.get_hypothesis(
                evolution_result.get("evolved_hypothesis_id", pg_hypothesis.hypothesis_id)
            ) or pg_hypothesis
            
            # Convert back to Jnana format
            evolved_unified_hypothesis = self.converter.protognosis_to_unified(evolved_pg_hypothesis)
            
            return evolved_unified_hypothesis
            
        except Exception as e:
            self.logger.error(f"Error evolving hypothesis: {e}")
            return hypothesis  # Return original if evolution fails
    
    def _convert_model_config(self) -> AgentLLMConfig:
        """
        Convert Jnana model configuration to ProtoGnosis format.

        Returns:
            AgentLLMConfig for ProtoGnosis
        """
        try:
            # Get the raw config dictionary from UnifiedModelManager
            # We can't use model_manager.agent_llm_config directly because it might
            # be created with a different AgentLLMConfig class (import issue)
            default_config = self.model_manager.config.get("default", {})
            agents_config = self.model_manager.config.get("agents", {})

            # Create default LLM config using ProtoGnosis classes
            default_llm_config = LLMConfig(
                provider=default_config.get("provider", "openai"),
                model=default_config.get("model", "gpt-4o"),
                api_key=default_config.get("api_key"),
                base_url=default_config.get("base_url"),
                temperature=default_config.get("temperature", 0.7),
                max_tokens=default_config.get("max_tokens", 2048),
                model_adapter=default_config.get("model_adapter")
            )

            # Helper function to create LLMConfig from dict
            def create_llm_config(config_dict: Dict[str, Any]) -> LLMConfig:
                return LLMConfig(
                    provider=config_dict.get("provider", default_llm_config.provider),
                    model=config_dict.get("model", default_llm_config.model),
                    api_key=config_dict.get("api_key", default_llm_config.api_key),
                    base_url=config_dict.get("base_url", default_llm_config.base_url),
                    temperature=config_dict.get("temperature", default_llm_config.temperature),
                    max_tokens=config_dict.get("max_tokens", default_llm_config.max_tokens),
                    model_adapter=config_dict.get("model_adapter", default_llm_config.model_adapter)
                )

            # Create agent-specific configs
            agent_configs = {}
            for agent_type in ["supervisor", "generation", "reflection", "ranking", "evolution", "proximity", "meta_review"]:
                if agent_type in agents_config:
                    agent_configs[agent_type] = create_llm_config(agents_config[agent_type])

            # Create AgentLLMConfig using ProtoGnosis class
            return AgentLLMConfig(
                default=default_llm_config,
                **agent_configs
            )

        except Exception as e:
            self.logger.warning(f"Error converting model config, using defaults: {e}")
            import traceback
            traceback.print_exc()
            # Return basic default configuration
            return AgentLLMConfig(
                default=LLMConfig(provider="openai", model="gpt-4o")
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the ProtoGnosis integration.
        
        Returns:
            Dictionary containing status information
        """
        return {
            "initialized": self.is_initialized,
            "coscientist_available": self.coscientist is not None,
            "storage_path": self.storage_path,
            "max_workers": self.max_workers,
            "timestamp": datetime.now().isoformat()
        }
