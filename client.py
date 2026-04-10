"""Client wrapper for the Enterprise Code Review OpenEnv environment."""
from __future__ import annotations
from typing import Any
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from env import CodeReviewAction, CodeReviewObservation, EnterpriseCodeReviewState

class EnterpriseCodeReviewClient(EnvClient[CodeReviewAction, CodeReviewObservation, EnterpriseCodeReviewState]):
    """Typed client wrapper for a running Enterprise Code Review environment."""

    def _step_payload(self, action: CodeReviewAction) -> dict[str, Any]:
        """Serialize an action for transport to the server."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CodeReviewObservation]:
        """Parse a step/reset response into a typed result."""
        observation_payload = payload.get('observation', payload)
        observation = CodeReviewObservation.model_validate(observation_payload)
        return StepResult(observation=observation, reward=payload.get('reward', observation.reward), done=payload.get('done', observation.done))

    def _parse_state(self, payload: dict[str, Any]) -> EnterpriseCodeReviewState:
        """Parse the state payload into the typed state model."""
        return EnterpriseCodeReviewState.model_validate(payload)