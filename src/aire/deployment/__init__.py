"""Deployment: serve agents, knowledge pipelines and models as production APIs."""

from aire.deployment.artifacts import DeployArtifacts, generate_artifacts
from aire.deployment.fastapi_app import create_app

__all__ = ["DeployArtifacts", "create_app", "generate_artifacts"]
