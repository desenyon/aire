"""Deployment: serve agents, knowledge pipelines and models as production APIs."""

from aire.deployment.artifacts import DeployArtifacts, generate_artifacts
from aire.deployment.fastapi_app import create_app
from aire.deployment.gateway import Gateway, create_gateway
from aire.deployment.scale import (
    ScaleArtifacts,
    ScaleConfig,
    generate_compose,
    generate_k8s,
    generate_scale_pack,
)

__all__ = [
    "DeployArtifacts",
    "Gateway",
    "ScaleArtifacts",
    "ScaleConfig",
    "create_app",
    "create_gateway",
    "generate_artifacts",
    "generate_compose",
    "generate_k8s",
    "generate_scale_pack",
]
