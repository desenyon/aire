"""Docker Compose + Kubernetes deployment manifests for aire apps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aire.deployment.artifacts import generate_artifacts

# deploy.replicas is honored by `docker compose --compatibility` / swarm mode.
# For plain `docker compose up`, also set AIRE_SCALE_REPLICAS and use deploy.sh
# which passes `--scale aire=N`.
_COMPOSE = """\
services:
  aire:
    build: .
    image: {image}
    ports:
      - "{port}:8000"
    env_file:
      - .env
    environment:
      AIRE_PROJECT: {project}
      AIRE_MODEL__REF: ${{AIRE_MODEL__REF:-{model_ref}}}
      AIRE_SCALE_REPLICAS: "{replicas}"{extra_env}
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
    restart: unless-stopped
    # Swarm / compose --compatibility: deploy.replicas. Plain compose: use deploy.sh --scale.
    deploy:
      replicas: {replicas}
      resources:
        limits:
          cpus: "{cpu_limit}"
          memory: {memory_limit}
    {depends}
{extras}
"""

_K8S_DEPLOYMENT = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
  labels:
    app: {name}
    app.kubernetes.io/managed-by: aire
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      containers:
        - name: aire
          image: {image}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          envFrom:
            - secretRef:
                name: {name}-secrets
                optional: true
            - configMapRef:
                name: {name}-config
                optional: true
          env:
            - name: AIRE_PROJECT
              value: "{project}"
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 15
            periodSeconds: 20
          resources:
            requests:
              cpu: "{cpu_request}"
              memory: {memory_request}
            limits:
              cpu: "{cpu_limit}"
              memory: {memory_limit}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  selector:
    app: {name}
  ports:
    - port: 80
      targetPort: http
      name: http
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {name}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {name}
  minReplicas: {replicas}
  maxReplicas: {max_replicas}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {cpu_target}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}-config
data:
  AIRE_PROJECT: "{project}"
  AIRE_MODEL__REF: "{model_ref}"{config_extra}
"""

_REDIS_URL = "redis://redis:6379/0"
_DATABASE_URL = "postgresql://aire:aire@postgres:5432/aire"


class ScaleConfig(BaseModel):
    """Sizing knobs for compose / kubernetes manifests."""

    project: str = "aire-app"
    name: str = "aire"
    image: str = "aire-app:latest"
    port: int = 8000
    replicas: int = 2
    max_replicas: int = 10
    cpu_request: str = "100m"
    cpu_limit: str = "1"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    cpu_target: int = 70
    model_ref: str = "mock:echo"
    with_redis: bool = False
    with_postgres: bool = False


class ScaleArtifacts(BaseModel):
    directory: str
    files: list[str] = Field(default_factory=list)
    config: ScaleConfig = Field(default_factory=ScaleConfig)


def _sidecar_env_lines(cfg: ScaleConfig) -> str:
    """YAML environment entries for compose (leading newline + indent)."""
    lines: list[str] = []
    if cfg.with_redis:
        lines.append(f"      AIRE_REDIS_URL: {_REDIS_URL}")
    if cfg.with_postgres:
        lines.append(f"      AIRE_DATABASE_URL: {_DATABASE_URL}")
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def _configmap_extra(cfg: ScaleConfig) -> str:
    lines: list[str] = []
    if cfg.with_redis:
        lines.append(f'  AIRE_REDIS_URL: "{_REDIS_URL}"')
    if cfg.with_postgres:
        lines.append(f'  AIRE_DATABASE_URL: "{_DATABASE_URL}"')
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def generate_compose(directory: str | Path, config: ScaleConfig | None = None) -> Path:
    """Write ``docker-compose.yml`` for an aire service (+ optional redis/postgres)."""
    cfg = config or ScaleConfig()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    depends = ""
    extras = ""
    if cfg.with_redis:
        # Sidecar: env AIRE_REDIS_URL points the app scaffold at redis:6379.
        depends = "depends_on:\n      - redis"
        extras += """
  # Sidecar: AIRE_REDIS_URL is injected into the aire service.
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
"""
    if cfg.with_postgres:
        # Sidecar: env AIRE_DATABASE_URL points the app scaffold at postgres:5432.
        if depends:
            depends = "depends_on:\n      - redis\n      - postgres"
        else:
            depends = "depends_on:\n      - postgres"
        extras += """
  # Sidecar: AIRE_DATABASE_URL is injected into the aire service.
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: aire
      POSTGRES_USER: aire
      POSTGRES_DB: aire
    ports:
      - "5432:5432"
    restart: unless-stopped
"""
    content = _COMPOSE.format(
        image=cfg.image,
        port=cfg.port,
        project=cfg.project,
        replicas=cfg.replicas,
        cpu_limit=cfg.cpu_limit,
        memory_limit=cfg.memory_limit,
        model_ref=cfg.model_ref,
        depends=depends,
        extras=extras,
        extra_env=_sidecar_env_lines(cfg),
    )
    path = target / "docker-compose.yml"
    path.write_text(content)
    return path


def generate_k8s(directory: str | Path, config: ScaleConfig | None = None) -> Path:
    """Write a combined Kubernetes manifest (Deployment + Service + HPA + ConfigMap)."""
    cfg = config or ScaleConfig()
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    content = _K8S_DEPLOYMENT.format(
        name=cfg.name,
        image=cfg.image,
        project=cfg.project,
        replicas=cfg.replicas,
        max_replicas=cfg.max_replicas,
        cpu_request=cfg.cpu_request,
        cpu_limit=cfg.cpu_limit,
        memory_request=cfg.memory_request,
        memory_limit=cfg.memory_limit,
        cpu_target=cfg.cpu_target,
        model_ref=cfg.model_ref,
        config_extra=_configmap_extra(cfg),
    )
    path = target / "k8s.yaml"
    path.write_text(content)
    return path


def generate_scale_pack(
    directory: str | Path,
    *,
    config: ScaleConfig | None = None,
    include_base_artifacts: bool = True,
) -> ScaleArtifacts:
    """Generate Dockerfile pack + compose + k8s in one call."""
    cfg = config or ScaleConfig()
    target = Path(directory)
    files: list[str] = []
    if include_base_artifacts:
        extras: list[str] = []
        if cfg.with_redis:
            extras.append("aire[redis]>=0.1.0")
        if cfg.with_postgres:
            extras.append("aire[pgvector]>=0.1.0")
        base = generate_artifacts(target, project=cfg.project, extra_requirements=extras or None)
        files.extend(base.files)
    files.append(str(generate_compose(target, cfg)))
    files.append(str(generate_k8s(target, cfg)))
    # Helper scripts — scale via compose CLI for non-swarm hosts
    deploy_sh = target / "deploy.sh"
    deploy_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'REPLICAS="${{AIRE_SCALE_REPLICAS:-{cfg.replicas}}}"\n'
        'echo "Building and deploying aire stack (replicas=$REPLICAS)..."\n'
        'docker compose up -d --build --scale aire="$REPLICAS"\n'
        'echo "For Kubernetes: kubectl apply -f k8s.yaml"\n'
    )
    deploy_sh.chmod(0o755)
    files.append(str(deploy_sh))
    return ScaleArtifacts(directory=str(target), files=files, config=cfg)


def describe() -> dict[str, Any]:
    return {
        "kind": "scale",
        "artifacts": ["docker-compose.yml", "k8s.yaml", "deploy.sh"],
        "features": ["replicas", "HPA", "redis", "pgvector", "health/readiness probes"],
        "env": {
            "redis": "AIRE_REDIS_URL=redis://redis:6379/0",
            "postgres": "AIRE_DATABASE_URL=postgresql://aire:aire@postgres:5432/aire",
        },
        "notes": [
            "with_redis/with_postgres inject AIRE_*_URL into compose + ConfigMap",
            "app.py scaffold wires Redis model cache and PgVectorStore Knowledge when set",
            "compose scale via deploy.sh --scale or AIRE_SCALE_REPLICAS",
        ],
        "factory": "aire.deployment.scale.generate_scale_pack",
    }
