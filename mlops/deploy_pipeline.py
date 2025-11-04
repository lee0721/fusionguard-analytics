#!/usr/bin/env python3
"""Utility script to build, push, and deploy the FusionGuard inference service."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: List[str], *, dry_run: bool = False, env: dict | None = None) -> None:
    printable = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return

    print(f"→ {printable}")
    try:
        subprocess.run(cmd, check=True, env=env)
    except FileNotFoundError as exc:
        raise SystemExit(f"Command not found: {cmd[0]}. Please install it and retry.") from exc


def build_image(tag: str, context: Path, *, platform: str, dry_run: bool) -> None:
    run(
        [
            "docker",
            "build",
            "--platform",
            platform,
            "-t",
            tag,
            str(context),
        ],
        dry_run=dry_run,
    )


def push_image(tag: str, *, dry_run: bool) -> None:
    run(["docker", "push", tag], dry_run=dry_run)


def deploy_to_cloud_run(
    *,
    image: str,
    service: str,
    region: str,
    project: str,
    allow_unauthenticated: bool,
    min_instances: int,
    max_instances: int,
    cpu: str,
    memory: str,
    dry_run: bool,
) -> None:
    cmd = [
        "gcloud",
        "run",
        "deploy",
        service,
        "--image",
        image,
        "--region",
        region,
        "--project",
        project,
        "--platform",
        "managed",
        "--min-instances",
        str(min_instances),
        "--max-instances",
        str(max_instances),
        "--cpu",
        cpu,
        "--memory",
        memory,
        "--port",
        "8000",
    ]
    if allow_unauthenticated:
        cmd.append("--allow-unauthenticated")

    run(cmd, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FusionGuard deployment automation.")
    parser.add_argument("--image", required=True, help="Container image tag, e.g., ghcr.io/org/fusionguard:latest")
    parser.add_argument("--context", type=Path, default=PROJECT_ROOT, help="Docker build context (default: repo root).")
    parser.add_argument("--platform", default="linux/amd64", help="Docker build target platform.")
    parser.add_argument("--service", default="fusionguard-agent", help="Cloud Run service name.")
    parser.add_argument("--region", default="us-central1", help="Cloud Run region.")
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument("--push", action="store_true", help="Push image after build.")
    parser.add_argument("--deploy", action="store_true", help="Deploy to Cloud Run after build/push.")
    parser.add_argument("--allow-unauthenticated", action="store_true", help="Expose the Cloud Run service publicly.")
    parser.add_argument("--min-instances", type=int, default=0, help="Minimum Cloud Run instances.")
    parser.add_argument("--max-instances", type=int, default=1, help="Maximum Cloud Run instances.")
    parser.add_argument("--cpu", default="1", help="CPU allocation for Cloud Run.")
    parser.add_argument("--memory", default="1Gi", help="Memory allocation for Cloud Run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    build_image(args.image, args.context, platform=args.platform, dry_run=args.dry_run)

    if args.push:
        push_image(args.image, dry_run=args.dry_run)

    if args.deploy:
        if not args.push and not args.dry_run:
            print("⚠️ Deploying without pushing the image; ensure the registry already has the tag.")
        deploy_to_cloud_run(
            image=args.image,
            service=args.service,
            region=args.region,
            project=args.project,
            allow_unauthenticated=args.allow_unauthenticated,
            min_instances=args.min_instances,
            max_instances=args.max_instances,
            cpu=args.cpu,
            memory=args.memory,
            dry_run=args.dry_run,
        )

    print("✅ Deployment pipeline finished.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
