#!/usr/bin/env bash
#
# Stop the trnsolver neuron CI instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/stop_neuron_ci.sh [instance_type]
#
# Default instance_type is trn1 (looks for Name=trnsolver-ci-trn1). Also
# handles g5 (CUDA baseline) and any other trnsolver-ci-$SUFFIX tag.
#
# Use this to halt the hourly burn after running tests with AUTO_STOP=0
# (the default). With AUTO_STOP=1, run_neuron_tests.sh stops the instance
# itself on script exit.

set -euo pipefail

INSTANCE_TYPE="${1:-trn1}"
TAG="trnsolver-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"

if [[ -z "${AWS_PROFILE:-}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
  echo "ERROR: set AWS_PROFILE or AWS_ACCESS_KEY_ID" >&2
  exit 1
fi

INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=running,pending,stopping" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "No running instance found with Name=$TAG in $REGION. Already stopped?"
  exit 0
fi

echo "Stopping $INSTANCE_ID ($TAG)..."
aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'StoppingInstances[].[InstanceId,CurrentState.Name]' --output text
