#!/usr/bin/env bash
#
# Run CUDA / cuSOLVER benchmarks on the trnsolver GPU CI instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_cuda_tests.sh [instance_suffix]
#
# Default instance_suffix is g5 (looks for Name=trnsolver-ci-g5).
# Provision the instance with:
#   cd infra/terraform && terraform apply -var=enable_gpu_ci=true \
#     -var=vpc_id=... -var=subnet_id=...
#
# This script:
#   1. Starts the tagged GPU instance (if stopped)
#   2. Waits for SSM agent
#   3. Runs `pytest benchmarks/bench_cuda.py -v -m cuda --benchmark-only`
#   4. Prints benchmark output
#   5. Stops the instance (always, even on failure)

set -euo pipefail

INSTANCE_SUFFIX="${1:-g5}"
TAG="trnsolver-ci-${INSTANCE_SUFFIX}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

if [[ -z "${AWS_PROFILE:-}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
  echo "ERROR: set AWS_PROFILE (local) or AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY (CI/OIDC)" >&2
  exit 1
fi

echo "Looking up instance with Name=$TAG in $REGION..."
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=$TAG" \
            "Name=instance-state-name,Values=stopped,running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text \
  --region "$REGION")

if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "None" ]]; then
  echo "ERROR: No GPU instance found with Name=$TAG" >&2
  echo "Provision with: cd infra/terraform && terraform apply -var=enable_gpu_ci=true" >&2
  exit 1
fi

echo "Instance: $INSTANCE_ID"

cleanup() {
  local exit_code=$?
  echo ""
  echo "Stopping $INSTANCE_ID..."
  aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  exit "$exit_code"
}
trap cleanup EXIT

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query 'Reservations[0].Instances[0].State.Name' --output text)

if [[ "$STATE" == "stopped" ]]; then
  echo "Starting instance..."
  aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
fi

echo "Waiting for instance-running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

echo "Waiting for SSM agent..."
for _ in $(seq 1 60); do
  PING=$(aws ssm describe-instance-information \
    --filters "Key=InstanceIds,Values=$INSTANCE_ID" \
    --region "$REGION" \
    --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)
  [[ "$PING" == "Online" ]] && break
  sleep 5
done
if [[ "$PING" != "Online" ]]; then
  echo "ERROR: SSM agent not Online after 5 minutes (last PingStatus=$PING)" >&2
  exit 1
fi

echo "Sending benchmark command (SHA=$SHA)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnsolver cuda benchmarks @ $SHA" \
  --parameters "commands=[
    \"bash -c 'set -euo pipefail; cd /home/ubuntu/trnsolver && sudo -u ubuntu git fetch --all && sudo -u ubuntu git checkout $SHA && PY_VENV=\$(ls -d /opt/pytorch* 2>/dev/null | head -1 || echo /home/ubuntu/venv) && sudo -u ubuntu \$PY_VENV/bin/pip install -e /home/ubuntu/trnsolver[dev] --quiet && sudo -u ubuntu \$PY_VENV/bin/pytest /home/ubuntu/trnsolver/benchmarks/bench_cuda.py -v -m cuda --benchmark-only --benchmark-columns=mean,stddev,rounds'\"
  ]" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (this may take several minutes)..."

aws ssm wait command-executed \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" || true

STATUS=$(aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'Status' --output text)

echo ""
echo "=== STDOUT ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardOutputContent' --output text

echo ""
echo "=== STDERR ==="
aws ssm get-command-invocation \
  --command-id "$CMD_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query 'StandardErrorContent' --output text

echo ""
echo "=== Status: $STATUS ==="

[[ "$STATUS" == "Success" ]]
