#!/usr/bin/env bash
#
# Run neuron-marked pytest tests on the trnsolver CI instance.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_neuron_tests.sh [instance_type]
#
# Default instance_type is trn1 (looks for Name=trnsolver-ci-trn1).
# Provision the instance with:
#   cd infra/terraform && terraform apply -var=vpc_id=... -var=subnet_id=...
#
# This script:
#   1. Starts the tagged instance (if stopped)
#   2. Waits for SSM agent
#   3. Runs `pytest tests/ -v -m neuron` via SSM send-command
#   4. Prints stdout/stderr
#   5. Stops the instance (always, even on failure)

set -euo pipefail

WARM=0
if [[ "${1:-}" == "--warm" ]]; then
  WARM=1
  shift
fi

INSTANCE_TYPE="${1:-trn1}"
TAG="trnsolver-ci-${INSTANCE_TYPE}"
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
  echo "ERROR: No instance found with Name=$TAG" >&2
  echo "Provision with: cd infra/terraform && terraform apply" >&2
  exit 1
fi

echo "Instance: $INSTANCE_ID"

# Auto-stop on script exit is optional. When the script is invoked under a
# timeout-bound runner (background task, CI wrapper, etc.), the trap firing
# mid-compile would kill an in-flight NKI run. Set AUTO_STOP=1 to opt in;
# otherwise the operator is responsible for running scripts/stop_neuron_ci.sh.
AUTO_STOP="${AUTO_STOP:-0}"

cleanup() {
  local exit_code=$?
  echo ""
  if [[ "$AUTO_STOP" == "1" ]]; then
    echo "Stopping $INSTANCE_ID (AUTO_STOP=1)..."
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
  else
    echo "Leaving $INSTANCE_ID running. Stop with: AWS_PROFILE=$AWS_PROFILE ./scripts/stop_neuron_ci.sh $INSTANCE_TYPE"
  fi
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
# `aws ssm wait instance-information` isn't available in all CLI versions —
# poll describe-instance-information instead.
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

# --warm: run the suite twice to expose the NEFF cache delta — the second
# pass gets warm /var/tmp/neuron-compile-cache/.
if [[ "$WARM" == "1" ]]; then
  PYTEST_INVOCATION="pytest tests/ -v -s -m neuron --tb=short && echo === WARM PASS === && pytest tests/ -v -s -m neuron --tb=short"
else
  PYTEST_INVOCATION="pytest tests/ -v -m neuron --tb=short"
fi

# Activate the Neuron venv inside the user shell so libneuronpjrt-path
# (needed by torch_neuronx PJRT init) is on PATH. sudo'ing to ubuntu
# without activation doesn't propagate the venv PATH.
REQ_NKI="${TRNSOLVER_REQUIRE_NKI:-1}"
# --optlevel=1 drops the Neuron compiler to fast codegen (less runtime
# optimization, much faster compile). Appropriate for Phase 1 iteration;
# Phase 3 perf work should drop the flag or raise to --optlevel=2.
NEURON_CC_FLAGS_DEFAULT="--optlevel=1"
NEURON_CC_FLAGS_SETTING="${NEURON_CC_FLAGS:-$NEURON_CC_FLAGS_DEFAULT}"
TEST_SCRIPT="source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate && \
  cd /home/ubuntu/trnsolver && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  export NEURON_CC_FLAGS='$NEURON_CC_FLAGS_SETTING' && \
  TRNSOLVER_REQUIRE_NKI=$REQ_NKI $PYTEST_INVOCATION"

echo "Sending test command (SHA=$SHA, warm=$WARM, require_nki=$REQ_NKI, cc_flags=$NEURON_CC_FLAGS_SETTING)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnsolver neuron tests @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$TEST_SCRIPT\\\"\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete (this may take several minutes)..."

# `aws ssm wait command-executed` has a short built-in timeout that fires
# before NKI compilation finishes. Poll manually — match trnfft's pattern.
STATUS="InProgress"
for i in $(seq 1 120); do
  STATUS=$(aws ssm get-command-invocation \
    --command-id "$CMD_ID" \
    --instance-id "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Status' --output text 2>/dev/null || echo "Unknown")
  case "$STATUS" in
    Success|Failed|TimedOut|Cancelled) break ;;
  esac
  sleep 15
done

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
