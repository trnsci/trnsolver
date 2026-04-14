#!/usr/bin/env bash
#
# Run NKI simulator-backed tests on the trnsolver CI instance.
#
# The simulator bypasses torch_xla + NEFF compile: kernels run on CPU via
# `nki.simulate(kernel)(numpy_args)`. Use for correctness / constraint
# iteration; hardware (run_neuron_tests.sh) still owns perf numbers.
#
# Usage:
#   AWS_PROFILE=aws ./scripts/run_simulator_tests.sh
#
# Same trap-stop pattern as run_neuron_tests.sh. Runs the nki_simulator-
# marked test suite with TRNSOLVER_USE_SIMULATOR=1 set in the SSM env.
#
# Still AWS-resident here even though the simulator is CPU-only: the `nki`
# wheel is linux_x86_64 and lives on the Neuron pip index. The CI job
# `nki-simulator` in .github/workflows/ci.yml does the same work on
# ubuntu-latest runners at zero AWS cost — prefer that for PR-gating.

set -euo pipefail

INSTANCE_TYPE="${INSTANCE_TYPE:-trn1}"
TAG="trnsolver-ci-${INSTANCE_TYPE}"
REGION="${AWS_REGION:-us-east-1}"
SHA="$(git rev-parse HEAD)"

if [[ -z "${AWS_PROFILE:-}" && -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
  echo "ERROR: set AWS_PROFILE (local) or AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY" >&2
  exit 1
fi

AUTO_STOP="${AUTO_STOP:-0}"

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

# Activate the Neuron venv so the bundled `nki` import is on PATH. Then set
# TRNSOLVER_USE_SIMULATOR=1 so dispatch routes through `nki.simulate` on
# the instance's CPU rather than going through torch_xla to the NeuronCore.
TEST_SCRIPT="NEURON_VENV=\\\$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | sort -V | tail -1) && \
  source \\\$NEURON_VENV/bin/activate && \
  cd /home/ubuntu/trnsolver && \
  git fetch --all && \
  git checkout $SHA && \
  pip install -e '.[dev]' --quiet && \
  TRNSOLVER_USE_SIMULATOR=1 pytest tests/ -v -m nki_simulator --tb=short"

echo "Sending simulator test command (SHA=$SHA)..."
CMD_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --comment "trnsolver nki simulator tests @ $SHA" \
  --parameters "{\"commands\":[\"sudo -u ubuntu bash -c \\\"$TEST_SCRIPT\\\"\"]}" \
  --region "$REGION" \
  --output text --query 'Command.CommandId')

echo "Command ID: $CMD_ID"
echo "Waiting for command to complete..."

STATUS="InProgress"
for _ in $(seq 1 120); do
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
  --query 'StandardErrorContent' --output text | tail -30

echo ""
echo "=== Status: $STATUS ==="
[[ "$STATUS" == "Success" ]]
