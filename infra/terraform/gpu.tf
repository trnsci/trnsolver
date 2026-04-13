# ---------------------------------------------------------------------------
# GPU CI instance — vintage-matched Nvidia peer for the Trainium chip.
#
# Default: g5.xlarge (A10G, Ampere, 2021 launch) — peer of trn1 (2022).
# Opt in with `-var=enable_gpu_ci=true` on `terraform apply`. Disabled by
# default so a default apply doesn't accrue GPU costs.
# ---------------------------------------------------------------------------

variable "enable_gpu_ci" {
  description = "Provision the GPU CI instance (g5.xlarge A10G)"
  type        = bool
  default     = false
}

variable "gpu_instance_type" {
  description = "EC2 instance type for the GPU CI box"
  type        = string
  default     = "g5.xlarge"
  # Alternatives:
  #   g5.xlarge       — A10G, ~$1.006/hr, vintage peer of trn1
  #   p5.48xlarge     — H100 x8, ~$98/hr, vintage peer of trn2
  #   p4d.24xlarge    — A100 x8, ~$32/hr, peer of trn1 on a higher tier
}

variable "gpu_instance_tag" {
  description = "Tag used by scripts/run_cuda_tests.sh to locate the instance"
  type        = string
  default     = "trnsolver-ci-g5"
}

# Deep Learning OSS Nvidia Driver AMI (Ubuntu 22.04) — PyTorch preinstalled.
data "aws_ami" "cuda" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.*Ubuntu 22.04*"]
  }
}

resource "aws_instance" "gpu" {
  count = var.enable_gpu_ci ? 1 : 0

  ami                         = data.aws_ami.cuda.id
  instance_type               = var.gpu_instance_type
  subnet_id                   = var.subnet_id
  iam_instance_profile        = aws_iam_instance_profile.instance.name
  vpc_security_group_ids      = [aws_security_group.instance.id]
  associate_public_ip_address = true

  root_block_device {
    volume_size = 100
    volume_type = "gp3"
  }

  user_data = <<-EOF
    #!/bin/bash
    set -euxo pipefail
    cd /home/ubuntu
    sudo -u ubuntu git clone https://github.com/trnsci/trnsolver.git trnsolver
    # DLAMI ships a PyTorch venv at /opt/pytorch. Install dev extras into it.
    PY_VENV=$(ls -d /opt/pytorch* 2>/dev/null | head -1 || true)
    if [ -z "$PY_VENV" ]; then
      PY_VENV=/home/ubuntu/venv
      sudo -u ubuntu python3 -m venv "$PY_VENV"
    fi
    sudo -u ubuntu "$PY_VENV/bin/pip" install -e '/home/ubuntu/trnsolver[dev]'
  EOF

  tags = {
    Name = var.gpu_instance_tag
  }
}

output "gpu_instance_id" {
  value       = try(aws_instance.gpu[0].id, "")
  description = "GPU CI instance ID (empty if enable_gpu_ci=false)"
}

output "gpu_instance_tag" {
  value       = var.gpu_instance_tag
  description = "Tag used by scripts/run_cuda_tests.sh to find the instance"
}
