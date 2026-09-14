data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "batch" {
  name_prefix = "${var.name}-batch-"
  description = "Outbound-only networking for RigFL AWS Batch workers"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/batch/${var.name}"
  retention_in_days = 14
}

resource "aws_batch_compute_environment" "gpu" {
  name         = "${var.name}-gpu-spot"
  type         = "MANAGED"
  state        = "ENABLED"
  service_role = aws_iam_service_linked_role.batch.arn

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    bid_percentage      = 100

    min_vcpus     = 0
    desired_vcpus = 0
    max_vcpus     = var.max_vcpus

    instance_role       = aws_iam_instance_profile.batch_instance.arn
    instance_type       = var.gpu_instance_types
    security_group_ids  = [aws_security_group.batch.id]
    spot_iam_fleet_role = aws_iam_role.spot_fleet.arn
    subnets             = data.aws_subnets.default.ids

    ec2_configuration {
      image_type = "ECS_AL2023_NVIDIA"
    }
  }

  depends_on = [
    aws_iam_role_policy_attachment.batch_instance,
    aws_iam_role_policy_attachment.spot_fleet,
  ]
}

resource "aws_batch_job_queue" "gpu" {
  name     = "${var.name}-gpu"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.gpu.arn
  }
}

resource "aws_batch_job_definition" "gpu" {
  name = "${var.name}-gpu"
  type = "container"

  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.rigfl.repository_url}:${var.image_tag}"
    jobRoleArn = aws_iam_role.job.arn
    resourceRequirements = [
      {
        type  = "GPU"
        value = "1"
      },
      {
        type  = "VCPU"
        value = tostring(var.vcpus_per_job)
      },
      {
        type  = "MEMORY"
        value = tostring(var.memory_mib_per_job)
      },
    ]
    environment = [
      {
        name  = "PYTHONUNBUFFERED"
        value = "1"
      },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.batch.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = var.name
      }
    }
  })

  retry_strategy {
    attempts = 2
  }

  timeout {
    attempt_duration_seconds = var.job_timeout_seconds
  }
}
