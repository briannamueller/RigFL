variable "aws_profile" {
  description = "Local AWS CLI profile used by Terraform."
  type        = string
  default     = "rigfl"
}

variable "aws_region" {
  description = "AWS region where RigFL resources are created."
  type        = string
  default     = "us-east-2"
}

variable "name" {
  description = "Prefix used to name RigFL cloud resources."
  type        = string
  default     = "rigfl"
}

variable "image_tag" {
  description = "Immutable ECR image tag used by the Batch job definition."
  type        = string
  default     = "bootstrap"

  validation {
    condition     = length(trimspace(var.image_tag)) > 0
    error_message = "image_tag must not be empty."
  }
}

variable "gpu_instance_types" {
  description = "EC2 GPU instance types that AWS Batch may launch."
  type        = list(string)
  default     = ["g4dn.xlarge"]
}

variable "max_vcpus" {
  description = "Maximum aggregate vCPUs for the GPU compute environment."
  type        = number
  default     = 4
}

variable "vcpus_per_job" {
  description = "vCPUs reserved for each RigFL Batch job."
  type        = number
  default     = 4
}

variable "memory_mib_per_job" {
  description = "Memory in MiB reserved for each RigFL Batch job."
  type        = number
  default     = 14000
}

variable "job_timeout_seconds" {
  description = "Hard timeout for one RigFL task."
  type        = number
  default     = 21600
}
