output "artifacts_bucket" {
  description = "Private S3 bucket for grids and experiment results."
  value       = aws_s3_bucket.artifacts.id
}

output "ecr_repository_url" {
  description = "ECR repository to receive the RigFL GPU image."
  value       = aws_ecr_repository.rigfl.repository_url
}

output "job_definition_arn" {
  description = "AWS Batch GPU job definition."
  value       = aws_batch_job_definition.gpu.arn
}

output "job_queue_arn" {
  description = "AWS Batch GPU job queue."
  value       = aws_batch_job_queue.gpu.arn
}

output "idle_gpu_capacity" {
  description = "The environment's idle vCPU capacity; zero means no idle GPU instances."
  value       = aws_batch_compute_environment.gpu.compute_resources[0].min_vcpus
}
