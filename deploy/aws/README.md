# RigFL on AWS Batch

This Terraform configuration defines a deliberately small GPU experiment platform:

- one private, versioned S3 bucket for grids and results;
- one immutable, scan-on-push ECR repository;
- a least-privilege Batch job role that can access only that S3 bucket;
- one AWS Batch Spot compute environment backed by `g4dn.xlarge`;
- one job queue and one GPU job definition; and
- a CloudWatch log group with 14-day retention.

The compute environment has `min_vcpus = 0` and `desired_vcpus = 0`, so it does not keep an idle GPU instance running. Its initial `max_vcpus = 4` permits only one `g4dn.xlarge` worker at a time. It uses the account's default VPC and public subnets so this learning deployment does not require a continuously billed NAT gateway. An EC2 public IPv4 address and the GPU instance itself can still incur charges while a job is running.

## Account safety gates

Before applying this configuration:

1. Upgrade from the AWS free plan only if AWS requires it for GPU services.
2. Request a quota of **4 vCPUs** for either `Running On-Demand G and VT instances` or `All G and VT Spot Instance Requests` in `us-east-2`. The Spot quota is needed for the default configuration here.
3. Review `terraform plan` before approving any AWS changes.
4. Do not submit a Batch job until the expected per-job cost has been reviewed.

## Deployment sequence

Initialize and inspect the plan:

```bash
cd deploy/aws
terraform init
terraform validate
terraform plan
```

After approval, create only ECR first:

```bash
terraform apply -target=aws_ecr_repository.rigfl
```

Build and push the GPU image with a unique tag, preferably the Git commit SHA. Then create the remaining infrastructure using that exact tag:

```bash
terraform apply -var='image_tag=YOUR_IMMUTABLE_TAG'
```

Preview a submission first (this performs no AWS calls):

```bash
python -m rigfl.experiment.aws_submit \
  --grid ../../results/YOUR_SWEEP/grid.jsonl \
  --bucket "$(terraform output -raw artifacts_bucket)" \
  --job-queue "$(terraform output -raw job_queue_arn)" \
  --job-definition "$(terraform output -raw job_definition_arn)"
```

After reviewing the task count and S3 locations, repeat the command with `--submit`. That explicit flag uploads the immutable, content-addressed grid and launches the Batch array job. The array index is zero-based in AWS and is translated to RigFL's one-based task IDs by `rigfl.experiment.aws_batch`.

Terraform state may contain infrastructure details. Keep `*.tfstate` and saved plan files out of version control; commit `.terraform.lock.hcl` so provider versions remain reproducible.
