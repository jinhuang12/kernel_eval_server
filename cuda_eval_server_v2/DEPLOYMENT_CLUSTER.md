# CUDA Evaluation Server - Kubernetes Deployment Guide

## Table of Contents
1. [DS3 Cluster Connection](#ds3-cluster-connection)
2. [Architecture Overview](#architecture-overview)
3. [Kubernetes Files](#kubernetes-files)
4. [Deployment Instructions](#deployment-instructions)
5. [Accessing the Server](#accessing-the-server)
6. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
7. [Clean-up](#clean-up)

---

## DS3 Cluster Connection

### Prerequisites
Before deploying, you must connect to the DS3 HyperPod cluster:

```bash
# 1. Assume the IAM Role that gives Silverhand access to the DS3 cluster account
source ./assume-role.sh --role-arn arn:aws:iam::592892253131:role/hyperpod-silverhand-team-scientist-role

# 2. Update the Kubernetes Config to connect to the DS3 Cluster
aws eks update-kubeconfig --region us-east-1 --name sagemaker-hyperpod-eks-cluster-ds3

# 3. Set default namespace to Silverhand
kubectl config set-context --current --namespace=hyperpod-ns-ironfist

# 4. Verify connection
kubectl get nodes
kubectl get pods -n hyperpod-ns-ironfist
```

### Quick Start Guide

After connecting to the DS3 cluster, here's how to quickly access the CUDA eval server:

```bash
# 1. Check if the deployment is running
kubectl get pods -n hyperpod-ns-ironfist -l app=cuda-eval-server-v2

# 2. If pods are running, check the NLB service status
kubectl get svc cuda-eval-nlb-internal -n hyperpod-ns-ironfist

# 3. Create port forward to the NLB service
kubectl port-forward -n hyperpod-ns-ironfist service/cuda-eval-nlb-internal 8000:8000

# 4. In a new terminal, test the connection
curl http://localhost:8000/health

# 5. Now you can use the server locally
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**Note**: Keep the port-forward command running in its terminal while accessing the service.

---

## Architecture Overview

### SageMaker HyperPod
- **Purpose**: AWS managed service for resilient ML clusters
- **Integration**: 1-to-1 mapping between EKS control plane and HyperPod worker nodes
- **Namespace**: `hyperpod-ns-ironfist` - dedicated namespace for Silverhand team

### Kueue Integration
- **Queue System**: Manages GPU resource allocation and job scheduling
- **LocalQueue**: `hyperpod-ns-ironfist-localqueue` - routes workloads to appropriate resources
- **Labels**: All pods must include `kueue.x-k8s.io/queue-name` label for scheduling

### GPU Instance Types
| Instance Type | GPU Type | GPUs | Memory | vCPUs | EFA Interfaces |
|--------------|----------|------|---------|-------|----------------|
| ml.p4d.24xlarge | A100 | 8 | 1.1TB | 96 | 4 |
| ml.p5.48xlarge | H100 | 8 | 2TB | 192 | 32 |
| ml.p5e.48xlarge | H200 | 8 | 2TB | 192 | 32 |

---

## Kubernetes Files

### Core Deployment Files

#### 1. `deploy-a100.yaml`
- **Purpose**: Deploys CUDA eval server on A100 GPUs
- **Components**:
  - Deployment with 2 replicas
  - ClusterIP Service
  - Resource requests: 8 GPUs, 48 CPUs, 576Gi memory, 4 EFA
- **Node Selector**: `ml.p4d.24xlarge`
- **Volumes**: Shared memory, local NVMe, FSx storage

#### 2. `deploy-h100.yaml` & `deploy-h200.yaml`
- Similar structure for H100/H200 deployments
- Higher resource allocations (32 EFA interfaces)
- Different node selectors (`ml.p5.48xlarge`, `ml.p5e.48xlarge`)

#### 3. `nlb-internal-service.yaml`
- **Purpose**: Internal Network Load Balancer for secure access
- **Type**: LoadBalancer with internal scheme (VPC access only)
- **Port**: 8000
- **Access Method**: kubectl port-forward or VPC-internal access
- **Benefits**: No ALB Controller needed, works out-of-the-box

#### 4. `nlb-external-service.yaml`
- **Purpose**: Optional external Network Load Balancer
- **Type**: LoadBalancer with internet-facing scheme
- **Port**: 80 (maps to container port 8000)
- **Warning**: Exposes service to internet - use with caution
- **Security**: Can restrict by source IP in annotations

#### 5. `deploy.sh`
- **Purpose**: Automated deployment script
- **Actions**:
  - Validates namespace exists
  - Substitutes AWS variables (region, account ID)
  - Deploys resources to HyperPod namespace

---

## Deployment Instructions

### Quick Deploy (A100 Only)

```bash
# 1. Connect to DS3 cluster (see DS3 Cluster Connection above)

# 2. Set AWS environment variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=592892253131

# 3. Run deployment script
./deploy.sh

# 4. Deploy NLB service for access (choose one):
# For secure internal access (recommended):
kubectl apply -f nlb-internal-service.yaml

# OR for external access (use with caution):
kubectl apply -f nlb-external-service.yaml
```

### Manual Deployment

```bash
# Deploy A100 configuration
sed -e "s/\${AWS_REGION}/$AWS_REGION/g" \
    -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
    deploy-a100.yaml | kubectl apply -n hyperpod-ns-ironfist -f -

# Deploy NLB service (choose one)
kubectl apply -n hyperpod-ns-ironfist -f nlb-internal-service.yaml
# OR
kubectl apply -n hyperpod-ns-ironfist -f nlb-external-service.yaml
```

### Multi-GPU Deployment

To deploy H100 or H200 variants:
```bash
# For H100
kubectl apply -n hyperpod-ns-ironfist -f deploy-h100.yaml

# For H200
kubectl apply -n hyperpod-ns-ironfist -f deploy-h200.yaml
```

---

## Accessing the Server

### Method 1: kubectl Port Forward (Local Development)

```bash
# Forward from the ClusterIP service 
kubectl port-forward -n hyperpod-ns-ironfist service/cuda-eval-service-v2-a100 8000:8000

# Or forward from a specific pod
kubectl port-forward -n hyperpod-ns-ironfist \
  pod/cuda-eval-server-v2-a100-xxxxx 8000:8000

# Access locally
curl http://localhost:8000/health
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Method 2: Direct NLB Access (VPC)

```bash
# Get the NLB endpoint
kubectl get svc cuda-eval-nlb-internal -n hyperpod-ns-ironfist
# Will output an External IP, which can be used to communicate with the Profiling Server

# Access directly from any EC2 instance in the VPC
curl http://a5f6ad4be5cf348a68522e40ebbda742-0a31398741634b2e.elb.us-east-1.amazonaws.com:8000/health
```

### Method 3: Internal Service Access (Within Cluster)

From within the cluster or a test/training pod:
```bash
# Create test pod
kubectl run test-curl --image=curlimages/curl \
  -n hyperpod-ns-ironfist --rm -it -- sh

# Inside the pod
curl http://cuda-eval-service-v2-a100:8000/health
```

### Method 4: External NLB (Optional - Direct Internet Access)

To send requests to the NLB
```bash
# Only if you deployed nlb-external-service.yaml
# Get the external NLB DNS name
kubectl get svc cuda-eval-nlb-external -n hyperpod-ns-ironfist
# Note the EXTERNAL-IP (will be like: xxxxx.us-east-1.elb.amazonaws.com)

# Access directly from anywhere on the internet
curl http://<external-nlb-dns-name>/health
curl -X POST http://<external-nlb-dns-name>/evaluate \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**Warning**: External NLB exposes your service to the open internet! Use with caution and configure security groups appropriately. For most use cases, prefer the Internal NLB (Method 2) which is only accessible within your VPC.

---

## Monitoring and Troubleshooting

### Manual Deployment of Updated Image
```bash
# Set image to latest & then 2nd command updates the image, triggering a rolling update
kubectl set image deployment/cuda-eval-server-v2-a100 -n hyperpod-ns-ironfist cuda-eval-server-v2=592892253131.dkr.ecr.us-east-1.amazonaws.com/cuda-eval-server:latest
kubectl set image deployment/cuda-eval-server-v2-a100 -n hyperpod-ns-ironfist cuda-eval-server-v2=592892253131.dkr.ecr.us-east-1.amazonaws.com/cuda-eval-server   
```

### Check Deployment Status

```bash
# View pod status
kubectl get pods -n hyperpod-ns-ironfist -l app=cuda-eval-server-v2

# Check deployment rollout
kubectl rollout status deployment/cuda-eval-server-v2-a100 -n hyperpod-ns-ironfist

# Describe pods for details
kubectl describe pods -n hyperpod-ns-ironfist -l app=cuda-eval-server-v2
```

### View Logs

```bash
# View pod logs
kubectl logs -n hyperpod-ns-ironfist -l app=cuda-eval-server-v2 --tail=50

# Follow logs in real-time
kubectl logs -n hyperpod-ns-ironfist -l app=cuda-eval-server-v2 -f

# Logs from specific pod
kubectl logs -n hyperpod-ns-ironfist pod/cuda-eval-server-v2-a100-xxxxx
```

### Check Resources

```bash
# View resource usage
kubectl top pods -n hyperpod-ns-ironfist

# Check node resources
kubectl describe nodes | grep -A 5 "Allocated resources"

# View GPU allocation
kubectl describe nodes | grep nvidia.com/gpu
```

### Common Issues

#### Pods Stuck in Pending
- **Cause**: No nodes match selector or insufficient resources
- **Fix**: Check node labels and available resources
```bash
kubectl get nodes --show-labels
kubectl describe pod <pod-name> -n hyperpod-ns-ironfist
```

#### ImagePullBackOff
- **Cause**: ECR authentication or image not found
- **Fix**: Verify ECR login and image exists
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  592892253131.dkr.ecr.us-east-1.amazonaws.com

aws ecr describe-images --repository-name cuda-eval-server \
  --region us-east-1
```

#### CrashLoopBackOff
- **Cause**: Application crashing on startup
- **Fix**: Check pod logs for error messages
```bash
kubectl logs -n hyperpod-ns-ironfist <pod-name> --previous
```

---

## Clean-up

### Remove Specific Components

```bash
# Delete deployment
kubectl delete deployment cuda-eval-server-v2-a100 -n hyperpod-ns-ironfist

# Delete services
kubectl delete service cuda-eval-service-v2-a100 -n hyperpod-ns-ironfist
kubectl delete service cuda-eval-nlb-internal -n hyperpod-ns-ironfist
kubectl delete service cuda-eval-nlb-external -n hyperpod-ns-ironfist
```

### Remove Everything

```bash
# Delete all resources with app label
kubectl delete deployment,service \
  -l app=cuda-eval-server-v2 -n hyperpod-ns-ironfist
```

---

## Additional Notes

### Security Considerations
- All deployments run as root (required for NCU profiling)
- Security context includes SYS_ADMIN capability for GPU profiling
- Internal NLB restricts access to VPC only (use kubectl port-forward for secure remote access)
- External NLB exposes service to internet - restrict with security groups or source IP annotations

### Resource Management
- Kueue manages GPU allocation and scheduling
- EFA interfaces enable high-performance networking for multi-node training
- FSx provides shared storage across pods

### Image Management
- Images stored in ECR: `592892253131.dkr.ecr.us-east-1.amazonaws.com/cuda-eval-server`
- Always use `imagePullPolicy: Always` for latest updates
- Tag images with meaningful versions for production

### Scaling
- Adjust `replicas` in deployment files to scale pods
- Consider resource availability when scaling
- Monitor GPU utilization to optimize resource usage

---

## Support

For issues or questions:
1. Check pod logs and events
2. Verify cluster connection and namespace
3. Ensure all prerequisites are met
4. Contact the platform team for HyperPod-specific issues