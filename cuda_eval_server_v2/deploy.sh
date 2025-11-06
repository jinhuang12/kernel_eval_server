#!/bin/bash

# Deployment script for A100 CUDA Evaluation Server on AWS EKS (HyperPod)

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-2}
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
NAMESPACE="hyperpod-ns-silverhand"

echo "Deploying CUDA Evaluation Server for A100 on HyperPod"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo "Namespace: $NAMESPACE"

# Verify namespace exists
if ! kubectl get namespace $NAMESPACE &>/dev/null; then
    echo "Warning: Namespace '$NAMESPACE' not found. Please ensure you're connected to the correct cluster."
    exit 1
fi

# Deploy A100 configuration
echo "Deploying A100 deployment to namespace $NAMESPACE..."
sed -e "s/\${AWS_REGION}/$AWS_REGION/g" \
    -e "s/\${AWS_ACCOUNT_ID}/$AWS_ACCOUNT_ID/g" \
    deploy-a100.yaml | kubectl apply -n $NAMESPACE -f -

# Deploy ALB ingress
echo "Deploying ALB ingress to namespace $NAMESPACE..."
kubectl apply -n $NAMESPACE -f alb-ingress.yaml

echo "Deployment complete"
echo ""
echo "Routes available:"
echo "  /gpu/a100 - Explicit route to A100 GPUs"
echo "  /         - Default route (A100)"