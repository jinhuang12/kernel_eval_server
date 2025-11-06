#!/bin/bash
# Deploy and test the MCP server refactoring on EC2 instance

set -e  # Exit on error

echo "🚀 Deploying MCP Server Refactoring to EC2..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# EC2 instance details
EC2_HOST="p5e-cmh"
REMOTE_DIR="~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2"

echo -e "${YELLOW}Step 1: Syncing files to EC2...${NC}"
rsync -avz --exclude='*.pyc' --exclude='__pycache__' --exclude='.git' \
  --exclude='old_tests' --exclude='*.ncu-rep' \
  ./ ${EC2_HOST}:${REMOTE_DIR}/

echo -e "${YELLOW}Step 2: Running tests on EC2...${NC}"
ssh ${EC2_HOST} << 'EOF'
cd ~/AIRE-TFL-KernelBench/KernelBench/scripts/cuda_eval_server_v2

# Activate conda environment if needed
source /home/ec2-user/miniconda3/bin/activate base 2>/dev/null || true

echo "Testing Python syntax..."
python3 -c "import ast; ast.parse(open('shared/models.py').read())" || exit 1
echo "✓ Syntax check passed"

echo "Running refactoring tests..."
python3 test_refactoring.py || exit 1
echo "✓ All tests passed"

# Backup original files
echo "Creating backups..."
cp mcp_server.py mcp_server.backup.py 2>/dev/null || true
echo "✓ Backups created"

# Deploy new MCP server
echo "Deploying refactored MCP server..."
cp mcp_server_refactored.py mcp_server.py
echo "✓ MCP server deployed"

echo "Deployment successful!"
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. SSH to EC2: ssh ${EC2_HOST}"
    echo "2. Start server: cd ${REMOTE_DIR} && python3 main.py --mode mcp --port 8001"
    echo "3. Test MCP endpoints with the test client"
    echo ""
    echo "To rollback if needed:"
    echo "  ssh ${EC2_HOST} 'cd ${REMOTE_DIR} && mv mcp_server.backup.py mcp_server.py'"
else
    echo -e "${RED}❌ Deployment failed! Check the error messages above.${NC}"
    exit 1
fi