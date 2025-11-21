# Proposal: Automated vLLM Kernel Benchmark Generation Service

**Author**: Kernel Evaluation Team
**Date**: 2025-11-21
**Status**: Draft for Review
**Target**: H1 2025 Implementation

---

## Executive Summary

We propose integrating an automated vLLM kernel benchmark generation workflow into our existing CUDA Evaluation Server. This enhancement will enable on-demand generation of **standalone, production-ready benchmark scripts** for any vLLM kernel through a single API call.

**Key Benefits:**
- **Fully automated**: From kernel name → standalone Python benchmark in minutes
- **Zero manual work**: Agent extracts dependencies, inlines code, generates tests
- **GPU-verified**: Executes on reserved GPU hardware during generation
- **Production quality**: Follows vLLM's own test patterns and validation logic

**Impact**: Reduces benchmark creation time from **days of manual work** → **30 minutes automated**.

---

## Problem Statement

### Current Workflow

Creating benchmark scripts for vLLM kernels currently requires:

1. **Manual code exploration** (2-4 hours)
   - Navigate vLLM codebase to find kernel location
   - Identify dependencies and imports
   - Understand input/output contracts from tests

2. **Dependency extraction** (4-8 hours)
   - Manually inline all `vllm.*` imports
   - Trace transitive dependencies
   - Handle external library bindings (torch.ops)

3. **Benchmark development** (8-16 hours)
   - Write parameter sweeps
   - Implement reference implementations
   - Add correctness validation
   - Debug CUDA errors and shape mismatches

4. **Hardware verification** (2-4 hours)
   - Test on target GPU
   - Iterate on bugs found only in deployment

**Total: 16-32 hours per kernel** (2-4 days of engineering time)

### Pain Points

- **Not scalable**: vLLM has 100+ kernels, constantly evolving
- **Expertise required**: Deep vLLM knowledge needed
- **Error-prone**: Easy to miss dependencies or validation logic
- **No reusability**: Each kernel extraction is unique

---

## Proposed Solution

Add a new FastAPI endpoint `/vllm-benchmark-generate` that accepts a kernel name and automatically:

1. **Researches** the kernel in vLLM codebase (tests, benchmarks, source)
2. **Extracts** kernel + dependencies → standalone Python script (zero vLLM imports)
3. **Generates** comprehensive benchmark suite with parameter sweeps
4. **Validates** on reserved GPU hardware
5. **Returns** production-ready Python scripts to client

### Example Usage

```bash
curl -X POST http://localhost:8000/vllm-benchmark-generate \
  -H "Content-Type: application/json" \
  -d '{
    "symbol_name": "fused_moe_kernel",
    "model": "llama-70b",
    "hardware": "H200",
    "precision": "FP16"
  }'

# Response (after ~30 minutes):
{
  "job_id": "abc-123",
  "bundle_script": "#!/usr/bin/env python\n# Standalone fused_moe_kernel...",
  "benchmark_script": "#!/usr/bin/env python\n# Benchmark suite...",
  "research_plan": "# Kernel Analysis\n...",
  "bundle_validated": true,
  "benchmark_validated": true,
  "sample_metrics": {
    "latency_ms": 2.34,
    "throughput_tokens_per_sec": 12500
  }
}
```

### What Makes This Possible?

We have a **proven workflow** (the `/generate-benchmark` slash command) that:
- Successfully extracts complex vLLM kernels (FlashAttention, MoE, PagedAttention)
- Handles external library bindings (torch.ops)
- Generates test-driven benchmark suites
- Has been battle-tested on production kernels

**Now we make it a service.**

---

## Implementation Options

### Option A: Claude Code CLI (Headless Mode) ⭐ **RECOMMENDED**

**Architecture:**
```
FastAPI Endpoint
    ↓
Reserve GPU
    ↓
Spawn subprocess: claude-code --headless --task "slash_command.md"
    ↓
Claude Code executes workflow autonomously
    ↓
Collect generated files from workspace
    ↓
Return to client
```

**Pros:**
- ✅ **Minimal integration**: Just subprocess + file I/O
- ✅ **Full Claude Code features**: All tools, sub-agents, MCP servers
- ✅ **Proven workflow**: Uses existing `/generate-benchmark` slash command as-is
- ✅ **Easy debugging**: Can run same command interactively for development
- ✅ **Fastest to implement**: ~2-3 days

**Cons:**
- ⚠️ Requires Claude Code CLI installed in server environment
- ⚠️ Less programmatic control (parsing structured output from markdown)
- ⚠️ Potential environment isolation challenges

**Implementation Effort:** 3-5 days

---

### Option B: Anthropic Agent SDK

**Architecture:**
```
FastAPI Endpoint
    ↓
Reserve GPU
    ↓
Instantiate Anthropic Agent with tools (bash, read, write, edit)
    ↓
Agent executes workflow with tool calls
    ↓
Collect results from agent response
    ↓
Return structured data to client
```

**Pros:**
- ✅ **Full programmatic control**: Structured tool calls and responses
- ✅ **Better observability**: Track agent state, iterations, tool usage
- ✅ **No external dependencies**: Pure Python + Anthropic SDK
- ✅ **Easier testing**: Mock tools for unit tests
- ✅ **Structured outputs**: JSON responses, not markdown parsing

**Cons:**
- ⚠️ More implementation complexity (~2x LOC)
- ⚠️ Need to reimplement tool execution layer
- ⚠️ No direct access to Claude Code's advanced features (MCP, etc.)

**Implementation Effort:** 7-10 days

---

### Option C: Hybrid Approach

**Architecture:**
```
Use Agent SDK for orchestration + subprocess for Explore subagent
```

**Pros:**
- ✅ Best of both worlds: control + power
- ✅ Agent SDK for main flow, delegate research to Claude Code Explore

**Cons:**
- ⚠️ Most complex: Two agent systems
- ⚠️ Longer development time

**Implementation Effort:** 10-14 days

---

## Recommended Implementation: Option A (Claude Code CLI)

### Rationale

1. **Speed to production**: 3-5 days vs 7-10 days for SDK
2. **Lower risk**: Leverage proven slash command with minimal changes
3. **Maintainability**: Slash command updates automatically benefit server
4. **Developer experience**: Can test workflow interactively before deploying

### High-Level Integration Plan

#### Phase 1: Environment Setup (Day 1)

```bash
# Install Claude Code CLI in server container
npm install -g @anthropic-ai/claude-code

# Verify installation
claude-code --version

# Configure Anthropic API key
export ANTHROPIC_API_KEY=<key>
```

#### Phase 2: Endpoint Implementation (Days 2-3)

**New Models** (`shared/models.py`):
```python
class VLLMBenchmarkGenerateRequest(BaseModel):
    symbol_name: str
    model: Optional[str] = "llama-70b"
    hardware: Optional[str] = None  # Inferred from GPU
    precision: Optional[str] = "FP16"
    skip_remote: bool = False
    vllm_repo_path: str = "/workspace/vllm"
    timeout: int = 1800  # 30 minutes

class VLLMBenchmarkGenerateResponse(BaseModel):
    job_id: str
    symbol_name: str
    bundle_script: str  # Generated bundle
    benchmark_script: str  # Generated benchmark
    research_plan: str  # Research findings
    stages_completed: List[str]
    bundle_validated: bool
    sample_metrics: Optional[Dict]
    status: str
```

**FastAPI Endpoint** (`app.py`):
```python
@app.post("/vllm-benchmark-generate")
async def vllm_benchmark_generate(request: dict):
    # Parse request
    # Submit to JobManager
    # Wait for completion
    # Return response
```

**JobManager Extension** (`orchestration/job_manager.py`):
```python
async def submit_vllm_benchmark_job(self, request):
    # Create job ID
    # Acquire GPU
    # Execute Claude Code in subprocess
    # Collect generated files
    # Validate outputs
    # Return results

async def _execute_claude_code_headless(self, request, gpu_id):
    # Create workspace
    # Write slash command to file
    # Run: claude-code --headless --task slash_command.md
    # Monitor subprocess output
    # Parse results
    # Return artifacts
```

#### Phase 3: Testing & Validation (Day 4)

**Test Suite:**
1. **Unit tests**: Request/response serialization
2. **Integration tests**: Simple kernel (e.g., `silu_and_mul`)
3. **E2E tests**: Complex kernel (e.g., `fused_moe_kernel`)
4. **GPU tests**: Verify CUDA_VISIBLE_DEVICES isolation

**Success Criteria:**
- [ ] Simple kernel extracts in <5 minutes
- [ ] Complex kernel extracts in <30 minutes
- [ ] Generated bundles run successfully
- [ ] Benchmark suites produce metrics
- [ ] No vLLM imports in generated code

#### Phase 4: Documentation & Deployment (Day 5)

1. **API documentation**: OpenAPI spec, examples
2. **User guide**: How to use the endpoint
3. **Operator guide**: Troubleshooting, monitoring
4. **Deploy to staging**: Validate on staging cluster
5. **Production rollout**: Gradual rollout with monitoring

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Server                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ POST /vllm-benchmark-generate                        │  │
│  │   - Validate request                                 │  │
│  │   - Submit to JobManager                             │  │
│  │   - Return job_id                                    │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                         │
│  ┌────────────────▼─────────────────────────────────────┐  │
│  │ JobManager                                           │  │
│  │   - GPU Resource Manager (existing)                  │  │
│  │   - Job State Tracking (existing)                    │  │
│  │   - NEW: Claude Code Executor                        │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                         │
└───────────────────┼─────────────────────────────────────────┘
                    │
                    │ Acquire GPU
                    ▼
        ┌───────────────────────┐
        │ GPU Resource Pool     │
        │ (existing)            │
        └───────────┬───────────┘
                    │
                    │ Reserve GPU N
                    ▼
    ┌───────────────────────────────────────┐
    │ Subprocess Executor                   │
    │                                       │
    │ $ claude-code --headless \            │
    │     --task slash_command.md \         │
    │     --workspace /tmp/job_XYZ          │
    │                                       │
    │ Env: CUDA_VISIBLE_DEVICES=N           │
    └───────────────┬───────────────────────┘
                    │
                    │ Claude executes workflow
                    ▼
        ┌───────────────────────┐
        │ Claude Code Agent     │
        │  - Stage 1: Research  │
        │  - Stage 2.1: Extract │
        │  - Stage 2.2: Bench   │
        │  - Stage 2.3: Verify  │
        └───────────┬───────────┘
                    │
                    │ Writes files
                    ▼
        ┌───────────────────────────────────┐
        │ /tmp/job_XYZ/                     │
        │   bundled_benchmarks/             │
        │     fused_moe_kernel.py           │
        │     fused_moe_kernel_bench.py     │
        │   tmp/                            │
        │     plan_fused_moe_kernel.md      │
        └───────────┬───────────────────────┘
                    │
                    │ Read artifacts
                    ▼
        ┌───────────────────────┐
        │ Parse & Validate      │
        │  - Check files exist  │
        │  - Validate syntax    │
        │  - Extract metrics    │
        └───────────┬───────────┘
                    │
                    │ Return results
                    ▼
        ┌───────────────────────┐
        │ Client Response       │
        │  - Bundle script      │
        │  - Benchmark script   │
        │  - Research plan      │
        │  - Validation status  │
        └───────────────────────┘
```

---

## Integration with Existing Infrastructure

### GPU Resource Management (No Changes)

```python
# Existing pattern continues to work
async with self.gpu_manager.acquire_gpu(timeout=1800, job_id=job_id) as gpu_id:
    # Claude Code subprocess runs here
    # CUDA_VISIBLE_DEVICES={gpu_id} ensures isolation
    pass  # GPU auto-released
```

### Job Tracking (Minimal Changes)

```python
# Existing JobState works with new job type
job_state = JobState(
    job_id=job_id,
    status="running_claude_code",  # New status
    result=None,
    error=None
)
```

### Health Monitoring (Existing Endpoints)

```python
# Existing /health and /stats endpoints automatically track new job type
GET /health  # Reports Claude Code job status
GET /stats   # Includes benchmark generation metrics
GET /job/{job_id}  # Works for new job type
```

---

## Success Metrics

### Functional Metrics
- [ ] **Correctness**: 95%+ of generated bundles run without errors
- [ ] **Completeness**: 100% of generated bundles have zero vLLM imports
- [ ] **Validation**: 90%+ of benchmarks pass correctness checks

### Performance Metrics
- [ ] **Latency**: Simple kernels <5 min, complex kernels <30 min
- [ ] **Throughput**: Support 10 concurrent generation jobs
- [ ] **GPU efficiency**: GPU utilization >80% during generation

### Business Metrics
- [ ] **Time savings**: 16-32 hours → 0.5 hours (96% reduction)
- [ ] **Coverage**: Enable benchmarking for 100+ vLLM kernels
- [ ] **Reliability**: 99% uptime for benchmark generation service

---

## Risk Assessment & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Claude Code CLI not stable in headless mode** | High | Low | Fallback to Agent SDK; test headless mode early |
| **vLLM codebase changes break extraction** | Medium | Medium | Version pin vLLM; add compatibility checks |
| **GPU OOM during generation** | Medium | Low | Add memory checks; reduce batch sizes dynamically |
| **Long-running jobs timeout** | Low | Medium | Increase timeout; add job resumption capability |
| **External library kernels fail** | Low | High | Expected; provide clear warnings + build instructions |
| **Concurrent job contention** | Medium | Low | Queue system already handles this (existing) |

---

## Timeline & Milestones

### Sprint 1 (Week 1): Foundation
- **Day 1**: Environment setup, Claude Code CLI installation
- **Day 2**: Request/response models, endpoint skeleton
- **Day 3**: JobManager extension, subprocess executor
- **Day 4**: Testing with simple kernel (silu_and_mul)
- **Day 5**: Documentation, code review

**Deliverable**: Working endpoint for simple kernels

### Sprint 2 (Week 2): Production Readiness
- **Day 1-2**: Complex kernel testing (MoE, attention)
- **Day 3**: Error handling, retry logic, monitoring
- **Day 4**: Performance optimization, concurrent jobs
- **Day 5**: Staging deployment, integration tests

**Deliverable**: Production-ready service

### Sprint 3 (Week 3): Rollout & Iteration
- **Day 1-2**: Production deployment, monitoring setup
- **Day 3-4**: User feedback, bug fixes
- **Day 5**: Documentation finalization, team training

**Deliverable**: GA release with documentation

---

## Alternative Considered: Manual Extraction (Status Quo)

**Why not continue manual extraction?**

| Aspect | Manual | Automated (This Proposal) |
|--------|--------|---------------------------|
| Time per kernel | 16-32 hours | 0.5 hours |
| Scalability | Poor (100+ kernels) | Excellent |
| Quality consistency | Varies by engineer | Consistent (follows vLLM patterns) |
| Expertise required | High (vLLM expert) | None (API call) |
| Maintenance burden | High (each vLLM update) | Low (automatic adaptation) |

**Verdict**: Automation is 32-64x faster and removes bottleneck.

---

## Open Questions

1. **vLLM version management**: Pin to specific version or track latest?
   - **Recommendation**: Pin initially, add version parameter later

2. **Output artifact storage**: Where to store generated benchmarks long-term?
   - **Recommendation**: Return in response; client stores if needed; add S3 backup later

3. **Remote verification**: Enable p5e-cmh testing in Phase 1?
   - **Recommendation**: Add in Phase 2; focus on local validation first

4. **Concurrent job limits**: How many simultaneous extractions?
   - **Recommendation**: Start with GPU pool size (typically 8); monitor and adjust

5. **Pricing model**: Charge per generation or flat rate?
   - **Recommendation**: Track costs first; decide pricing model after metrics

---

## Conclusion

Integrating automated vLLM kernel benchmark generation into our CUDA Evaluation Server represents a **high-value, low-risk enhancement** that:

- **Eliminates manual bottleneck**: 96% time reduction per kernel
- **Enables scale**: Benchmark 100+ vLLM kernels efficiently
- **Leverages proven technology**: Battle-tested slash command workflow
- **Minimal integration**: Reuses existing GPU management and job infrastructure
- **Fast time-to-value**: 3-5 days for Option A implementation

**Recommendation**: Proceed with **Option A (Claude Code CLI in headless mode)** for fastest deployment and lowest integration complexity.

---

## Appendix: Example Generated Output

### Input Request
```json
{
  "symbol_name": "fused_moe_kernel",
  "model": "llama-70b",
  "hardware": "H200",
  "precision": "FP16"
}
```

### Output Response (Snippet)
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol_name": "fused_moe_kernel",
  "bundle_script": "#!/usr/bin/env python3\n\"\"\"\nfused_moe_kernel - Standalone Bundle\n...\n\"\"\"",
  "benchmark_script": "#!/usr/bin/env python3\n\"\"\"\nBenchmark Suite for fused_moe_kernel\n...\n\"\"\"",
  "research_plan": "# fused_moe_kernel Analysis\n\n## Kernel Type\nPure Triton kernel...",
  "stages_completed": [
    "Stage 0: Initialization",
    "Stage 1: Research",
    "Stage 2.1: Bundle Creation",
    "Stage 2.2: Benchmark Suite",
    "Stage 2.3: Verification"
  ],
  "bundle_validated": true,
  "benchmark_validated": true,
  "sample_metrics": {
    "latency_ms": 2.34,
    "throughput_tokens_per_sec": 12500,
    "memory_gb": 0.256
  },
  "gpu_type": "H200",
  "status": "success"
}
```

---

**Ready for review and feedback.**
