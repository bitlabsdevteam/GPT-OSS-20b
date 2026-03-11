# Blockers and Assumptions

## Assumptions
1. Node has 3x A100 with functioning NCCL/NVLink
2. CUDA/NVIDIA driver stack is healthy
3. We can use bf16 and flash-attn-compatible environment
4. GitHub push permissions are available
5. Status delivery integrations (Slack + email) can be configured from this environment

## Current Blockers
1. **Slack delivery path not yet configured in this runtime**
   - Need webhook URL or bot token/channel mapping
2. **Email delivery path not yet configured in this runtime**
   - Need SMTP account/tooling or approved mail CLI configuration
3. **Perplexity-backed research access not configured**
   - Need API key/CLI or approved workflow
4. Repository is currently empty; baseline architecture is being authored from scratch

## Risk Notes
- Dense 20B from-scratch quality target on 3x A100 is not a realistic near-term objective
- Multi-parallel stack complexity (TP+PP+SP+EP) raises debugging burden; we will stage features incrementally
