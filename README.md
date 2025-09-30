# GMTS
GMTS: GRADIENT MAGNITUDE-BASED TOKEN SELECTION IMPROVES RLVR TRAINING FOR LLM REASONING

## GMTS Frameworks 

This repo contains two sub-frameworks, one is based on [VERL](https://github.com/volcengine/verl) and one is based on [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero):
- `GMTS-Framework1/` → **GRPO-Zero-GMTS**
- `GMTS-Framework2/` → **VERL-GMTS**

We provide **GMTS-Framework1/** for resource-constrained setups: **it does not require VLLM**, and it can train/test **1.5B and 7B** models with minimal resources, with a simple, easy-to-implement KV-cache.

## Quick Start

