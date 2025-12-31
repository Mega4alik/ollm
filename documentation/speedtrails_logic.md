Logic Summary for Your Next Steps

    The Swap Ratio: Your library's competitive edge is shown best with Qwen2.5-14B and Moonlight-16B. With only ~3B active parameters, you are proving that a high-IQ model can run with less than 6GB of active VRAM if the SSD swapping is optimized.

    The Baseline: Qwen3-8B is your "no-swap" control. If the 16B MoE models (using 3B active) are faster than the 8B dense model, your library's indexing is officially state-of-the-art.

    The Architecture Diversity: You now have Mistral, Falcon, and DeepSeek/Qwen styles. Ensure your loader can differentiate between block_sparse_moe (Mistral/Falcon) and experts (DeepSeek) naming conventions.
