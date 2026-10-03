# Credits

This repository is an integration layer. The model, inference engines,
transport, checkpoints, and research methods were built by other people. The
repository's own work is Apache-2.0; third-party work keeps its own license.
Exact tested revisions are recorded in `UPSTREAM.lock`.

## Core projects

- [Mia's AI Lab GLM-5.3 Flash TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  by MiaAI-Lab and its contributors (Apache-2.0). It supplied the verified
  two-Spark baseline, image, patch stack, EXL3 checkpoint preparation, and
  operating pattern from which this three-machine recipe started. Its own
  [`CREDITS.md`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/main/CREDITS.md)
  and [`NOTICE`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/main/NOTICE)
  are part of this attribution chain.
- [TensorFold](https://github.com/ashhart/TensorFold) by Ash Hart and the
  TensorFold contributors (Apache-2.0; older source also retains its MIT
  notice). TensorFold is the Metal and CUDA inference engine. The patch under
  `patches/tensorfold/` modifies TensorFold source and retains its notices.
- [MCDMA](https://github.com/ashhart/MCDMA) by Ash Hart and the MCDMA
  contributors (Apache-2.0). MCDMA supplies the Mac-to-Spark RDMA driver,
  provider, daemon, mailbox, and validation tools. Patches under
  `patches/mcdma/` modify MCDMA source and retain its notice.
- [oMLX](https://github.com/jundot/omlx) by jundot and the oMLX contributors
  (Apache-2.0). Its MCDMA stage and remote-prefill work informed the protocol,
  safety gates, and cache-handoff design. oMLX remains reference code and is
  not the serving engine in this recipe.

## Models and formats

- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.ai
  (MIT): model architecture, training, tokenizer, and base weights.
- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw),
  a mirror of Brandon M. Music's EXL3 quantization made with ShapleyMcg. The
  checkpoint license requires the attribution reproduced in `NOTICE`.
- [TensorFold/GLM-5.3-Flash-MLX-4bit-MTP](https://huggingface.co/TensorFold/GLM-5.3-Flash-MLX-4bit-MTP),
  formerly published under the Vontra namespace (MIT): the portable MLX
  checkpoint used for the measured prefill/decode handoff.
- [GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  by IncoAI (CC BY-NC-ND 4.0). It was measured only as an optional proof-of-
  concept baseline; the accepted recipe uses the checkpoint's MTP head.
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp and
  contributors (MIT): the EXL3 format and related kernels used upstream.

## Research and control tools

- [OpenResearch](https://github.com/alphaXIV/OpenResearch) by alphaXIV
  (MIT): used as an external experiment controller and record viewer. Its code
  is not bundled here.
- [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy
  (MIT): inspired the fixed evaluator, one-change-per-trial, and explicit
  keep/reject method. Its training code is not copied or run here.
- [Inference Engineering](https://www.baseten.co/inference-engineering/llms.txt)
  by Philip Kiely and Baseten: measurement and serving methodology.

## Upstream runtime stack

The pinned Mia/TensorFold runtime also builds on the work it credits:

- [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0) for the
  upstream RoCE transport work;
- [glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark)
  by Jay Leaton (Apache-2.0) for upstream GLM serving patches;
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
  (Apache-2.0), [z-lab/dflash](https://github.com/z-lab/dflash) (MIT), and
  [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0) for model and API
  reference behavior;
- [MLX](https://github.com/ml-explore/mlx) and
  [MLX-LM](https://github.com/ml-explore/mlx-lm) by Apple and contributors;
- [PyTorch](https://github.com/pytorch/pytorch),
  [Triton](https://github.com/triton-lang/triton),
  [NCCL](https://github.com/NVIDIA/nccl), and
  [rdma-core](https://github.com/linux-rdma/rdma-core);
- [Hugging Face Hub](https://github.com/huggingface/huggingface_hub),
  [safetensors](https://github.com/huggingface/safetensors), Docker, NVIDIA's
  PyTorch container, and the NVIDIA Container Toolkit.

Please read the linked upstream notices before redistributing their code or
images. If a credit is missing or inaccurate, open an issue so it can be fixed.
