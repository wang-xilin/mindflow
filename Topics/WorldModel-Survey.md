---
title: "World Model 文献调研"
tags: [world-model, RL, VLA, diffusion-policy, task-planning, 3D-representation, cross-embodiment]
date_updated: 2026-03-30
year_range: "2023-2026"
papers_analyzed: 20
---
## Overview

World model 是 AI 系统中用于理解环境动态并预测未来状态的内部模拟器。这一概念源于 Kenneth Craik (1943) 的 mental model 理论，在深度学习时代获得了全新的技术内涵。如 [[2411-WorldModelSurvey]] 所总结，world model 的核心功能可分为两大类：**implicit representation**（将外部现实转化为内部表征以支持决策）和 **future prediction**（生成环境演化的仿真）。

近三年（2023-2026）world model 领域经历了爆发式增长，呈现出多线并进的格局。**在 reinforcement learning 中**，[[2405-DIAMOND]] 等工作证明 diffusion-based world model 能以更高 visual fidelity 训练出超越人类的 agent。**在游戏/交互模拟中**，[[2402-Genie]] 和 [[2408-GameNGen]] 开创了从无标注视频生成可交互环境的新范式，DeepMind 的 Genie 系列已从 2D platformer 演进到实时 3D world generation（Genie 3, 2025）。**在自动驾驶中**，[[2405-Vista]] 和 [[2405-OccSora]] 分别从 video prediction 和 4D occupancy 两个表征空间推进了 driving world model 的能力边界。**在机器人领域**，[[2602-DreamZero]]、[[2504-UWM]]、[[2512-Motus]] 等工作将 world model 与 action prediction 统一，催生了 World Action Model (WAM) 这一新范式（详见 [[Topics/WorldActionModel-Survey]]）。**在 LLM 推理中**，[[2305-RAP]] 率先将 LLM 本身视为 world model，用 MCTS 进行 deliberate reasoning。**在自监督学习路线上**，Meta 的 [[2506-VJEPA2]] 通过纯 self-supervised pretraining 在 representation space 实现了理解、预测和规划的统一。

研究活跃度方面，NVIDIA、Google DeepMind、Meta FAIR、OpenAI、Tsinghua 等顶级机构全面布局。2024-2025 年至少发表了 5 篇系统性综述（ACM CSUR, arXiv），GitHub 上多个 awesome list 持续更新。World model 正在从一个分散的研究方向汇聚为 physical AI 的核心基础设施。

## 技术路线

### 路线 1：生成式视频世界模型 (Generative Video World Models)

**核心思路**：以 video generation 为核心，将未来视觉观测的生成作为环境动态建模的基本手段。分为 action-conditioned（可控）和 unconditional（自由生成）两种模式。

**代表论文**：
- [[2501-Cosmos|Cosmos]]：NVIDIA 的 World Foundation Model 平台，2000 万小时视频训练，提供 diffusion/AR 两种 backbone，开源 video tokenizer 和预训练模型，成为下游工作的基础设施。
- [[2405-Vista|Vista]]：基于 Stable Video Diffusion 的自动驾驶 world model，创新 dynamics enhancement loss + structure preservation loss，支持 4 种 action modality 的统一控制，FID 6.9 on nuScenes。
- [[2406-IRASim|IRASim]]：frame-level action conditioning 实现精确的 action-video 对齐，policy evaluation 与 ground-truth simulator 相关度达 0.99。

**优势**：视觉表达力强，可直接生成人类可理解的预测；可利用海量 internet-scale video data 进行预训练。
**劣势**：pixel-space generation 计算开销大；长时序一致性难以保证；缺乏显式 3D 几何信息。

### 路线 2：联合世界-动作模型 (Joint World-Action Models / WAM)

**核心思路**：在同一 generative model 中联合建模 video prediction 和 action prediction，使 world understanding 与 motor control 深度耦合。"World model 即 policy"。

**代表论文**：
- [[2602-DreamZero|DreamZero]]：14B autoregressive diffusion transformer，定义了 WAM 概念，seen tasks 62.2%（baseline 27.4%），unseen tasks 39.5%（baseline <1%）。
- [[2504-UWM|UWM]]：decoupled diffusion timesteps 统一 video + action diffusion，单模型支持 4 种推理模式。RSS 2025。
- [[2512-Motus|Motus]]：Mixture-of-Transformers + optical flow latent action，RoboTwin 87.02%。
- [[2505-DreamGen|DreamGen]]：4 阶段 pipeline，video world model 作为 synthetic data generator，从单任务泛化到 22 种新行为。
- [[2602-WorldVLALoop|World-VLA-Loop]]：world model 作为 RL post-training 环境，与 VLA 闭环迭代优化。

**优势**：泛化能力远超纯 action prediction（VLA）；自然利用 action-free video data。
**劣势**：计算开销极大（DreamZero 需 2×GB200 实现 7Hz）；long-horizon 能力有限。

> 该路线的详细分析见 [[Topics/WorldActionModel-Survey]]。

### 路线 3：交互式环境生成 (Interactive Environment Generation)

**核心思路**：从数据中学习生成可交互的虚拟环境，用户可通过 action 实时与生成内容交互。目标是用 neural model 替代传统 game engine 或 simulator。

**代表论文**：
- [[2402-Genie|Genie]]：11B 参数 foundation world model，核心创新是通过 VQ-VAE 从无标注视频中无监督发现 latent action space（8 个 discrete code），支持从文本/图片/草图生成可交互的 2D 世界。ICML 2024。
- [[2408-GameNGen|GameNGen]]：基于 fine-tuned Stable Diffusion 在 DOOM 上实现 20 FPS 实时交互，noise augmentation 技术有效解决 auto-regressive drift。人类评估者区分真实/生成画面准确率仅 58-60%。ICLR 2025。
- **Genie 2/3**（DeepMind, 2024-2025）：从 2D 扩展到 quasi-3D 再到实时 720p 3D world generation，分钟级连贯交互。

**优势**：开辟了"从视频学习可交互世界"的全新范式；Genie 的 unsupervised latent action discovery 不依赖 action label。
**劣势**：分辨率和 context window 有限；计算成本高；生成环境的物理一致性难以保证。

### 路线 4：Latent Dynamics / Model-Based RL

**核心思路**：在低维 latent space 或 state space 建模环境动力学，用于 imagination-based planning 或 policy optimization。经典的 model-based RL 路线。

**代表论文**：
- [[2405-DIAMOND|DIAMOND]]：diffusion-based world model 用于 Atari 100k，证明保留 visual details 对 RL 至关重要。3 步 denoising 实现高效推理，mean HNS 1.46（超越 DreamerV3、IRIS、STORM）。NeurIPS 2024 Spotlight。
- [[2501-RoboticWorldModel|Robotic World Model]]：dual-autoregressive GRU 在低维 state space 建模，MBPO-PPO 实现 zero-shot sim-to-real，inference 仅 1ms/step。
- [[2506-VJEPA2|V-JEPA 2]]：Meta 的 self-supervised approach，在 representation space（非 pixel space）做 prediction 和 planning，62 小时无标签机器人视频即可训练 action-conditioned world model，latent planning 比 pixel-space generation 快 15×。

**优势**：计算高效（latent space 远小于 pixel space）；DIAMOND 证明 diffusion 比 discrete tokenization 更适合保留 RL 关键信息；V-JEPA 2 展示了 self-supervised 路线的数据效率。
**劣势**：latent space 的可解释性有限；从 latent 到 pixel 的 reconstruction 存在信息损失。

### 路线 5：LLM 作为世界模型 (LLM as World Model)

**核心思路**：利用 LLM 在大规模文本预训练中获得的 world knowledge，将 LLM 本身视为 implicit world model，用于 reasoning 和 planning。

**代表论文**：
- [[2305-RAP|RAP]]：将 LLM 同时用作 world model（state transition prediction）和 reasoning agent，结合 MCTS 进行 deliberate reasoning。LLaMA-33B + RAP 在 Blocksworld 上超越 GPT-4 + CoT。EMNLP 2023。
- [[2602-GigaBrain|GigaBrain]]：VLA 结合 world model-based RL，将语言模型的 reasoning 能力与物理世界的 dynamics modeling 结合。

**优势**：不需要额外训练独立的 dynamics model；框架通用，可应用于多种推理任务；inference-time scaling 的先驱思路。
**劣势**：LLM 的 world model 能力依赖 in-context learning，对复杂物理环境可能不够准确；MCTS 推理开销大。

### 路线 6：3D/4D 结构化世界模型 (Structured 3D/4D World Models)

**核心思路**：在显式 3D 或 4D 空间中建模世界状态，保留几何结构信息，用于需要空间理解的任务（如自动驾驶、导航）。

**代表论文**：
- [[2405-OccSora|OccSora]]：diffusion-based 4D occupancy 生成，FID 8.348 优于 image-based 方法，支持 trajectory-conditioned 16 秒场景生成。ICLR 2025。
- [[2602-GTA|GTA]]：显式 world representation 用于 VLN，证明结构化空间表征对导航推理的重要性。

**优势**：保留了 3D 几何信息，对需要空间推理的任务更友好；不受 2D video generation 的视角限制。
**劣势**：3D reconstruction 质量是核心瓶颈（OccSora mIoU 仅 27.4%）；训练数据获取困难。

## 发展时间线

```
2023.05  RAP — LLM as World Model + MCTS (EMNLP 2023)
         └─ 先驱：将 LLM 视为 world model 用于 deliberate reasoning

2024.02  Genie — Generative Interactive Environments (ICML 2024, DeepMind)
         └─ 开创：从无标注视频生成可交互世界，unsupervised latent action discovery

2024.05  DIAMOND — Diffusion World Model for Atari (NeurIPS 2024, Geneva/Edinburgh)
         └─ 突破：证明 visual details matter for RL，diffusion WM SOTA on Atari 100k

2024.05  Vista — Driving World Model (NeurIPS 2024, HKUST/OpenDriveLab)
         └─ 推进：高保真 + 多模态可控的驾驶 world model

2024.05  OccSora — 4D Occupancy World Simulator (ICLR 2025, Beihang/Tsinghua)
         └─ 新方向：从 2D video 到 4D occupancy 的表征升级

2024.06  IRASim — Action-Conditioned Video Prediction (ICCV 2025, ByteDance)
         └─ 奠基：frame-level action conditioning 实现精确 video-action alignment

2024.08  GameNGen — Neural Game Engine (ICLR 2025, Google)
         └─ 里程碑：首个 neural model 实时交互 game engine (DOOM @ 20FPS)

2024.11  World Model Survey — ACM CSUR 2025 (Tsinghua)
         └─ 综述：implicit representation vs. future prediction 双功能分类体系

2025.01  Cosmos — World Foundation Model Platform (NVIDIA)
         └─ 基础设施：开源 video tokenizer + pre-trained WFM

2025.01  Robotic World Model — State-Space WM (ETH Zurich)
         └─ 经典路线：model-based RL + zero-shot sim-to-real

2025.04  UWM — Unified Video+Action Diffusion (RSS 2025, UW+TRI)
         └─ 统一：decoupled timesteps 联合 world model 与 policy

2025.05  DreamGen — Video WM as Data Generator (CoRL 2025, NVIDIA)
         └─ 应用：world model 驱动的 data flywheel

2025.06  V-JEPA 2 — Self-Supervised World Model (Meta FAIR)
         └─ 新范式：representation space prediction + 62h zero-shot robotics

2025.12  Motus — Unified Latent Action WM (Tsinghua)
         └─ 推进：MoT 架构 + latent action，5 种建模模式统一

2026.02  DreamZero — World Action Model 定义 (NVIDIA)
         └─ 定义时刻：正式提出 WAM 概念，14B 模型 2x 超越 VLA

2026.02  World-VLA-Loop — WM + VLA 闭环 (NUS)
         └─ 闭环：world model 与 VLA policy 迭代互利

2026.02  GigaBrain — VLA + WM-based RL (-)
         └─ 融合：language reasoning + world model RL
```

**关键趋势**：
1. **从 pixel 到 latent 到 3D**：world model 的表征空间不断丰富——pixel-level video (Vista, GameNGen) → latent dynamics (DIAMOND, V-JEPA 2) → 4D occupancy (OccSora) → unified architecture (UWM, DreamZero)
2. **从辅助到核心**：world model 从 policy evaluation 工具 → data generator → 直接作为 policy（WAM）
3. **从 domain-specific 到 foundation**：从 task-specific model → platform-level foundation model (Cosmos, Genie series)
4. **Self-supervised 路线崛起**：V-JEPA 2 证明纯 self-supervised pretraining 可以同时赋能理解、预测和规划，挑战了 language-supervised 的主流路线

## Paper Comparison

| Paper | Venue | 技术路线 | 核心方法 | 关键结果 | 局限性 |
|:------|:-----|:---------|:---------|:---------|:-------|
| [[2305-RAP\|RAP]] | EMNLP 2023 | LLM as WM | LLM + MCTS deliberate reasoning | Blocksworld 超越 GPT-4+CoT | MCTS 推理开销大；reward 需手动设计 |
| [[2402-Genie\|Genie]] | ICML 2024 | 交互式环境 | 11B ST-Transformer + unsupervised LAM | FVD 40.1；从无标注视频学交互 | 1 FPS；16 帧 context；160×90 分辨率 |
| [[2405-DIAMOND\|DIAMOND]] | NeurIPS 2024 | Latent Dynamics/RL | EDM diffusion WM, 3-step denoising | Atari 100k HNS 1.46 SOTA | 仅 discrete action space；有限 memory |
| [[2405-Vista\|Vista]] | NeurIPS 2024 | 视频 WM (驾驶) | SVD + dynamics/structure loss, 4 action modalities | FID 6.9 nuScenes; 576×1024@10Hz | 计算开销大；跨域评估有限 |
| [[2405-OccSora\|OccSora]] | ICLR 2025 | 3D/4D WM | DiT on 4D occupancy tokens | FID 8.348; 16s 一致生成 | 重建 mIoU 仅 27.4%；数据规模有限 |
| [[2406-IRASim\|IRASim]] | ICCV 2025 | 视频 WM (机器人) | DiT + frame-level action conditioning | Policy eval r=0.99 | 非实时生成 |
| [[2408-GameNGen\|GameNGen]] | ICLR 2025 | 交互式环境 | Fine-tuned SD + noise augmentation | DOOM 20 FPS; human eval ~random | 3s memory；仅模拟已有游戏 |
| [[2411-WorldModelSurvey\|WM Survey]] | CSUR 2025 | 综述 | Implicit rep vs. future prediction 分类 | 覆盖 game/driving/robotics/social | 边界模糊；技术细节偏浅 |
| [[2501-Cosmos\|Cosmos]] | arXiv 2025 | 视频 WM 平台 | Diffusion/AR WFM + video tokenizer | PSNR 35.85; 2-12x faster tokenizer | 缺 robotics 定量评估；10K H100 |
| [[2501-RoboticWorldModel\|Robotic WM]] | arXiv 2025 | State-space WM | Dual-AR GRU + MBPO-PPO | Zero-shot sim-to-real; 1ms/step | 仅 proprioceptive state |
| [[2504-UWM\|UWM]] | RSS 2025 | Joint WAM | Decoupled diffusion timesteps | Real robot +20% over DP | 主要收益来自 pretraining |
| [[2505-DreamGen\|DreamGen]] | CoRL 2025 | WM-driven policy | Video WM → IDM → co-train | 22 novel behaviors; cross-embodiment | 1500 L40 GPU; IDM error propagation |
| [[2506-VJEPA2\|V-JEPA 2]] | arXiv 2025 | Latent Dynamics (SSL) | Mask-denoising ViT 1B; latent planning | SSv2 77.3; zero-shot Franka 65-75% | 仅简单任务验证；22M 视频训练 |
| [[2512-Motus\|Motus]] | arXiv 2025 | Joint WAM | MoT + optical flow latent action | RoboTwin 87.02%; +45% over π0.5 | 依赖 optical flow; 18K GPU hours |
| [[2602-DreamZero\|DreamZero]] | arXiv 2026 | Joint WAM | 14B AR diffusion, joint video-action | Seen 62.2% (2.3x); unseen 39.5% | 2×GB200 for 7Hz; 6.6s context |
| [[2602-WorldVLALoop\|World-VLA-Loop]] | arXiv 2026 | WM-driven policy | State-aware WM + co-evolving loop | LIBERO +12.7%; real +23.4% | Short-horizon ~20s |
| [[2602-GigaBrain\|GigaBrain]] | arXiv 2026 | WM + RL | VLA + world model-based RL | — | — |

## Datasets & Benchmarks

| Dataset/Benchmark | 领域 | 规模 | 评估指标 | 特点 |
|:------------------|:-----|:-----|:---------|:-----|
| Atari 100k | RL/Game | 26 games, 100K env steps | Human Normalized Score (HNS) | 经典 world model RL benchmark |
| nuScenes | 自动驾驶 | 1000 scenes, 1.4M frames | FID, FVD, mIoU | 标准驾驶数据集 |
| LIBERO | 机器人 | 130 tasks, 5 suites | Success Rate | 多任务桌面操作 |
| RoboTwin | 机器人 | 多 embodiment | Success Rate | 双臂操作 benchmark |
| Push-T | 机器人 | 推块任务 | IoU | 简单 planar manipulation |
| Something-Something v2 | 视频理解 | 220K videos | Top-1 Accuracy | 动作/物体交互理解 |
| Epic-Kitchens-100 | 视频预测 | 100 hours | Recall@5 | 长时序厨房活动 |
| WorldModelBench | 通用 | 多 domain | Physics adherence, instruction following | NeurIPS 2025，首个系统化 WM 评测 |
| Blocksworld | 推理 | 逻辑规划 | Success Rate | 经典 planning domain |

## Key Takeaways

1. **World model 正在从分散研究汇聚为统一范式**。从 model-based RL (DIAMOND) 到 video generation (Cosmos, Vista) 到 interactive simulation (Genie, GameNGen) 到 robotics (DreamZero, UWM) 到 LLM reasoning (RAP)，所有方向都在向"构建环境内部模拟器"这一共同目标收敛。[[2411-WorldModelSurvey]] 的 implicit representation vs. future prediction 分类体系提供了统一理解框架。

2. **Diffusion model 成为 world modeling 的主导技术**。无论是 game simulation (GameNGen)、RL training (DIAMOND)、autonomous driving (Vista, OccSora) 还是 robotics (DreamZero, UWM)，diffusion 都是核心 generation backbone。DIAMOND 证明 diffusion 比 discrete tokenization 更适合保留 RL 关键信息，这一 insight 具有广泛适用性。

3. **Self-supervised 路线是值得关注的替代范式**。V-JEPA 2 证明纯 self-supervised pretraining（无语言监督）可以同时赋能理解、预测和规划，仅 62 小时无标签机器人视频即可实现 zero-shot manipulation。在 representation space 而非 pixel space 做 planning 带来 15× 速度提升，对 real-time robot control 有重要实践意义。**建议加入 DomainMaps**：world-model 作为新 domain。

4. **Unsupervised action discovery 打开了数据飞轮**。Genie 的 latent action model 证明可以从无 action label 的视频中发现可控的 action space，UWM/DreamGen/Motus 进一步验证了利用 action-free video data 的多种路径。这是 world model 相比纯 VLA 的核心数据优势。

5. **计算成本是全领域共性瓶颈**。从 DreamZero 的 2×GB200 到 Cosmos 的 10K H100，从 DIAMOND 的 1 GPU-year 到 V-JEPA 2 的 22M 视频训练，world model 的计算需求远超常规 supervised learning。Model distillation、efficient architecture、latent-space planning 是缓解这一瓶颈的三个主要方向。

## Open Problems

1. **Long-horizon reasoning 缺失**：当前 world model 本质上是 short-horizon reactive model——DreamZero 6.6s context，World-VLA-Loop ~20s，GameNGen 3s memory。如何引入 hierarchical planning、explicit memory 或 multi-scale temporal modeling 来支持分钟级 long-horizon 任务，是核心未解难题。

2. **物理一致性不足**：即使 Sora 级别的 video model 也"难以一致性地复现正确物理定律"（[[2411-WorldModelSurvey]]）。OccSora 的 mIoU 仅 27.4%，移动物体细节不一致。如何在 world model 中注入 causal reasoning 和 physics prior，是从"看起来像"到"真正理解"的关键跨越。

3. **统一评测框架缺失**：现有 benchmark 分散在各 domain（Atari 100k for RL, nuScenes for driving, LIBERO for robotics），缺乏跨领域的统一 world model 评测标准。WorldModelBench (NeurIPS 2025) 是初步尝试，但远未形成共识。需要评估 physical consistency、temporal coherence、action-video alignment、causal reasoning 等维度。

4. **Scaling laws 未知**：DreamGen 展示了 log-linear scaling 趋势，但 world model 是否存在类似 LLM 的 power-law scaling？Optimal compute allocation 在 video vs. action vs. representation 之间如何分配？World model 的 emergent ability 在什么 scale 出现？均无明确结论。

5. **Real-time inference gap**：联合 video+action generation 的计算开销远超纯 action prediction。DreamZero 的 7Hz on 2×GB200 vs. VLA 的 20Hz+ on consumer GPU。V-JEPA 2 的 latent planning 提供了 15× 加速，但 robot control 需要的 real-time 性能仍是挑战。

6. **Implicit vs. Explicit representation 的最优平衡**：pixel-space video (Vista, GameNGen) 表达力强但计算昂贵，latent space (DIAMOND, V-JEPA 2) 高效但信息有损，3D/4D (OccSora) 结构化但重建困难。如何在不同表征空间之间找到最优 trade-off，或设计能自适应切换的 multi-scale representation，是架构层面的开放问题。

7. **Cross-domain transfer**：能否训练一个跨 game/driving/robotics/reasoning 的 universal world model？Cosmos 和 Genie 系列在 foundation model 方向的探索初见端倪，但真正的 cross-domain generalization 尚未实现。

## 调研日志

- **调研日期**: 2026-03-30
- **论文统计**: vault 已有 12 篇 + 新 digest 8 篇 + 跳过 0 篇 + 失败 0 篇
- **未能获取**: 无
