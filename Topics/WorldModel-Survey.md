---
title: World Model — A 2024–2026 Survey
description: 从 Ha & Schmidhuber 的 MBRL world model 到 Sora/Genie 浪潮后的 video world foundation model，系统梳理当前 implicit latent WM、pixel-space video WM、3D/4D WM、unified VLA+WM、WM-as-RL-simulator 五条技术路线，聚焦 Embodied AI 场景的关键证据与 open problems
tags: [world-model, VLA, RL]
date_updated: 2026-04-23
year_range: 2024-2026
---

## Overview

**一句话定位**：**World Model (WM)** 指"用感知构建内部表示、能基于 action 条件模拟未来动态、并服务于决策/规划"的 model 或 framework（[[2411-WorldModelSurvey|Ding et al. 2024/CSUR]] 的二元定义 + [[2604-OpenWorldLib|OpenWorldLib 2026]] 对 core-objective 的可操作收敛）。在 2024–2026 这轮叙事里，它不再等同于经典 Ha & Schmidhuber 的 latent MBRL，而是成为连接 video generation、VLA 政策学习、自动驾驶仿真、3D 场景生成的**枢纽概念**。

**领域活跃度**（基于 vault 本地可追溯 ~20 篇 rating ≥ 2 的代表工作）：

- **时间线**：2024-02 [[2402-Genie|Genie]] 打开 unsupervised frame-level-controllable video 象限 → 2024-05 [[2405-DIAMOND|DIAMOND]] 证 pixel-diffusion MBRL → 2024-08 [[2408-GameNGen|GameNGen]] 跑通 20 FPS 实时 DOOM → 2025-01 [[2501-Cosmos|Cosmos]] 给出 100M clip 级 World Foundation Model platform → 2025-04 [[2504-UWM|UWM]] 用 timestep-as-mask 统一 policy/dynamics/IDM/video → 2025-06 [[2506-VJEPA2|V-JEPA 2]] 在 latent 空间打通 understanding/prediction/planning → 2025-05 [[2505-DreamGen|DreamGen]] 明确把 video WM 定位为 offline data engine → 2025-12/2026-02 [[2512-Motus|Motus]] / [[2602-DreamZero|DreamZero]] 把 video diffusion 做成 VLA backbone → 2026-02 [[2602-WorldVLALoop|World-VLA-Loop]] / [[2602-GigaBrain05M|GigaBrain-0.5M*]] 把 WM 嵌入 VLA RL 闭环 → 2026-04 [[2604-HYWorld2|HY-World 2.0]] / [[2604-OpenWorldLib|OpenWorldLib]] 标志开源生态成型。
- **参与格局**：工业主力 NVIDIA（Cosmos/CosmosReason/DreamGen/DreamZero，GEAR team）、Google DeepMind（Genie 系）、Meta FAIR（V-JEPA 系）；中国工业主力 Tencent Hunyuan（HY-World）、GigaAI（GigaBrain）、AgiBot（GenieReasoner）；驾驶主力 OpenDriveLab/Wayve 系（Vista、GAIA-1、Drive-WM）；学术 ETH（RWM legged）、HKUST/ByteDance（IRASim）、UW/TRI（UWM）。Open-source 代码/权重密度在 2025 下半年后显著上升（Cosmos、V-JEPA 2、DreamGen/GR00T-Dreams、HY-World 2.0、UWM 均 open-weight）。
- **整体趋势**（2024 → 2026）：(1) **video generation ↔ world model 的边界被激活后又被 OpenWorldLib 重新收紧**，text-to-video（Sora）被显式排除；(2) **WM 的目标从"逼真视频"迁移到"可控、可作 agent backend 的动力学"**——DreamGen Bench / WorldSimBench / Physics-IQ 的出现使 video community 第一次拿到"对 robotics 有用"的 proxy metric；(3) **WM × VLA 的五种耦合方式**（offline data engine → inference-time latent conditioning → joint model → RL simulator → evaluator）全部被实证验证，成为 Embodied AI 上 WM 研究的主战场；(4) **latent vs pixel 的路线之争**首次进入可比较阶段（V-JEPA 2 给出 15× 计算优势 + success rate 反超 Cosmos 的 zero-shot 对比）。

---

## Problem & Motivation

**核心问题**：给定高维感知流与可选 action，学一个能"内化世界动态"的模型，使 agent 能在 imagination 中进行决策、评估或数据生成，而不必每次都真机交互。

**为什么重要**：

1. **数据/安全/成本**：真实 embodied 数据采集成本极高（[[2501-Cosmos|Cosmos]] 提到 exploratory action 会损坏硬件）；[[2501-RoboticWorldModel|RWM]] 显示 6M state transition pretraining + imagination-PPO 可换 250M transitions 的 model-free 水平。
2. **Generalization 瓶颈**：[[2602-DreamZero|DreamZero]] 的核心诊断——VLA 从 VLM 继承了 *semantic* prior 但缺 *spatiotemporal/dynamics* prior；"untie shoelaces" 这类 motion-level novel task，VLA 在 baseline 零成功，DreamZero 的 video-prior-first 路线直接拉到 39.5%。
3. **Sim-to-real 与合成数据**：[[2505-DreamGen|DreamGen]] 从单一 pick-and-place teleop 数据 + neural trajectory 解锁 22 个新动词 / 10 个新环境，把 "video model as data engine" 明确成可 scale 的新 sub-paradigm。
4. **VLA RL 的仿真器缺口**：真实环境做 RL prohibitive，手工 digital twin 缺 photorealism，3D 重建泛化差；video WM 是当前最有希望的"通用可交互仿真器"，但 [[2602-WorldVLALoop|World-VLA-Loop]] 把它的硬伤（action-following 差、对错 action 仍 hallucinate 成功）摆到桌面上。

**为什么适合现在做**：

- **技术栈成熟**：大规模 video diffusion（Sora/Wan/Cosmos 级）+ 高保真 latent tokenizer（Cosmos causal wavelet、SDXL VAE）+ flow matching/rectified flow 已经同时可用，工程门槛从 2023 年"学术 toy"级降到"NVIDIA stack 可复用"级。
- **VLA 主流范式已成熟**：π0/GR00T/OpenVLA/π0.5 等 VLA baseline 建立了明确的对比基准，"WM 对 VLA 有没有用"从叙事变成可量化问题（如 [[2602-DreamZero|DreamZero]] 的 62.2% vs 27.4%、[[2512-Motus|Motus]] 的 +43% over π0.5）。
- **开源生态就位**：[[2501-Cosmos|Cosmos]] 放 7 tokenizer + 8 WFM checkpoint、[[2506-VJEPA2|V-JEPA 2]] 放完整 ckpt、[[2505-DreamGen|DreamGen]] 放 GR00T-Dreams、[[2604-HYWorld2|HY-World 2.0]] 全套开源——"stable-diffusion moment for video WM"已经开始。
- **对齐 AGI 议程**：LeCun 对 JEPA 路线的长期倡导（latent prediction as world model）、Ha & Schmidhuber 思路的 Sora-era 再激活，把 WM 作为"通向 AGI 的基础设施"这个叙事重新带回学术主流。

---

## 技术路线对比

下列五条路线按"技术架构 + 使用目标"切分；每条路线的代表工作、核心 insight、优缺点与 open gap 如下。

### 1. Pixel-space Video Diffusion WM（future prediction 主脉）

**核心思路**：直接在 RGB（或其 VAE latent）空间用 diffusion/flow matching 建 $p(o_{t+1:} \mid o_{\le t}, c)$，$c$ 可以是 text / action / trajectory / camera / goal。backbone 迅速从 U-Net 迁向 DiT/MMDiT。

**代表工作与证据**：

- [[2408-GameNGen|GameNGen]]（ICLR 2025）：fine-tune SD 1.4，20 FPS 实时模拟 DOOM，**noise augmentation 解决 auto-regressive drift** 成为此后 AR video WM 的标配 trick；4 步 DDIM ≈ 64 步的工程证据启发 robot WM 不必堆 sampling steps。
- [[2405-DIAMOND|DIAMOND]]（NeurIPS 2024 Spotlight）：EDM-flavored 像素空间 diffusion WM 在 Atari 100k 拿 mean HNS 1.46（agents-trained-in-WM 新 SOTA），**EDM vs DDPM 的 $c_{\text{skip}}$ 稳定性分析**对所有长时序自回归生成（video gen / VLA imagined rollout）都 transferable；并 scale 到 CS:GO Dust II 的 10 Hz neural game engine。
- [[2501-Cosmos|Cosmos]]（NVIDIA, 2025-01）：**20M hour 视频 → 100M clips 的 industrial data curation + causal wavelet tokenizer（DAVIS PSNR +4dB, 12× 加速）+ 7B/14B diffusion & 4B/12B AR 双路线**。在 3D consistency（SfM 成功率 43%→82%）和 camera-conditioned generation 上拿出硬数据；但**物理对齐（Isaac Sim 8 rigid-body scenes）大小模型基本不变**，首次给出 "scale 不能 alone 解决 physics" 的 negative evidence。
- [[2405-Vista|Vista]] / [[2405-OccSora|OccSora]]：自动驾驶 WM 的 video vs occupancy 两端。Vista（NeurIPS 2024）基于 SVD + **latent replacement + dynamics enhancement loss + structure preservation loss** 把 nuScenes FVD 从 GenAD 184 压到 89.4；OccSora 把 DiT 搬到 4D 占据生成，但 512× 压缩导致 bicycle=0、motorcycle=8.7 的 VRU 重建崩塌——driving WM 的"几何-语义精度权衡"还远未解。
- [[2406-IRASim|IRASim]]（ICCV 2025）：**Frame-level AdaLN conditioning**——把 text-to-video 的 video-level embedding 改为 per-frame action embedding modulate spatial block——成为 trajectory-to-video 方向的 reference 设计。与 LIBERO Mujoco 在 policy evaluation 上 Pearson 0.99；但 A100 30 s/16 帧的推理开销决定它短期内无法替代 Mujoco。

**实际效果与优点**：视觉保真度天花板高（GameNGen 人类辨真伪仅 58–60%）；天然吸收 internet video prior；和成熟 video diffusion 工程栈复用。

**缺点与未解 gap**：

- **Action-following 不可靠**：[[2602-WorldVLALoop|World-VLA-Loop]] 展示 [[2501-Cosmos|Cosmos-Predict 2]] 在错 action 下仍 hallucinate 成功——policy 在此类 WM 上做 RL 会 reward-hack。
- **长时序 drift**：[[2408-GameNGen|GameNGen]] 的 3 秒 context + frame-stacking、[[2405-DIAMOND|DIAMOND]] 的 memory bottleneck、[[2602-WorldVLALoop|World-VLA-Loop]] 主动放弃 LIBERO-Long——>200 帧后视觉/几何普遍漂移。
- **物理对齐不随 scale 解决**（Cosmos Tab. 20）：需 data curation 或 hybrid physics inductive bias，尚无公认方案。
- **推理成本高**：典型 14B DiT naive 5.7 s/chunk（DreamZero 的 baseline），即使 38× 工程栈加速后仍需 2×GB200 才能 7 Hz 闭环。

### 2. Latent-space / JEPA-style WM（implicit representation）

**核心思路**：不重建像素，只在 representation 空间做 mask-denoising / next-state prediction，让 predictor 学 "latent dynamics"，下游用 CEM / MPC 做 planning。

**代表工作**：

- [[2506-VJEPA2|V-JEPA 2]]（FAIR, 2025-06）：1M+ 小时视频 mask-denoising 预训练 1B 参数 encoder → 冻结 + 62 小时 unlabeled Droid 视频训 300M 参数 action-conditioned predictor → CEM 在 latent 上 receding-horizon planning。在 Franka pick-and-place 上 zero-shot 65–80% vs Octo 0–15%；**关键对比：V-JEPA 2-AC 16 s/action vs Cosmos 4 min/action 且 success rate 反超**。
- [[2501-RoboticWorldModel|RWM]]（ETH, NeurIPS 2025 Workshop Outstanding Paper）：GRU + 多步 autoregressive 训练 + privileged head（contact 等）学 legged robot dynamics；**architecture 不是关键，autoregressive training 才是**——ablation 显示 RSSM 用同样训练 scheme ≈ RWM，Transformer 因多步梯度显存爆炸不 work。在 ANYmal D / Unitree G1 上 zero-shot 硬件部署，reward 打平 250M-step model-free PPO 但只用 6M transitions。

**优点**：

- **计算高效**：latent 无需像素重建，V-JEPA 2 对 Cosmos 的 15× 推理优势是 actionable 工程信号。
- **数据效率极高**：V-JEPA 2-AC 62 小时无标签视频即可 cross-lab 部署；RWM 6M transitions 就撑起 PPO。
- **与 MPC/CEM 天然兼容**：latent energy minimization 就是老的 sampling-based 控制范式。

**缺点**：

- **像素生成能力弱**：latent WM 不能直接做"数据 engine"或"可视化调试"，在 data flywheel 场景不如 pixel WM。
- **Goal specification 受限**：V-JEPA 2-AC 靠 goal image；换成 language goal 仍是 open problem。
- **Cross-embodiment 验证薄**：V-JEPA 2 只在 Franka 同平台跨 lab 验证，"cross-embodiment" 措辞偏乐观；Camera position 仍需 manual tune。
- **Latent 不可解释**：在 RL 里 exploit latent 的 drift 难以诊断。

### 3. 3D / 4D Generative WM（空间侧）

**核心思路**：把 WM 绑到显式 3D 表示（occupancy grid、3DGS、point cloud）上，用 diffusion 在 4D 体素/pointcloud latent 空间做未来生成或 navigable scene 合成。

**代表工作**：

- [[2405-OccSora|OccSora]]：nuScenes 上 DiT + 4D VQVAE 生成 16 s 驾驶 occupancy video，trajectory-conditioned；是 occupancy WM 从 autoregressive（OccWorld）迁向 diffusion 的代表，但小物体（VRU）重建崩塌与 trajectory 控制有效性存疑（去 trajectory FID 仅 8→17.5 vs 去 timestep 是 10×恶化）。
- [[2604-HYWorld2|HY-World 2.0]]（Tencent Hunyuan, 2026-04）：四阶段 pipeline **panorama → WorldNav trajectory → WorldStereo 2.0 keyframe-latent VDM → WorldMirror 2.0 feed-forward reconstruction → 3DGS**，端到端 712 s 生成一张可交互 navigable 3D 场景；WorldMirror 2.0 在 7-Scenes / NRGBD / DTU 全面超 VGGT / π³。**核心 insight 是 keyframe-latent VDM**（放弃时空压缩的 Video-VAE，改 spatial-only + 稀疏 keyframe）。但严格地说，它是 navigable 3D scene generator 而非 world model（无 dynamics、无 action-conditioning）。
- [[2604-GenWorldRenderer|Generative World Renderer]]（Alaya Studio, 2026-04）：ReShade + RenderDoc 从 AAA 游戏（Cyberpunk 2077、Black Myth: Wukong）"非侵入式"截取 4M 帧 G-buffer，fine-tune Cosmos-DiffusionRenderer，metallic RMSE –55%。把 "AAA 游戏 = photorealistic supervision 工厂" 变可行。

**优点**：显式 3D 可验证几何；直接对接 CG 渲染 / 物理引擎；cross-scene reuse 便利。

**缺点**：

- **Temporal dynamics / action 缺失**（HY-World 2.0、GenWorldRenderer）：场景冻结后只能 camera control，无事件级 intervention，本质不是"world model"而是"scene generator"。
- **数据稀缺**：占据序列、G-buffer 都依赖模拟器/游戏侧抽取，实拍数据几乎不可得。
- **精度-压缩权衡**：OccSora 的 VRU 崩塌是典型代价。

### 4. Unified Video-Action / VLA+WM Joint Models

**核心思路**：把 VLA（policy）、forward dynamics（WM）、inverse dynamics、video generation 统一进一个模型的若干"推理 mode"，通过 timestep / mask 切换。

**代表工作**：

- [[2504-UWM|UWM]]（RSS 2025, UW & TRI）：**"diffusion timestep ≡ soft mask"**——给 action 和 future obs 独立采样 timestep，推理时固定 $t_a, t_{o'} \in \{0, T\}$ 切换 policy / forward dynamics / inverse dynamics / video prediction 四个条件分布。在 DROID 2K 预训练 + 5 个 Franka 任务 fine-tune 全面超 DP/PAD/GR1，**GR1 共训 3/5 降点而 UWM 5/5 都涨**——是"video 作为预训练信号"的最干净架构方案。规模偏小（180M params, 2K trajectories）、forward dynamics 质量仍有 artifact。
- [[2512-Motus|Motus]]（Tsinghua, 2025-12）：**Mixture-of-Transformers + Tri-modal Joint Attention + UniDiffuser-style scheduler**，把 5-mode（VLA/WM/IDM/VGM/Joint）真正跑通。Optical-flow latent action via DC-AE 对齐 14-dim 典型机器人 action 空间。RoboTwin 2.0 randomized +43% over π0.5 / +14% over X-VLA；real-world AC-One coffee/grinder/towel 任务从 0/8% → 62/92%。但评测全是 Aloha 类平台，"cross-embodiment"卖点未 directly 验证。
- [[2602-DreamZero|DreamZero]]（NVIDIA GEAR, 2026-02）：14B **World Action Model** 从 Wan2.1-I2V-14B 初始化，joint 预测 video + action；**38× 工程加速 + DreamZero-Flash（decoupled video/action noise schedule）** 做到 7 Hz 闭环。AgiBot G1 unseen-env+unseen-object 上 62.2% vs best pretrained VLA 27.4%（>2×）；**20 分钟人类 egocentric video 或 robot video 即可 cross-embodiment transfer +16pp**，为"用人类视频喂机器人 foundation model"给出早期证据；同 5B VLA 在 diverse data 上 0%，14B VLA 仍 0%，而 WAM 5B→14B 从 21%→50%，**显示了 VLA 未显现的 model scaling 信号**。
- [[2512-GenieReasoner|GenieReasoner]]（AgiBot, 2025-12）：**FACT (Flow-matching Action Tokenizer)**——VQ-encoder 把动作压成离散 code，flow-matching decoder 重建高保真连续轨迹，解决 "discrete 精度差 / continuous 梯度冲突" 的 reasoning-precision trade-off。ERIQ 82.72% 超同量级 baseline。与 VLA+WM 的交叉点在于"统一 reasoning + action in one autoregressive AR 框架"。

**优点**：参数共享 / 部署简化；**video prior 显式注入 action learning** 的最自然方式；unified model 在未来潜在 world model rollout 时不需要跨模型对接。

**缺点**：

- **算力门槛极高**：Motus 18 000 GPU-hours + Stage 3 400 GPU-hours、DreamZero 需 2×GB200 才 7 Hz——对学术界基本不可复现。
- **边际收益不一定大**：Motus 的 Joint mode 比 VLA mode 只 +3pp，多数增益来自预训+架构本身而非"五种 mode 协同推理"——**"unify 是否必要"的证据仍弱**。
- **高精度任务不 hold**：DreamZero 承认 sub-cm 精度任务 video token 信息密度不够。

### 5. WM-as-RL-Simulator / WM-Conditioned VLA (Loop 路线)

**核心思路**：用 video WM 替代物理仿真器跑 GRPO / PPO，或把 WM 预测的 future latent + value 作为 VLA policy 的 inference-time condition；policy 与 WM 迭代 co-evolve。

**代表工作**：

- [[2602-WorldVLALoop|World-VLA-Loop]]（Show Lab NUS, 2026-02）：**SANS dataset（success + near-success trajectories） + DiT reward head + co-evolving loop**。核心诊断：video WM 的 action-following 偏差让它对错 action 也生成成功 → policy reward-hack。SANS 强迫模型学 fine-grained 失败模式；reward head 与视频生成联合监督起到**双向 regularizer**（去掉 reward head，visual alignment 也掉 30%）。LIBERO 三 suite +12.7% SR；real-world 13.3% → 36.7% → 50.0% 两轮迭代。主动放弃 LIBERO-Long（AR drift）。
- [[2602-GigaBrain05M|GigaBrain-0.5M*]]（GigaAI, 2026-02）：**RAMP** 把 [[2511-PiStar06|π*₀.₆]] 的 RECAP 从 advantage-only 条件化推广为 (future latent, advantage) 联合条件化；概率上证明 RECAP 是 RAMP 在 z 上 marginalize 的特例。4-stage iterative HIL rollout pipeline 在 Box Packing / Espresso Preparation 长程任务 +30%；WM 联合预测 future state + value 比 only-value 精度更好（MAE 0.062 vs 0.084）；**stochastic attention masking (p=0.2)** 让 WM 条件在推理时可选，直接产出 fast / standard 双模式部署。
- [[2501-RoboticWorldModel|RWM + MBPO-PPO]]：legged 场景证明 "long-horizon PPO + learned model" 可行；是 manipulation 领域尚未复现的 hardware-grounded 基线。

**优点**：把 WM 从"能生成什么视频"转向"能否闭环训 policy"的 actionable metric；**co-evolving loop 给出 reward hacking 的实证 narrative**（World-VLA-Loop），证明单次训练的 WM 必然有 policy 利用的盲区。

**缺点**：

- **仿真器质量瓶颈**：WM 必须真正建模 action-outcome causality 才能撑 RL，而当前 video WM 仍主要靠视觉 prior，action-following 普遍弱。
- **Long-horizon 死穴**：AR video drift 决定 >200 帧任务（真实长 horizon 操作）暂无解。
- **评估样本量小**：WorldVLALoop 30 rollouts、GigaBrain 真机只比 π0.5 中间版——**30% 提升的置信区间很宽**。

**路线间对比小结**：

| 路线 | 代表 | 主要 use case | 推理代价 | 主要 open gap |
|---|---|---|---|---|
| Pixel video diffusion | [[2501-Cosmos\|Cosmos]] / [[2505-DreamGen\|DreamGen]] / [[2406-IRASim\|IRASim]] | Data engine / Evaluator | 14B × 多步 → 秒级 | Action-following / physics / AR drift |
| Latent JEPA | [[2506-VJEPA2\|V-JEPA 2]] / [[2501-RoboticWorldModel\|RWM]] | Agent brain / MPC | 16s → ms 级 | Goal spec / cross-embodiment / 不生成像素 |
| 3D/4D generative | [[2604-HYWorld2\|HY-World 2.0]] / [[2405-OccSora\|OccSora]] | Scene generation / driving sim | 分钟级/场景 | 无 dynamics / 小物体精度 |
| Unified VLA+WM | [[2504-UWM\|UWM]] / [[2512-Motus\|Motus]] / [[2602-DreamZero\|DreamZero]] | VLA policy backbone | 百 ms 级（工程后） | 算力门槛 / unify 必要性 |
| WM-as-RL-simulator | [[2602-WorldVLALoop\|World-VLA-Loop]] / [[2602-GigaBrain05M\|GigaBrain-0.5M*]] | VLA RL post-train | 30 h / 任务级 | Action-following / 样本量 |

---

## Datasets & Benchmarks

**Training Datasets（按规模/用途排序）：**

| Dataset | Scale | 内容 | 主要使用方 |
|---|---|---|---|
| Internal 20 M hours → 100 M clips | 20 M hour | Physical AI-balanced video curation (driving 11% / manipulation 16% / human 10% / first-person 8% / nature 20% / dynamic camera 8% / synthetic 4%) | [[2501-Cosmos\|Cosmos]] WFM pretraining |
| VideoMix22M | 22 M videos | SSv2 / Kinetics / HowTo100M / YT-Temporal-1B / ImageNet | [[2506-VJEPA2\|V-JEPA 2]] |
| OpenDV-YouTube | 最大公开驾驶视频集 | 驾驶第一视角 | [[2405-Vista\|Vista]] / GenAD |
| Internet 200K+ hours | 200K hours | 2D platformer | [[2402-Genie\|Genie]] |
| DROID | 76K trajectories | Franka manipulation (Teleop + GT video) | [[2504-UWM\|UWM]] / [[2506-VJEPA2\|V-JEPA 2]] / [[2602-DreamZero\|DreamZero]] |
| AgiBot World | 728K episodes | Genie-1 humanoid teleop | [[2512-Motus\|Motus]] / [[2512-GenieReasoner\|GenieReasoner]] / [[2602-DreamZero\|DreamZero]] |
| Open X-Embodiment / Bridge / RT-1 | 100K–1M trajectories | Cross-embodiment robot manipulation | 广泛使用 |
| Cosmos-1X | 200 h egocentric humanoid | 具身人形第一视角 | [[2501-Cosmos\|Cosmos]] instruction post-train |
| nuScenes + Occ3D | 1000 scenes / 6 cam | 驾驶 RGB + 4D occupancy | [[2405-Vista\|Vista]] / [[2405-OccSora\|OccSora]] |
| Dust II 5.5 M frames | 87 h | CS:GO human gameplay | [[2405-DIAMOND\|DIAMOND]] neural game engine |
| SANS | ~100 K trajectories | success + near-success + failure robot rollouts | [[2602-WorldVLALoop\|World-VLA-Loop]] |
| AAA-Game G-buffer 4 M frames | 40 h | Cyberpunk 2077 + Black Myth 的 RGB + depth + normal + albedo + metallic + roughness | [[2604-GenWorldRenderer\|Generative World Renderer]] |

**Benchmarks（按 WM 能力维度归类）：**

| 能力 | Benchmark | 代表方法 & 关键分数 |
|---|---|---|
| **Video simulation quality** | WorldSimBench | human preference ↔ action-level consistency 挂钩；[[2411-WorldModelSurvey\|survey Sec 6.3]] cheatsheet |
|  | WorldScore | 3000 camera-spec 场景；controllability/quality/dynamics |
|  | VBench / VBench-2.0 | 通用 T2V；VBench-2.0 强调 intrinsic faithfulness (physics, commonsense) |
|  | T2V-CompBench | compositional T2V，attribute/action/relation binding |
| **Physical / spatial reasoning** | PhysBench | 10 k video-image-text triplet，VLM 物理 gap |
|  | Physics-IQ | 5 个物理域，law adherence vs realism |
|  | T2VPhysBench | 12 条 first-principle laws |
|  | VideoPhy | action-centric prompts，semantic + commonsense |
|  | Cosmos rigid-body (Isaac Sim) | Avg IoU 0.59 @ 9-frame cond；scale-不敏感 |
|  | Cosmos-Reason Intuitive Physics | Arrow of Time / Spatial Puzzle / Object Permanence；SOTA VLM ≈ 随机，[[2503-CosmosReason1\|Cosmos-Reason1]] SFT+RL 到 81.5% |
| **Robot policy via WM** | DreamGen Bench (RoboCasa / GR1) | Instruction Following + Physics Alignment，与 policy success rate 正相关；fine-tuned Cosmos/WAN2.1 进入 60–95 区间 |
|  | LIBERO (Obj / Goal / Spatial / Long) | [[2504-UWM\|UWM]] / [[2602-WorldVLALoop\|World-VLA-Loop]] / [[2512-Motus\|Motus]]（97.6 持平 X-VLA）；Long suite 多数 AR WM 放弃 |
|  | RoboTwin 2.0 randomized | [[2512-Motus\|Motus]] 87.02% +14% over X-VLA / +43% over π0.5 |
|  | RoboChallenge (30 任务, 4 平台) | [[2602-GigaBrain05M\|GigaBrain-0.5]] 51.67% > π0.5 42.67% |
|  | ERIQ (embodied reasoning IQ) | 6052 QA; [[2512-GenieReasoner\|GenieReasoner]]-3B 82.72% |
| **Planning via latent WM** | Franka grasp / reach / P&P (V-JEPA 2-AC) | 65–80% zero-shot vs Cosmos 0–20% |
|  | ANYmal D / Unitree G1 velocity tracking | [[2501-RoboticWorldModel\|RWM]]+MBPO-PPO 0.90 vs 250M-step PPO 0.90 |
|  | Push-T (IRASim ranking planner) | IoU 0.637 → 0.961 at $P{=}1000, K{=}50$ |
| **3D reconstruction / scene gen** | 7-Scenes / NRGBD / DTU / ScanNet | [[2604-HYWorld2\|WorldMirror 2.0]] 全面超 VGGT / π³ |
|  | Tanks-and-Temples / MipNeRF360 | [[2604-HYWorld2\|WorldStereo 2.0-DMD]] F1 43.16 / AUC 60.09，显著超 SEVA / Gen3C / Lyra / FlashWorld |
|  | RealEstate10K 3D consistency | [[2501-Cosmos\|Cosmos]] Sampson 0.355 接近 real video 0.431；camera-cond SfM 43%→82% |
| **Driving WM** | nuScenes val FID/FVD | [[2405-Vista\|Vista]] 6.9/89.4，GenAD 15.4/184 |
|  | nuScenes-Occ3D | [[2405-OccSora\|OccSora]] FID 8.35 / mIoU 27.4 (512× 压缩) vs OccWorld mIoU 65.7 (16× 压缩) |
| **Inverse rendering** | Black Myth held-out / Sintel | [[2604-GenWorldRenderer\|GenWorldRenderer]] metallic RMSE 0.104 vs DR 0.230 (–55%) |
|  | VLM judge (metallic/roughness) | Gemini 3 Pro ↔ 25 CG experts 60–85% agreement |

**Benchmark 使用的 caveat**：

1. **FID/FVD 偏视觉保真度**，对 latent WM (如 V-JEPA 2) 几乎无法公平比较——[[2411-WorldModelSurvey|WMSurvey]] 承认这是现在最没 settle 的问题。
2. **DreamGen Bench** 把 video-model quality 连到 downstream robot policy success，是 2025 以来最有 actionable value 的新 proxy，但评分用 VideoCon-Physics + Qwen-VL-2.5，评分器可靠性本身循环依赖。
3. **LIBERO-Long 被主流 video-AR WM 主动放弃**（[[2602-WorldVLALoop|World-VLA-Loop]] 明示），这是 long-horizon 评测目前的硬伤。
4. **Pearson 0.99 的 policy-evaluation correlation**（[[2406-IRASim|IRASim]] vs LIBERO Mujoco）只有 4 个数据点，不能外推。

---

## Open Problems

1. **Action-following faithfulness**：[[2602-WorldVLALoop|World-VLA-Loop]] 把 "video WM 对错 action 也生成成功" 这个此前被回避的痛点摆上桌面。当前最先进的 Cosmos-Predict 2 / Wan / Genie 系都依赖视觉 prior 而非显式物理动力学——policy 一定能找到 WM 盲区做 reward hacking。SANS 式的 near-success 数据 + reward head 是初步答案，但是否 scale 到 long-horizon / multi-agent / deformable 尚未验证。**真正的开放子问题：什么样的训练信号可以稳定 identify 物理失败模式？**

2. **Physics alignment 不随 scale 解决**：[[2501-Cosmos|Cosmos]] 的 7B vs 14B 在 rigid-body benchmark 上 IoU 基本不变（0.59 vs 0.60）；[[2411-WorldModelSurvey|WMSurvey Sec 6.1]] 总结 Sora 在 gravity/fluid/thermal 上的普遍失败。候选方向：(a) Genesis / PhysGen / physics-informed diffusion 的 hybrid physics；(b) [[2503-CosmosReason1|Cosmos-Reason1]] 式的 RL on intuitive physics MCQ——但第二条目前只涨 VLM-level reasoning，不直接 carry over 到 video generation。

3. **Long-horizon drift**：所有 autoregressive video WM 超过训练 horizon 都退化——[[2408-GameNGen|GameNGen]] 3 秒、[[2405-DIAMOND|DIAMOND]] frame-stacking、[[2602-WorldVLALoop|World-VLA-Loop]] 200 帧、[[2405-OccSora|OccSora]] 离开 32 帧 FID 飙 200+。Explicit compressed memory (learned latent state)、retrieval-based context、LLM-style KV cache + streaming 都是候选，但没有任何一种在 robot-relevant setting 上 demonstrated。

4. **Latent vs pixel 的路线之争**：[[2506-VJEPA2|V-JEPA 2]] 给出 15× 计算优势 + success rate 反超 [[2501-Cosmos|Cosmos]] 的工程事实，但其 "cross-lab" 评测只跨 Franka。[[2602-DreamZero|DreamZero]] 反过来用 14B pixel WAM 达到 62.2% task progress + 5B→14B scaling 信号。**真正的 open question**：long-term 哪一条路径 scale 更好？或两者互补（cloud-side pixel WM 做 data engine / policy evaluator，edge-side latent WM 做 on-device MPC）？这是 [[2411-WorldModelSurvey|WMSurvey Sec 5.5]] 的 cloud/edge 切分未完全 settled 的部分。

5. **Cross-embodiment transfer 真能靠 video 做到吗？**：[[2602-DreamZero|DreamZero]] 的 12 min 人类 egocentric / 20 min YAM robot video → unseen task +16pp 是至今最强信号；[[2512-Motus|Motus]] 的 optical-flow latent action 是工程层 bridge。但两者下游 eval 都主要集中在同 morphology family 内。humanoid 五指手 vs bimanual gripper 级的 morphology gap 尚未被 video WM 路线 attack。

6. **Benchmark metric 的 unresolved confound**：video fidelity (FID/FVD) ↔ physical faithfulness (VBench-2.0, PhysBench) ↔ policy success (DreamGen Bench / LIBERO SR) 三者相关但不等价。[[2505-DreamGen|DreamGen]] Bench 把 IF+PA 连到 downstream RoboCasa 成功率——**但这只在 Cosmos/WAN2.1 fine-tuned 后的区间内正相关**，zero-shot 下很多模型 IF/PA 极低但下游不一定归零。系统化的"哪个 metric 评 WM 公平" 的框架尚未建立。

7. **WM × VLA 耦合方式的 trade-off space**：当前 5 种耦合方式（offline data engine / inference-time condition / joint model / RL simulator / evaluator）都有代表工作，但没有 head-to-head 比较。一个开放的经验问题：**在同等 compute / data 预算下，哪种耦合方式对 sample efficiency 最敏感？**[[2504-UWM|UWM]] 与 [[2505-DreamGen|DreamGen]] / [[2602-DreamZero|DreamZero]] 甚至在 mental model 上有冲突（前者把 WM 降格为 policy 的 auxiliary，后者把它当 policy 主体）。

8. **开源 vs 工业化：可复现性断层**：[[2501-Cosmos|Cosmos]] 10 000 H100 × 3 个月、[[2512-Motus|Motus]] 18 000 GPU-hours、[[2602-DreamZero|DreamZero]] 2×GB200 的 baseline——任何"主脉络" WM 都远超学术实验室预算。后果是 open-source 社区的 Hunyuan/Wan/Cosmos fine-tuning 路线（[[2604-HYWorld2|HY-World 2.0]]、[[2604-OpenWorldLib|OpenWorldLib]]）只能做 integration，不能 push scientific frontier。这是结构性问题，短期无解。

9. **Agent memory 与 World Model 的边界**：[[2604-OpenWorldLib|OpenWorldLib]] 把 long-term memory 写进 world model 定义，但其 Memory 接口 (`record/select/compress/manage`) 留空没有 reference implementation。在 LLM agent 社区（CUA、VLA）已经成熟的 memory 机制（retrieval-based、episodic、semantic）如何与 video WM 的 latent space 交互——是 2026 之后的下一个大空白。

---

## DomainMap 更新建议

当前 `DomainMaps/_index.md` 的 "Cross-Domain Insights" 提到 "World Model 正在成为 VLA 的 foundation"（引 DreamZero / Motus / World-VLA-Loop）并注明应有 `DomainMaps/WorldModel.md`，**但该文件实际不存在**。建议新建 `DomainMaps/WorldModel.md`，以本 Survey 为初始填充，并纳入以下关键 takeaway：

1. **[[2411-WorldModelSurvey|Ding et al. 2024]] 的 implicit/predictive 二分** 和 **[[2411-WorldModelSurvey|WMSurvey Sec 5.5]] 的 cloud-side / edge-side 切分** 并列作为分类基线。
2. **[[2506-VJEPA2|V-JEPA 2]] 的 latent-space MPC 与 [[2501-Cosmos|Cosmos]] 的 video-foundation model 作为 edge 与 cloud 的两个典型 instantiation**。
3. **WM × VLA 耦合的 5 种方式**（data engine / inference-time condition / joint model / RL simulator / evaluator），每一条挂一个代表工作。
4. **Established Knowledge**：AR video WM >200 帧 drift 是通病；EDM c_skip 稳定性对低 NFE 自回归生成必要；Frame-level conditioning（IRASim）优于 video-level；scale 不单独解决 physics。
5. **Open Problems**：上 9 条。

---

## 调研日志

- **调研日期**：2026-04-23
- **Topic Survey 性质**：新建（vault 此前无 `Topics/WorldModel-Survey.md`，也无 `DomainMaps/WorldModel.md`；本次为完整展开而非 delta 报告）
- **认知基线**：`DomainMaps/_index.md` 的 Cross-Domain Insights 段 + [[2411-WorldModelSurvey|Ding et al. 2024/CSUR]] 作为唯一结构化 anchor survey（v1 2024-11 → v4 2025-12）
- **侦察 survey 数**：1（[[2411-WorldModelSurvey]]，本 vault 已有 digest）；额外无新的 WM 主题 survey paper 被纳入——2024 年 11 月之后社区主要发生在 method paper（非 survey）层
- **候选论文清单（vault 已有 rating ≥ 2）**：共 21 篇——  
  **Foundation (3)**：[[2402-Genie]]、[[2405-DIAMOND]]、[[2408-GameNGen]]、[[2501-Cosmos]]、[[2505-DreamGen]]、[[2506-VJEPA2]]、[[2602-DreamZero]]  
  **Frontier (2)**：[[2411-WorldModelSurvey]]、[[2405-OccSora]]、[[2405-Vista]]、[[2406-IRASim]]、[[2501-RoboticWorldModel]]、[[2503-CosmosReason1]]、[[2504-UWM]]、[[2512-GenieReasoner]]、[[2512-Motus]]、[[2602-GigaBrain05M]]、[[2602-WorldVLALoop]]、[[2604-GenWorldRenderer]]、[[2604-HYWorld2]]、[[2604-OpenWorldLib]]
- **未 digest 新论文**：本次未启动新 paper-digest（vault 覆盖充分，新增 canonical 论文的 marginal gain 低于时间成本）。若后续要深化，优先补：(1) Ha & Schmidhuber 2018 (World Models, 奠基)；(2) Dreamer v3；(3) Sora technical report；(4) WHAM（Microsoft gameplay WM）；(5) 1X World Model；(6) UniSim (Yilun Du)；(7) Matrix-Game / YUME / FlashWorld（被 OpenWorldLib 引用的 open-source baseline）
- **observation**：2024–2026 这轮 WM 研究从"能否生成逼真视频"迁到"能否作为 VLA agent 的 backend"——5 条技术路线（pixel video / latent JEPA / 3D-4D / unified VLA+WM / WM-as-RL-simulator）同时成熟，但 Action-following、Long-horizon drift、Physics alignment 三大 open problem 都还未被任何单一方法解决
- **issues**：none（所有引用论文 vault 已有笔记，无 paper-digest 失败）
