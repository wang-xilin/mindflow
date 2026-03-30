---
title: "Vision-Language-Action Models"
last_updated: "2026-03-27"
---
## Overview

用 Vision-Language Model 驱动 robot action generation 的统一框架。核心问题：如何将 VLM 的语义理解能力转化为精确的物理操作？研究意义在于实现 language-conditioned 的通用机器人控制。

**发展现状**：以 π₀ 系列为代表，flow matching + hierarchical inference 成为主流架构。开源生态（Octo → OpenVLA → SmolVLA）快速迭代，小模型（0.45B）已能超越大模型（7B）。但 real-world 大规模部署仍在早期，safety guarantee 和 autonomous self-improvement 几乎空白。

## Core Concepts

- **VLA (Vision-Language-Action Model)**: 接受视觉和语言输入、直接输出 robot action 的端到端模型
- **Flow Matching**: 基于连续 flow 的 action generation 方法，支持高频（50 Hz）连续控制，替代离散 token prediction
- **Action Expert**: VLA 架构中专门负责 action generation 的模块，与 VLM backbone 解耦
- **Hierarchical Inference**: 高层 VLM 做语义推理（subtask planning），低层 action expert 做运动生成
- **Cross-embodiment**: 跨不同机器人形态的数据共享与迁移学习

## Established Knowledge

1. **Flow matching + action expert 是当前最强 action generation 范式**：连续 action 生成（flow matching）在控制频率和灵巧操作上全面超越 autoregressive token prediction。π₀ 系列确立了 VLM backbone + flow matching action expert 的标准架构。
   - 来源：[[2410-Pi0]]、[[2504-Pi05]]、[[2412-RoboVLMs]]

2. **Hierarchical inference（高层语义推理 + 低层 action 生成）是 long-horizon 任务的有效架构**：π0.5、Hi Robot、NaVILA 均验证了分层设计的优越性。Hi Robot 的独立 VLM reasoning + VLA execution 超越 GPT-4o 40%+。
   - 来源：[[2504-Pi05]]、[[2502-HiRobot]]、[[2412-NaVILA]]

3. **Fine-tuning 设计选择比模型规模更重要**：OpenVLA-OFT（7B，优化 fine-tuning）达到 97.1%（LIBERO），超越 π₀（3.3B）的 94.2%；SmolVLA（0.45B）超越 OpenVLA（7B）。VLA 领域存在显著的过参数化。
   - 来源：[[2502-OpenVLA-OFT]]、[[2506-SmolVLA]]、[[2412-RoboVLMs]]

4. **Data diversity >> Data specificity**：π0.5 的 97.6% 训练数据不来自目标任务，但 co-training 对泛化至关重要。Post-training 策略（先 cross-embodiment 预训练，再 in-domain fine-tune）优于直接 co-training。
   - 来源：[[2504-Pi05]]、[[2412-RoboVLMs]]

5. **开源生态加速迭代**：Octo → OpenVLA → SmolVLA 不断降低研究门槛。Open X-Embodiment 数据集是事实标准。
   - 来源：[[2405-Octo]]、[[2406-OpenVLA]]、[[2506-SmolVLA]]

## Active Debates

1. **RL self-improvement vs 更多 demonstration，哪个更 cost-effective？** π\*₀.₆ 的 Recap 实现 >2× throughput 提升，但仍需人工 reward labeling。更多高质量 demonstration 是否能达到同样效果？
   - 来源：[[2511-PiStar06]]、[[2603-RoboClaw]]

2. **Cross-embodiment 数据的价值**：RoboVLMs 发现 in-domain 数据比 cross-embodiment 数据更有效，但 π0.5 表明 cross-embodiment co-training 对 open-world generalization 至关重要。两者在不同 context 下可能都对——关键在于使用方式（pre-training vs co-training vs post-training）。
   - 来源：[[2412-RoboVLMs]]、[[2504-Pi05]]

3. **最优 action representation**：Flow matching 对 multimodal action distribution 建模更好但计算成本高；L1 regression 在 unimodal 场景下更简洁高效。最优选择可能是 task-dependent。
   - 来源：[[2502-OpenVLA-OFT]]、[[2506-SmolVLA]]、[[2412-RoboVLMs]]

## Open Questions

1. **Navigation + Manipulation 统一架构**：如何在单一 VLA 中同时支持 building-scale navigation 和灵巧操作？详见 [[VLN-VLA-Unification]]。
2. **Fully autonomous self-improvement**：如何去除人工 reward labeling，实现 intrinsic motivation / self-supervised reward？
3. **长期空间记忆**：MEM 解决了 15 分钟级记忆，但缺乏 explicit spatial memory。如何维护 persistent 的空间表示支持跨房间任务？
4. **Safety 与 failure recovery**：VLA 走向真实世界部署后，系统性的 safety guarantee 和 failure recovery 机制几乎空白。
5. **VLA 的 scaling law**：模型规模、数据规模、任务多样性之间的 scaling 关系尚不清楚。SmolVLA 的成功暗示 VLA 的 scaling law 可能与 LLM 不同。

## Known Dead Ends

1. **纯 autoregressive token prediction 做灵巧操作**：RT-2 的 ~3 Hz 控制频率已被 flow matching 的 50 Hz 全面超越。离散化 action tokenization 在精细操作中精度不足。但 autoregressive 仍可用于高层语义推理（如 π0.5 的 subtask prediction）。
   - 来源：[[2307-RT2]]、[[2410-Pi0]]

2. **GPT-4 zero-shot 做机器人控制**：π0.5 对比 GPT-4 zero-shot 高层规划仅 ~20% 成功率，纯 LLM 规划不足以驱动真实 robot。需要 fine-tuned VLM + 视觉 grounding。
   - 来源：[[2504-Pi05]]、[[2502-HiRobot]]
