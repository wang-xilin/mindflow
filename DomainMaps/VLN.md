---
title: "Vision-Language Navigation"
last_updated: "2026-03-27"
---
## Overview

通过自然语言指令引导 agent 在真实或模拟环境中导航。核心问题：如何将语言指令 grounding 到可执行的导航策略？与 VLA 趋同的关键在于 hierarchical planning 和 spatial representation。

**发展现状**：Sim 环境（R2R、R2R-CE）上趋于成熟，ETPNav 等方法已达较高 SR。NaVILA 证明 VLN 可重构为 navigation-focused VLA，real-world 达 88% SR。但绝大多数工作仍限于单层室内短距导航，sim-to-real gap 和 building-scale long-horizon 场景仍是开放挑战。

## Core Concepts

- **VLN (Vision-Language Navigation)**: 给定自然语言指令，agent 在视觉环境中导航到目标位置
- **Topological Map**: 以节点（viewpoint）和边（connectivity）表示环境的图结构，是 VLN 的主流空间表示
- **Nav-graph vs Continuous**: 离散导航图（teleport between viewpoints）与连续动作空间（low-level velocity control）两种评测设定
- **Mid-level Action**: 介于高层语义规划和低层运动控制之间的动作抽象（如 "turn left and go to the door"）

## Established Knowledge

1. **Topological map 是 VLN 的核心空间表示**：从 DUET 的 dual-scale graph transformer 到 ETPNav 的 online topological mapping，topological map 作为 intermediate spatial representation 被广泛验证有效。
   - 来源：[[2202-DUET]]、[[2304-ETPNav]]

2. **Hierarchical planning（high-level waypoint + low-level control）是 continuous VLN 的有效范式**：ETPNav 在 R2R-CE 上大幅超越 prior SOTA，这一架构与 VLA 的 hierarchical inference 高度平行。
   - 来源：[[2304-ETPNav]]、[[2412-NaVILA]]

3. **VLN 可以重构为 navigation-focused VLA**：NaVILA 将 VLM 微调为 navigation VLA，用语言化 mid-level action 桥接高层规划和低层运动控制，R2R-CE 54% SR，real-world 88% SR。这是 VLN-VLA 架构趋同的直接证据。
   - 来源：[[2412-NaVILA]]

4. **YouTube 视频是 scalable 的 real-world navigation 数据源**：NaVILA 用 2000 个 YouTube egocentric touring videos 生成 20000 条 training trajectories，有效缓解了 robot navigation 数据稀缺问题。
   - 来源：[[2412-NaVILA]]

## Active Debates

1. **Discrete nav-graph vs continuous action space**：传统 VLN 在 discrete nav-graph 上性能更高（DUET R2R 72% SR），但 continuous 环境更接近真实场景。两种 action space 的最优统一方式未定。
   - 来源：[[2202-DUET]]、[[2304-ETPNav]]

2. **显式空间表示 vs end-to-end**：VLN 传统依赖 hand-designed spatial representation（topological map），而 VLA 趋势是 end-to-end。MTU3D 的 online query memory 是一种折中——learned 但仍 explicit。
   - 来源：[[2507-MTU3D]]、[[2412-NaVILA]]

## Open Questions

1. **Sim-to-real gap**：VLN 绝大多数工作在 Habitat/MP3D 中评估，真实世界部署案例极少。NaVILA 是少数例外，其 language mid-level action 策略是否可推广？
2. **Building-scale long-horizon navigation**：现有评测多为单层室内短距导航，跨楼层、跨建筑的长距导航缺乏研究。
3. **VLN 与 VLA 的 shared spatial representation**：导航过程中构建的空间表示能否直接服务到达目的地后的 manipulation？详见 [[DomainMaps/SpatialRep|SpatialRep]]。

## Known Dead Ends

1. **GPT-4 zero-shot navigation reasoning**：NavGPT 证明纯文本 LLM（无视觉 grounding）的 zero-shot navigation 性能远低于 trained models。需要 visual alignment（NavGPT-2）或 VLM fine-tuning（NaVILA）。
   - 来源：[[2305-NavGPT]]
