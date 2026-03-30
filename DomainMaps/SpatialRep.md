---
title: Spatial Representation
last_updated: 2026-03-27
---
## Overview

为 embodied agent 构建语义丰富的空间表示，支撑导航与操作。核心问题：如何设计一种 shared spatial representation 同时服务 VLN 的路径规划和 VLA 的物体操作？

**发展现状**：三类方案并行发展——dense feature maps（VLMaps）、3D scene graphs（ConceptGraphs）、neural/Gaussian fields（SplaTAM）。MTU3D 推动了 online spatial memory 方向。但 real-time incremental update 和 Nav+Manip 统一表示仍未解决，领域仍在快速发展中。

## Core Concepts

- **Semantic SLAM**: 在 SLAM 基础上融合语义信息（object class、language description）的空间建图方法
- **3D Scene Graph**: 以 object 为节点、spatial/semantic relation 为边的层次化场景表示
- **Dense Feature Map**: 每像素存储 VLM feature 的稠密表示（如 VLMaps），支持 open-vocabulary 空间查询
- **Neural/Gaussian Field**: 用 neural implicit 或 Gaussian Splatting 表示的连续 3D 场景（如 SplaTAM）
- **Online Spatial Memory**: 从 RGB-D 流实时增量构建的空间记忆，无需离线重建

## Established Knowledge

_高置信度的领域共识。_

1. **三类主流语义空间表示**：（1）dense feature maps（VLMaps：per-pixel VLM features）；（2）3D scene graphs（ConceptGraphs：object nodes + semantic relations）；（3）neural/Gaussian fields（SplaTAM：Gaussian Splatting SLAM）。各有适用场景。
   - 来源：[[2210-VLMaps]]、[[2309-ConceptGraphs]]、[[2312-SplaTAM]]

2. **3D scene graph 是最有潜力同时服务 VLN 和 VLA 的表示**：object nodes 提供 navigation waypoints（VLN）和 manipulation targets（VLA），semantic relations 支持 task planning。
   - 来源：[[2309-ConceptGraphs]]、[[VLN-VLA-Unification]]

3. **Online spatial memory 优于 offline 3D 重建**：MTU3D 的 online query merging 直接从 RGB-D 流构建 dynamic spatial memory，无需离线重建，更适合实时 agent。
   - 来源：[[2507-MTU3D]]

## Active Debates

_存在矛盾或未定论的观点。_

1. **Explicit spatial representation vs end-to-end**：VLA 趋向 end-to-end（无显式空间表示），但 VLN 和 long-horizon 任务仍需 explicit spatial memory。MEM 用 video + language memory 部分替代了 spatial representation，但缺乏 3D geometric 信息。最优方案可能是二者结合。
   - 来源：[[2603-MEM]]、[[2507-MTU3D]]

2. **Dense vs sparse 表示**：Dense feature maps 提供细粒度信息但计算成本高；sparse scene graphs 更高效但丢失几何细节。层次化方案（dense geometry + sparse semantic graph）是否可行？
   - 来源：[[2210-VLMaps]]、[[2309-ConceptGraphs]]、[[2312-SplaTAM]]

## Open Questions

_尚未回答的问题。_

1. **Shared spatial representation for Nav+Manip**：如何设计一种空间表示，既能支持 building-scale navigation planning，又能提供 manipulation 所需的 object-level 3D geometry？
2. **Real-time incremental update**：真实部署中 spatial representation 需要实时增量更新。ConceptGraphs 目前是 offline batch processing，如何改造为 online incremental？
3. **Language-grounded spatial querying**：如何让 VLA 的高层推理直接通过自然语言查询 spatial representation（如 "离我最近的杯子在哪里"）？
4. **Spatial memory 与 VLA memory 的融合**：MEM 的 video/language memory 和 spatial representation 如何统一？一个 agent 不应维护两套独立的记忆系统。

## Known Dead Ends

_已证伪或不推荐的方向。_

（暂无明确证伪的方向，该领域仍在快速发展。）
