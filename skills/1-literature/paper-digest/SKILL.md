---
name: paper-digest
description: Use when user asks to "read paper", "analyze paper", "summarize paper", "读论文", "分析文献", "帮我看一下这篇paper", "论文笔记", or provides a PDF file that appears to be an academic paper.
argument-hint: "[arXiv URL / blog URL / PDF path / title]"
allowed-tools: Read, Write, Edit, Glob, Grep, WebSearch, WebFetch
---

## Purpose

给定一篇论文或技术 blog 的来源（URL、PDF 路径、论文标题或者关键词），它自动获取内容、提炼核心信息，生成结构化笔记。

支持两种内容类型（在 Step 2.2 的 extraction 草稿中标记，仅用于 Step 2.3 的 sanity check；**不**用于决定笔记结构——body 结构统一由 `source_sections` 驱动）：
- **paper**：arXiv、PDF 等学术论文（图/公式/benchmark/ablation 完整）——触发 "0 公式 / 0 图" 重抓检查
- **blog**：所有 blog 类来源（Lilian Weng 教程、Google Research Blog、公司发布会、Generalist AI / OpenAI / Anthropic 的 announcement 等）——不触发 sanity check，允许 0 公式 / 0 数字表

## Steps

### Step 1：Dedup 预检（在 fetch 之前必做）

任何 fetch 之前先扫一遍 `Papers/`，避免对一篇已有笔记重复 digest 浪费 5 分钟和几十次 tool call。

1. **从 source 猜 ShortTitle / 关键标识**：
   - arXiv URL → 用 arxiv id 反查：`Grep "arxiv.org/abs/<id>" Papers/`（frontmatter `url:` 字段）
   - 论文标题/关键词 → 用关键词 CamelCase 形式 Glob
   - Blog URL → 从 URL slug 提取关键词（如 `apr-02-2026-GEN-1` → `GEN1`）
2. **多模式扫描**：
   ```
   Glob Papers/*<ShortTitleCamelCase>*.md
   Glob Papers/*<acronym>*.md         # 若有缩写
   Grep "<source_url>" Papers/        # frontmatter url 字段反查
   ```
3. **命中处理**：
   - 找到对应笔记 → **立刻停止**，告知 Human "已有笔记 [[Papers/xxx]]，是否需要 force-refresh / 增量更新 / 取消"
   - 没命中 → 进 Step 2 fetch + extract

> **为什么前置**：subagent 测试中跑完 Step 2.1–2.3 整套 verify 才在 Save 环节撞到 dedup，~5 分钟 + 几十次 tool call 全部浪费。dedup 是 fetch 的前置条件，不是 save 的后置检查。

### Step 2：Fetch + Extract + Verify（写笔记之前必做）

**目的**：拉取源材料 → 结构化抽取所有"事实性元素"（章节大纲、图、表、公式、视频）→ 逐条对照源材料验证它们真实存在。Step 3 的笔记 compose **只允许**使用这一步抽取并验证过的内容，不允许在写笔记时凭印象引入任何新 URL、新数字、新公式。

**为什么强制这一步**：fabrication（捏造图链接、目测图表数字、错引公式、漏掉视频）几乎永远发生在"边读边写"的模式里。把"抽取"和"compose"切开，事实层和叙事层各自负责，捏造路径就被堵死了。

#### Step 2.1：fetch 源材料并落盘到本地

后续 grep verification 必须有一个**本地可读的 raw 副本**——所有 fetch 直接 curl 到 `/tmp/`，不用 WebFetch（WebFetch 的 LLM summarizer 会丢图/视频/数字 URL）。

> **若 source 是论文标题或关键词**：先 WebSearch（建议加 `site:arxiv.org`）定位 arxiv id，再按 arXiv URL 处理。

**文件名约定**（避免并行 / 重复 digest 时互相覆盖）：源材料文件名必须**包含 source 唯一标识**，禁用 `/tmp/paper.html`、`/tmp/page.html` 这种通用名。

| 来源                                     | 落盘方式                                                                                              |
| -------------------------------------- | ------------------------------------------------------------------------------------------------- |
| arXiv URL `https://arxiv.org/abs/{id}` | Bash `curl -sL https://arxiv.org/html/{id} -o /tmp/arxiv_{id}.html`（如 `/tmp/arxiv_2512.04601.html`） |
| Blog URL                               | Bash `curl -sL <url> -o /tmp/blog_{last-path-segment}.html`（如 `/tmp/blog_apr-02-2026-GEN-1.html`） |
| PDF 路径                                 | 已经是本地文件，直接用路径                                                                                     |

**记住这个路径**，后续 verification 会反复 grep。

#### Step 2.2：输出结构化 extraction 草稿

在对话里产出一个 YAML/Markdown 结构化草稿（不写文件，只在工作记忆里），字段如下：

```yaml
content_type: paper | blog
source_local_path: /tmp/arxiv_2512.04601.html  # Step 2.1 的路径，必须是带 source 唯一标识的文件名
bibliographic:
  title: ...
  authors: [...]
  institutes: [...]                 # 作者机构，找不到就留空，不强制 grep 验证
  date_publish: YYYY-MM-DD or YYYY-MM  # arXiv 源可从 id 直接推断：`2504.16054` → 2025-04，`2602.12684` → 2026-02
  venue: ...
  source_url: ...
source_sections:                    # 源文档的大纲，顺序保留
  - id: sec1
    heading: "Introduction"
  - id: sec2
    heading: "The π0.5 Model"
  - id: sec3
    heading: "Experiments: Unseen Homes"
  - id: sec4
    heading: "Data Co-Training Ablation"
figures:
  - id: fig1
    caption: "..."
    url: "https://..."              # 或 null（canvas / 无静态 URL，只记 caption）
    section_id: sec2                # 挂载到哪个 source_section；compose 时就近嵌入
tables:
  - id: tab1
    caption: "..."
    content: |                        # 从源复制的 Markdown 表体（表头 + 数据行），compose 时直接粘贴到笔记，不在 compose 阶段重新组装
      | Col1 | Col2 | Col3 |
      |---|---|---|
      | ... | ... | ... |
    section_id: sec3
equations:
  - id: eq1
    name: "Flow matching loss"
    latex: "\\mathcal{L}_{FM} = ..."
    section_id: sec2
videos:
  - caption: "..."
    url: "https://.../clip.mp4"      # 或 youtube.com/watch?v=...
    section_id: sec3
```

**抽取原则**：
- 只抽取**源材料中实际存在**的元素。若某一项不存在，跳过，**禁止捏造**。
- 对 PDF：图通常无外链 URL，先标记 `url: null` + caption；后续 compose 阶段决定要不要本地下载。
- **`source_sections`** 按源文档的章节顺序记录大纲（paper 用 `\section{}` / `\subsection{}` 或 `<h2>` / `<h3>`；blog 用 `<h2>` / `<h3>` / 明显的段落标题）。层级和粒度根据源结构判断。**排除非技术 meta 章节**：Acknowledgments、Contributions / Author List、References / Bibliography、Appendix（附录只在明显包含核心 method / experiments 补充材料时才纳入）——这些章节不进 `source_sections[]`，也不承载 figures / equations / tables / videos。每个 `figure` / `equation` / `table` / `video` 必须挂 `section_id` 指向它在源里所属的章节——这决定了 Step 3 compose 时它出现在笔记的哪一段。

#### Step 2.3：URL & 事实存在性验证（硬性，不可跳过）

对 Step 2.2 抽取的草稿，逐条用 Grep 跑下面的检查，**任何一项失败必须修正或删除该条目**：

| 字段                       | 验证命令（伪代码）                                                                                    | 失败处理                           |
| ------------------------ | -------------------------------------------------------------------------------------------- | ------------------------------ |
| `figures[].url`          | `Grep <url> in <source_local_path>`                                                          | 命中 0 次 → 删除该 figure，不允许出现在最终笔记 |
| `videos[].url`           | `Grep <url> in <source_local_path>`                                                          | 同上                             |
| `tables[].content` 里的数字  | **非相邻**抽样 ≥3 个数字 grep（跨行 + 跨列，不取同一行或同一列的连续 cell），每个独立 `Grep <number> in <source_local_path>` | 任一未命中 → 整张表删除，禁止从图表目测          |
| `equations[].latex` 关键符号 | Grep 公式名或主要算子 in 源                                                                           | 命中 0 次 → 删除该公式                 |


**Verify 的硬性禁区**：
- ❌ 把 WebFetch summarizer 用文字描述的图当成"存在的图" —— 必须有可 grep 到的 URL
- ❌ 把 bar/line chart 目测的数字写成 Markdown 表 —— 数字必须能在源里 grep 到
- ❌ 把"论文领域常见公式"凭印象补进 Method 段 —— 公式必须在源里 grep 到

跑完这一步，extraction 草稿里只剩**已验证为真**的事实元素。

---

### Step 3：笔记生成与保存

**Compose 的硬约束**：所有出现在笔记里的 URL、数字、公式、表格，**必须**来自 Step 2 extraction 草稿。compose 阶段不允许引入任何新的事实性元素。如果 compose 时发现遗漏，**回到 Step 2 重新抽取并验证**，不要直接补写。

**Wikilink 推迟到 Step 4**：Step 3 写入的笔记正文里，所有技术术语（"π0"、"Flow Matching" 等）都保持**纯文本**，不包 `[[...]]`。Wikilink 注入在 Step 4 作为独立的 post-compose pass 进行，候选术语在那一步从正文扫描识别，不依赖 extraction 预先列出。

**两个参考文件**：
- **模板**：`references/paper-note-template.md` —— frontmatter + 固定 shell 骨架（top + outro）
- **Syntax 参考**：`references/obsidian-syntax.md` —— 公式 / 图 / 视频 / wikilink / 表格的 Obsidian-specific 语法 quirks

#### 结构模型：固定 shell + 源结构 body

**固定 shell**（每篇笔记不变）：
- **Top**: frontmatter → `## Summary` + Key Takeaways → 可选 teaser 媒体（inline，不特殊 slot）
- **Outro**: `## 论文点评`（Strengths / Weaknesses / 可信评估）→ `## 关联工作` → `## 速查卡片` → `## Notes`

**Body**（镜像源结构）：
- 逐个展开 `source_sections[]`：用每个 section 的 `heading`（源标题原文）作为笔记 body 的标题。层级（`##` / `###`）根据源结构和阅读顺畅度判断，不强制
- 在每个 section 下，嵌入 `section_id` 匹配的 figures / equations / tables / videos，**就近放置**
- 源里没有的 section 不写；源里有但 extraction 没挖出任何内容的 section 也不写

**Teaser 处理**：若 extraction 草稿里存在一个明确是 overview / concept / motivation 用途的 figure 或 video，嵌入 `## Summary` + Key Takeaways 之后的 top 区域（无特殊 slot 或标题）。若论文唯一的 high-level 视觉就是架构图，跳过 teaser——架构图自己会出现在对应的 body section 里。

#### 不变的硬规（所有 content_type）

- **extraction 完整性**：extraction 草稿里的每个 figure / equation / table / video 都必须进笔记。content_type 不豁免任何元素。
- **媒体就近嵌入**：figures / equations / tables / videos 必须放在挂载的 section_id 对应的 section 内，禁止集中的 `## Figures` / `## Equations` / `## Media` 倾倒段。
- **装饰性图片**：OG image / social card / banner / favicon 一律不嵌入。
- **模板注释**：模板里的 `%% ... %%` 提示必须全部替换或整块删除，禁止外泄到正文。
- **媒体语法**：YouTube / 外链 mp4 / 外链图 / 本地图 / 公式的写法全部按 `obsidian-syntax.md`。
- **wikilink 留空**：技术术语保持纯文本，不写 `[[...]]`，等 Step 4 统一扫描 + 注入。

#### 文件名 + Tag + Write

**文件名格式**：`YYMM-ShortTitle.md`

- `YYMM`：取自 `date_publish` 的年份后两位 + 月份，如 `2603`（2026年3月）
- `ShortTitle`：标题的 CamelCase 缩写，2-4 个关键词，如 `EvoScientist`、`RoboClaw`、`DiffusionPolicy`
- Blog 同理，如一篇 2026 年 2 月的 blog 关于 scaling laws → `2602-ScalingLaws.md`

**Tag 选择**：阅读 vault 根目录下的 `references/tags.md`（即 `{vault_root}/references/tags.md`，**不是**本 skill 内的 `references/`），按照规范选择 tag。

> Dedup 检查已在 Step 1 完成，此处不再重复。如果走到这里才发现命名冲突，说明 Step 1 的多模式扫描漏掉了——回头加 pattern。

**写入文件**：用 Write 将笔记（纯文本版，无 wikilink）保存到 `Papers/{笔记文件名}.md`。

---

### Step 4：Wikilink 注入（独立的 post-compose pass）

**目的**：扫描 Step 3 写出的笔记正文，识别技术术语，把能在 vault 找到对应笔记的 term 就地升级成 `[[...|term]]` wikilink。Compose 和 graph building 解耦，术语识别和 Concept Glob 协议集中在这一步处理。

**输入**：Step 3 刚写入的 `Papers/{笔记文件名}.md`（extraction 草稿已用完，不再参与）

**流程**：

1. **Read 笔记正文**，跳过 frontmatter、代码块、表格分隔线
2. **识别候选术语**（LLM 判断，不是机械正则）：
   - 论文名（`π0`、`RT-2`、`OpenVLA`、`Diffusion Policy`）
   - 核心方法 / 算法名（`Flow Matching`、`FAST tokenizer`、`Action Chunking`）
   - 数据集 / benchmark 名（`OXE`、`DROID`、`Bridge`）
   - 具体模型家族（`PaliGemma`、`Gemma-2B`、`SigLIP`）
   - **排除**：宽泛类别词（`VLA`、`world model`、`RL`、`transformer`）、通用英文词（`method`、`results`）、论文自己的术语
3. 对每个候选 term 执行 **Concept Glob 协议**
4. 命中 → 用**上下文锚定的 per-occurrence Edit**（**不是** `replace_all`）把笔记里出现的 term 包成 `[[{文件名}|{term}]]`（详见下方 *注入写法*）
5. 未命中 → 记入 dead-link inventory，term 保持纯文本

#### Concept Glob 协议

vault 的笔记命名约定是 `YYMM-CamelCase.md`（如 `2411-WorldModelSurvey.md`、`2410-Pi0.md`、`2604-GEN1.md`），**不是** `2411-World Model Survey.md`。所以朴素的 `Glob Papers/*<term>*.md` 对多词术语**几乎一定漏匹配**。

**匹配规则**：把 term 折叠成 CamelCase（去掉空格和 `-`），然后 Glob 三个可能的目录：

```
Glob Papers/*<TermCamelCase>*.md
Glob Concepts/*<TermCamelCase>*.md     # 若目录存在
Glob DomainMaps/*<TermCamelCase>*.md   # 若目录存在
```

例：
- `"World Model Survey"` → `*WorldModelSurvey*.md`（命中 `Papers/2411-WorldModelSurvey.md`）
- `"Flow Matching"` → `*FlowMatching*.md`
- `"GEN-1"` → `*GEN1*.md`
- `"π0.5"` → `*Pi05*.md`（Unicode / 特殊字符按读音或常用 ASCII 折叠）

任一目录命中即视为存在。命中后 Read 文件 frontmatter 的 `title:` 字段**确认主题相关**再注入——Glob 可能有无关的子串 false positive（如 `*Flow*` 匹配 `SnapFlow`）。

#### 注入写法

**绝对禁止使用 `replace_all=true`**。即使 term 看起来只出现一次，也必须用**上下文锚定的 per-occurrence Edit**。原因：技术术语互为前缀 / 子串是常态——

| 长 term | 短 term（被污染风险） |
|---|---|
| `π0.5`, `π0-FAST` | `π0` |
| `OpenVLA-OFT` | `OpenVLA` |
| `GEN-1`, `GEN-01` | `GEN-0` |
| `RT-2` | `RT-1` |
| `GPT-4o` | `GPT-4` |
| `Papers/2405-Octo.md`（路径） | `Octo` |

`replace_all` 对以上场景会：
- **污染已注入的 alias 文本**：先注入 `[[2504-Pi05|π0.5]]`，再 `replace_all("π0", ...)` → `[[2504-Pi05|[[2410-Pi0|π0]].5]]`（真实事故，见 2026-04-14 subagent 测试）
- **污染文件路径字符串**：`关联工作` 段如果出现 `Papers/2405-Octo.md`，`replace_all("Octo", ...)` → `Papers/2405-[[2405-Octo|Octo]].md`

**注入流程**（对每个命中 term，按顺序执行）：

1. **按长度降序处理 term**。先把 `π0.5` 全部注入完，再处理 `π0`。这样短 term 注入时，长 term 已经在 alias 里被保护了（alias 的前后会有 `|` 和 `]]`，可用作上下文锚点过滤）。
2. **Grep 枚举**：`Grep -n "<term>" <笔记路径>` 列出所有 occurrence 的行号和周围文本。
3. **过滤排除区**（必跳过）：
   - Frontmatter 区（`---` 之间）
   - 代码块（fenced ``` 或缩进 4 空格）
   - 已 wikilink：occurrence 左右窗口包含 `[[` / `|` / `]]`（说明该字符序列是已注入的 alias 或已有 wikilink 的一部分）
   - **更长术语的子串**：occurrence 右侧紧跟 `.5` / `-FAST` / `-OFT` / `-1` / `-2` / `o` 等后缀，或左侧被字母数字字符相邻（说明这是 `π0.5` / `OpenVLA-OFT` 等长 term 的一部分，不是 term 本身）
   - **文件路径**：occurrence 左侧包含 `Papers/` 或 `.md` 紧邻（说明是路径字符串的一部分）
4. **Per-occurrence Edit**：对每个合法 occurrence，用**包含周围字符的 old_string** 让匹配唯一，new_string 只把 term 本身包成 `[[文件|term]]`：

   ```
   # 例：注入 π0（π0.5 已注入完）
   # Grep 出两处：
   #   L42: "... baseline 是 π0 (Black et al.)"
   #   L58: "... π0.5 surpasses π0 in all tasks"  ← 右边是空格，合法
   
   Edit(old="baseline 是 π0 (Black", new="baseline 是 [[2410-Pi0|π0]] (Black")
   Edit(old="surpasses π0 in",       new="surpasses [[2410-Pi0|π0]] in")
   ```

   **表格 cell 内的 occurrence 必须用 `\|` 转义 alias 分隔符**（详见 `obsidian-syntax.md` §4.1）——否则 Markdown 会把 alias 的 `|` 当成列分隔符，wikilink 断裂。例：

   ```
   # 表格行：| Flow matching | π0 (Black et al.) | ...
   Edit(old="| π0 (Black", new="| [[2410-Pi0\\|π0]] (Black")
   ```

**为什么不用 replace_all**：省下的几次 Edit 调用代价是**静默损坏笔记**。Per-occurrence Edit 多几次工具调用，但每次替换都有上下文锚点，不会污染长 term 或文件路径。

**泛化类别词反模式**：宽泛类别词（`VLA` / `world model` / `scaling law` / `transformer`）在**候选识别阶段**就应被排除；即使漏网到 Glob 阶段并命中，也**不**自动 wikilink——它们在 vault 里可能有匹配文件，但论文正文用的是作为范畴词，不是指那篇具体 reference。

#### Dead-link Inventory

注入流程结束后，把所有**未命中**的 term 汇总成一个 dead-link inventory，写入 Step 5 的 log entry（见下一节）。这是 vault 长期建设的 TODO 信号，不写入笔记本身。

---

### Step 5：日志记录

用 Edit（或 Write 若文件不存在）将以下格式的 log entry 追加到 `Workbench/logs/YYYY-MM-DD.md`（日期为今天）：

   ```markdown
   ### [HH:MM] paper-digest
   - **input**: <source 的原始内容>
   - **output**: [[Papers/{笔记文件名}]]
   - **observation**: <一句话描述论文核心贡献>
   - **dead_links**: [term1, term2, ...]  # Step 4 未命中的 term，vault curation TODO
   - **status**: success
   ```

若日志文件不存在，先创建文件（包含一级标题 `# YYYY-MM-DD`），再追加 entry。

## Verify

### Step 2 (Extract & Verify) 自检
- [ ] 源材料已落盘到 `/tmp/...`，路径已记录
- [ ] extraction 草稿已产出，包含 `source_sections` / figures / tables / equations / videos 字段（**不**含 concept_links；术语识别在 Step 4 处理）
- [ ] **`source_sections`** 已按源顺序记录大纲；每个 figure / equation / table / video 都挂了 `section_id`
- [ ] **每个 `figures[].url` / `videos[].url` 都跑过 `Grep <url> in <source_local_path>` 且命中 ≥1**，未命中条目已删除
- [ ] **每张 table 的 `content` 字段**已从源复制为完整 Markdown（表头 + 数据行），非凭印象重组；至少 3 个非相邻数字已 Grep 到源里命中
- [ ] **每个公式**的关键算子或公式名已 Grep 到源里命中
- [ ] **Sanity check**：若 `content_type: paper` 且 extraction 找到 0 公式 / 0 图，已回 Step 2.1 重抓源材料

### Step 3 (Compose & Save) 自检
- [ ] `Papers/{笔记文件名}.md` 已创建
- [ ] 笔记中所有 URL / 数字 / 公式 / 表格 **均来自 Step 2 extraction 草稿**，无新增
- [ ] **笔记中所有技术术语保持纯文本**，未出现任何 `[[...]]` wikilink（注入推迟到 Step 4）
- [ ] **extraction 草稿里的每个 figure / equation / table / video 都已写入笔记**（无遗漏，content_type 不豁免）
- [ ] **Body 章节结构来自 `source_sections`**（标题用每个 section 的 `heading` 原文；层级根据源结构判断）
- [ ] 每个 figure / equation / table / video 嵌在 `section_id` 对应的 section，**就近放置**
- [ ] Teaser 媒体（若存在）嵌在 `## Summary` + Key Takeaways 之后的 top 区域
- [ ] 固定 shell 完整：frontmatter / `## Summary` + Key Takeaways / `## 论文点评` / `## 关联工作` / `## 速查卡片` / `## Notes`
- [ ] 模板中的 `%% ... %%` 提示已全部替换或留空，无提示文本外泄到正文
- [ ] `可信评估` 段两栏都已填：**Artifact 可获取性** checkbox 已勾 + **Claim 可验证性** 已用 ✅/⚠️/❌ 三档分类（任何 content_type 都填）
- [ ] `关联工作` 中无内容的子类已删除；子类标签与内容匹配（系列工作可用前作/后续/同类对比）
- [ ] YouTube 用 `![](https://www.youtube.com/watch?v=...)` markdown 语法；外链 mp4 用 `<video src=...>` 而非 `![](url.mp4)`
- [ ] 未嵌入 OG image / social card / banner 等装饰性图片

### Step 4 (Wikilink 注入) 自检 

**Read 最终写入的笔记文件**，肉眼逐个检查每个 `[[...]]` occurrence 是否合法。对每个 wikilink 过一遍下面的问题，任一 "否" 即需修复：

- [ ] **结构合法**：是 `[[文件名|显示文本]]` 形式，`[[` 和 `]]` 成对，`|` 恰好一个（表格 cell 内的 alias 分隔符已用 `\|` 转义）
- [ ] **无嵌套**：`|` 之后到 `]]` 之间是纯文本，不含另一个 `[[` —— 没有 `[[...|[[...|term]]]]` 这种双重嵌套
- [ ] **未污染路径**：wikilink 没有出现在 `Papers/xxxx-xxx.md` 这类文件路径字符串的中间（即路径里的 term 没有被错误包装成 `Papers/2405-[[2405-Octo|Octo]].md`）
- [ ] **未污染更长术语**：短 term（如 `π0`）的注入没有破坏更长术语的 alias（如 `[[2504-Pi05|π0.5]]` 里的 `π0` 没有被再次包装）

### Step 5 (Log) 自检
- [ ] 日志已追加到 `Workbench/logs/YYYY-MM-DD.md`
- [ ] Log entry 包含 `dead_links` 列表
