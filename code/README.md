# PDF 提取、输入封装和独立评审代码

这是 2025/2026 实验中实际使用的核心 Python 代码的可移植整理版。主 Agent 负责调度和检查异常，脚本负责批量读取、匿名化、图文封装和创建独立评审会话。

原来的本机目录、用户名、Java 安装路径、Codex 安装路径和模型缓存路径已改为命令行参数或相对路径。`provenance.json` 记录来源脚本和原始 SHA-256。图文提取及评审调用核心逻辑沿用原实现；新增了统一清单、离线导出和单篇五次评审入口。未包含账号文件、PDF、模型权重、运行结果或缓存。

## 文件

| 文件 | 作用 |
|---|---|
| `anonymize.py` | 本稿作者、单位、邮箱、身份说明等匿名化 |
| `detect_figures.py` | 调用 PDFFigures2，提取图表范围并核对图注 |
| `detect_layout.py` | 使用本地 DocLayout-YOLO 权重做独立版面检测 |
| `merge_figure_detections.py` | 合并检测结果，支持经过检查的校正记录 |
| `prepare_review_inputs.py` | 生成文字、论文插图和交错顺序索引 |
| `validate_review_inputs.py` | 核对文字覆盖、姓名残留、图片完整性及输入顺序 |
| `load_review_input.py` | 按索引加载文字和图片，检查图片 SHA-256 |
| `export_context.py` | 在图文输入前加固定 prompt，保存可搬动的评审任务 JSON |
| `review_protocol.py` | 两年实验实际使用的 user prompt 和返回 schema |
| `review_runner_appserver.py` | 单次独立评审的 app-server 通信与审计 |
| `run_five_reviews.py` | 同一输入依次运行五个独立会话，汇报均值和标准差 |

## 1. 准备环境和输入清单

以下命令均从仓库根目录运行，使用 Python 3.10 或更高版本。核心预处理依赖的版本来自本次本地环境：

```bash
python -m pip install -r code/requirements.txt
```

复制并修改 `code/examples/papers.jsonl`。每行是一篇论文，至少填写 `pdf_path` 和 `authors`；`paper_id` 可填写 OpenReview ID 或自己的编号。相对 PDF 路径以清单文件所在目录为基准。作者可以是姓名列表，也可以是分号分隔的字符串；已匿名且不知道作者的稿件可填 `[]`，仍需通过匿名署名检测。

清单顺序固定对应 `paper_001`、`paper_002` 等编号。不要在检测和提取之间重新排序。来源清单属于控制端，不发送给评测模型。

## 2. 图表检测

PDFFigures2 需要 Java；实验使用 Java 17。准备 [PDFFigures2](https://github.com/allenai/pdffigures2) 的源码和构建工具 sbt；为接近原实验，使用提交 `3d7ad46753d4a315cccd1c2bcab398380e88c534`，应用 `patches/pdffigures2-iclr.patch` 后执行 `sbt assembly`。补丁处理 ICLR 行号边栏和页眉线，第三方许可证一并保存在 `patches/`。将生成的 jar 放到自己指定的位置。

```bash
python code/detect_figures.py --input code/examples/papers.jsonl --output code/work/figures --jar code/work/vendor/pdffigures2.jar --remove-gutters
```

`java` 默认从 PATH 查找，也可以用 `--java` 指定。原文 PDF 保持不变，检测专用副本写入输出目录。不要在同一个输出目录混用不同的论文清单。

版面检测使用 [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)。按机器环境安装对应的 PyTorch，再安装布局依赖，手动下载 `juliozhao/DocLayout-YOLO-DocStructBench` 的 `doclayout_yolo_docstructbench_imgsz1024.pt` 权重。检测脚本只加载指定的本地权重，不自动下载。

```bash
python -m pip install -r code/requirements-layout.txt
python code/detect_layout.py --input code/examples/papers.jsonl --output code/work/layout --weights code/work/vendor/doclayout_yolo_docstructbench_imgsz1024.pt --device cpu
python code/merge_figure_detections.py --manifest code/examples/papers.jsonl --captions code/work/figures/normalized --layout code/work/layout --output code/work/fused
```

有适用 GPU 时可设置 `--device cuda:0`。原实验权重 SHA-256 为 `9a2ee0220fe3d9ad31b47e1d9f1282f46959a54e4618fce9cffcc9715b8286e2`。依赖、分词器资源和权重首次准备可能需要网络；这些步骤不调用评审模型。

## 3. 提取、检查、封装

```bash
python code/prepare_review_inputs.py --manifest code/examples/papers.jsonl --detections code/work/fused --output code/work/inputs
python code/validate_review_inputs.py code/work/inputs
python code/export_context.py code/work/inputs/packages/paper_001 --output code/work/jobs/paper_001.json
python code/run_five_reviews.py code/work/jobs/paper_001.json --dry-run
```

每篇生成 `paper.txt`、`text.json`、`figures.json`、`figures/*.png`、`index.json` 和 `preview.html`。正文、表格和图注尽量保留为文字，真正的论文插图作为图像穿插；图中的文字保留在图像里。`paper.txt` 还保留提取出的图内文字，实际交错输入通过索引避免重复添加这些图内文字。

`controller/` 保存作者映射、原始身份信息和检查记录，只由主 Agent 使用。查看 `preview.html` 和 `controller/*/audit.json`，检查图像漏检、正文误裁、图注错配等问题。原实验也有针对性视觉检查和校正，通用检测代码无法自动复现每一条历史人工校正。

需要修改检测框时，可编辑 `code/work/fused/paper_XXX.json`，或在 `code/work/inputs/controller/figure_overrides.json` 中按匿名编号提供 `figures`（完整替换）、`remove`（`[页码, kind, name]` 列表）和 `add`。检测项使用 1 起始页码，`bbox` 为 PDF 点坐标 `[x0,y0,x1,y1]`。合并脚本还支持 `--qa` 和 `--caption-corrections`，沿用原脚本的数据格式。修改后重跑提取和验证；模型正在使用的输入不得修改。

自动验证会核对覆盖、哈希和已知作者姓名，但不等于逐个字符语义、所有隐含身份线索和所有图像边界都正确。存在异常时先修正，再开始评审。

导出的任务 JSON 包含固定 prompt、system/developer 指令、模型设置、输出 schema、按顺序排列的文字和相对图片路径。图片路径相对任务 JSON 所在目录；搬动时应保留任务与输入包的相对关系。调用程序会加载图片，评测模型不需要调用工具读取文件。以上四条命令不发起模型评审。

## 4. 真正运行五次独立评审

使用已有 ChatGPT 登录的 Codex CLI，并确保账号可用 `gpt-5.6-sol`。程序沿用实验时的 app-server 参数、实验字段和功能禁用配置；原记录的客户端版本为 `0.155.0-alpha.9.2`，不同版本可能需要适配，不能保证任意版本都兼容。接口参考 [OpenAI 官方 app-server 文档](https://learn.chatgpt.com/docs/app-server)。

```bash
python code/run_five_reviews.py code/work/jobs/paper_001.json --output code/work/reviews/paper_001
```

**这条命令会实际调用模型并消耗账号额度。** CLI 默认从 PATH 查找 `codex`，也可以传 `--executable` 或设置 `CODEX_EXECUTABLE`。每轮检查额度；程序不读取、复制或保存账号凭据，不兑换额度，不自动重试失败评审。

五轮按顺序执行，每轮建立新的空临时会话，固定使用 GPT-5.6 Sol、medium 和同一输入。核对模型设置、输入回显顺序、线程独立性和工具调用事件；工具调用会导致评审隔离为无效结果。输出目录必须是新目录，避免覆盖已有结果。失败后先检查记录，不要按分数挑选重跑。

输出包含各轮原始 JSON/文本、配置、运行状态、用量和最终 `summary.json`。未保存隐藏推理或完整通信日志。运行时审计记录会引用调用者自己的输入文件位置；它们不是本仓库发布的代码或历史评审 JSON。

本入口只汇报分数，不自动估算个人中稿概率。按年份统计参考接收占比的约定见根目录 `AGENTS.md`。

## 离线检查

```bash
python -m unittest discover -s code -p "test_*.py" -v
```

测试使用模拟 app-server，检查输入顺序、非法工具调用、模型配置、schema 和失败处理，不启动真实评审。除通用入口和参数适配外，没有重写或重新生成仓库中的 1,000 份历史评审。

整理时已通过 14 项离线测试；用一篇 23 页论文跑通检测、匿名化、封装与验证，生成 24 张插图。使用原检测结果重建的输入包逐字节一致；两年共 200 个历史输入包经本版加载器加载后，输入顺序、文字哈希及图片哈希均与原评审配置一致。本次整理没有发起新的模型评审。
