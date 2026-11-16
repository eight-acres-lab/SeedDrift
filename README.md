# SeedDrift

**看看基因组里哪些基因像是从别的田里飘来的种子。**

SeedDrift 是一个检测水平基因转移（HGT）的开源工具：给它一个基因组，它找出可能来自哺乳动物、原核生物、其他真核生物或病毒的基因，并给出每个判定的证据和置信度。它自带网页、HTTP 接口和命令行，部署在你自己的服务器上。默认只报告远距离的转移（原核跨门，真核跨界），结果是候选而不是定论，详见[准确性](#准确性)和[适用范围与局限](#适用范围与局限)。

*SeedDrift finds genes in a genome that were likely acquired by horizontal gene transfer and shows the evidence for each call. Self-hosted, with a web page, an HTTP API and a command line.*

[当前版本 v0.1.0](https://github.com/eight-acres-lab/SeedDrift/releases/tag/v0.1.0) · [MIT 许可](LICENSE) · [数据与第三方软件的来源](REFERENCES.md)

## 功能

- **单基因组筛查**：逐个基因判断它最近的同源是在本谱系（基因组自己所在的大类群，原核默认是门）之内还是之外；在外的归到哺乳动物、原核、其他真核、病毒四类之一。哺乳动物单列，方便查看病原体和寄生虫里可能来自宿主的基因。
- **常见输入都能用**：蛋白或 CDS 的 FASTA、GenBank / EMBL、GFF3、未注释的基因组 contig（自动预测基因），压缩文件也可以；也可以只给一个下载地址。
- **每个判定都有证据**：比对命中、系统发育树、GC 组成、基因岛和污染检查，以及可选的结构检索（ESMFold + Foldseek）。
- **三档置信度**：高、中、低，方便先看最可靠的候选。
- **自己部署，数据不出机器**：所有计算在你的服务器上完成；参考数据由脚本从原始来源下载。
- **多基因组扫描（仅命令行，原型）**：给一组近缘基因组的物种树和基因树，用两者的调和找转移事件。网页和接口不提供这项功能，所需的 OrthoFinder 和 IQ-TREE 也不随部署安装，见 [docs/hgt-scan.md](docs/hgt-scan.md)。

## 快速开始

### 1. 部署

服务器需要：

- Linux，NVIDIA GPU（`nvidia-smi` 显示的 CUDA Version 为 12 或更高）；
- 约 120 GB 磁盘；
- 装有 `git`、`curl`、`bzip2`；
- 能访问 conda-forge、PyPI、GitHub、EBI、NCBI 和 HuggingFace。

不需要 root。

```bash
git clone https://github.com/eight-acres-lab/SeedDrift.git && cd SeedDrift && ./seeddrift.sh up
```

`up` 依次安装运行环境、下载并索引参考数据、启动服务、检查状态。所有内容都放在仓库目录里（运行环境在 `.mamba/`，数据在 `store/`，日志在 `.run/`），不改动系统环境。每一步都可以单独执行（`install`、`data`、`start`、`status`），中断后重跑会接着做。

下载和建索引要几个小时，建议在 `tmux` 或 `screen` 里运行。要换数据目录、改线程数（默认 16）或使用国内镜像，在运行 `up` 之前先编辑配置，见[配置](#配置)。

完成后检查：

```bash
./seeddrift.sh status
```

`tools`、`esmfold`、`api` 三行都应是 `running`。刚启动时 ESMFold 模型还在加载，过一两分钟再查。出错时看 `./seeddrift.sh logs api`（或 `tools`、`esmfold`）。

### 2. 打开网页

服务只监听服务器本机的 127.0.0.1:8600，网页也由它提供。在自己的电脑上开一条 SSH 隧道，保持这个终端不关：

```bash
ssh -N -L 8600:127.0.0.1:8600 user@your-server
```

浏览器访问 <http://localhost:8600>。页面右上角的圆点为绿色表示服务和数据都已就绪；黄色表示 ESMFold 模型还在加载；红色表示连不上或缺数据，鼠标悬停可以看到原因。

接口没有登录验证：能连上这个端口的人都可以提交任务、让服务器下载任意地址、删除任务。请通过 SSH 隧道访问，不要把端口直接开放到公网。

### 3. 跑一个示例

在网页的“示例基因组”里选一个，点“开始分析”。示例数据由服务器从 UniProt 和 Ensembl 下载，任务逐个排队执行。

| 示例 | 输入 | 可以看什么 |
|---|---|---|
| 大肠杆菌 *Escherichia coli* K-12 | 蛋白组，约 4,400 个蛋白 | 对照：外来候选应以原噬菌体和其他细菌门的基因为主，几乎不应出现哺乳动物来源 |
| 嗜肺军团菌 *Legionella pneumophila* Philadelphia 1 | 基因组 DNA，3.4 Mb | 从 DNA 开始自动预测基因，结果带基因组位置图；军团菌寄生在阿米巴里，有不少类似真核的蛋白 |
| 溶组织内阿米巴 *Entamoeba histolytica* HM-1:IMSS | 蛋白组，约 8,000 个蛋白 | 基因组论文报道过一批来自细菌的基因，看“疑似原核来源”的候选 |
| 兔脑炎微孢子虫 *Encephalitozoon cuniculi* GB-M1 | 真核基因组 DNA，2.5 Mb | 自动判断为真核，用 MetaEuk 预测基因 |

结果页顶部是各类别的基因数；点置信度“高”筛出最可靠的候选，点某一行展开它的证据。

一个 4,000 到 5,000 个基因的细菌基因组约 20 分钟出结果（16 线程、A800 80 GB、不做结构检索时的参考值）。

## 读懂结果

### 类别

| 类别 | 结果文件里的取值 | 含义 |
|---|---|---|
| 疑似哺乳动物 / 原核 / 其他真核 / 病毒来源 | `mammal` / `prokaryote` / `eukaryote` / `virus` | 最近的同源明显在本谱系之外（Alien Index ≥ 15，定义见[工作原理](#工作原理)）。来源类别由本谱系以外的前 10 个命中按得分加权得出 |
| 本谱系 | `native` | 外部同源没有明显优于本谱系的同源。本谱系的默认范围：原核按门；动物、真菌、植物按界；原生生物按所在的真核超类群 |
| 外部无同源 | `orphan` | 排除自身所在的属之后，深度检索也没有命中。可选做结构检索 |
| 仅重复结构命中 | `repeat` | 命中主要落在重复结构上，不做判定 |

### 置信度

| 置信度 | 取值 | 条件 |
|---|---|---|
| 高 | `high` | Alien Index ≥ 45，支持物种至少 2 个；并且本谱系补搜后仍无同源，或建树支持外来且外部最佳命中的序列一致度比本谱系最佳命中高至少 10 个百分点 |
| 中 | `medium` | Alien Index ≥ 45 或建树支持外来；建树没有落在本谱系；并且本谱系无同源，或上述一致度之差至少 5 个百分点 |
| 低 | `low` | 其余候选，以及位于疑似污染 contig 上的候选 |

“支持物种”是外部前 10 个命中里的不同物种数。建议先看“高”和“中”两档。真核来源的候选如果支持物种只有 1，要小心：这时命中的往往不是真正的真核基因，而是那个真核物种的参考组装里混入的细菌序列。

### 结果文件

| 文件 | 内容 |
|---|---|
| `genes.tsv` | 每个基因一行。主要的列：`category`（类别）、`confidence`（置信度）、`AI`（Alien Index）、`donor_group`（来源类群）、`support`（支持物种数）、`tree`（建树结论：`outside` / `home` / `mixed`）、`gc`、`gc3`、`comp_z`（组成偏离）、`island`（所在外来基因簇的大小）、`contig_flag`（疑似污染 contig），以及最佳命中和结构检索的各列 |
| `summary.json` | 整体摘要：基因数、各类别数量、自身类群、参数、各步用时 |
| `hits.tsv` | 参考库比对结果 |
| `proteins.faa` | 分析所用的蛋白序列 |
| `trees/*.nwk` | 外来候选的系统发育树 |
| `structures/*.pdb` | 结构检索预测的结构（开启时） |

网页和接口提交的任务，文件在服务器数据目录的 `jobs/<任务号>/` 下。网页上可以下载前四个；树和结构用接口取。

## 使用

### 网页

三种提交方式：上传文件、填一个地址让服务器去下载、选示例基因组。

| 输入 | 说明 |
|---|---|
| 蛋白 FASTA、CDS FASTA | 直接分析 |
| GenBank、EMBL | 用其中的基因注释；带序列时还会做组成和基因岛分析 |
| GFF3 | 基因组序列可以内嵌，也可以作为第二个文件上传 |
| 基因组 contig | 自动预测基因：原核用 pyrodigal，真核用 MetaEuk；没有指定基因组类型时，先抽一部分基因比对来判断 |
| 压缩文件 | gz、bz2、xz |

单个文件上限 2 GB。可选设置：

- **自身类群**：基因组所属的属，比对时排除它自己的命中。默认自动推断；参考库里没有同属物种时请手动填写。
- **本谱系的范围**：默认原核按门，可以改窄（例如到科），这样同一门内其他科来源的基因也会被报告。
- **结构检索**：网页和接口默认开启，命令行默认关闭。

### 命令行

在服务器的仓库目录里运行，服务需已启动：

```bash
SEEDDRIFT_DATA=$PWD/store SEEDDRIFT_TOOLS_URL=http://127.0.0.1:8601 SEEDDRIFT_ESMFOLD_URL=http://127.0.0.1:8602 .mamba/envs/sd-app/bin/python -m seeddrift genome proteins.faa --out result/
```

三个变量分别是数据目录、工具服务和 ESMFold 服务的地址；在配置里改过数据目录或端口的话，这里也要跟着改。

| 选项 | 作用 |
|---|---|
| `--self-taxon NAME` | 指定基因组自身的类群（名称或 NCBI taxid），默认自动推断 |
| `--home-rank RANK` | 本谱系的范围：`auto`、`genus`、`family`、`order`、`class`、`phylum`、`kingdom`、`domain` |
| `--domain prokaryote` / `eukaryote` | 输入是未注释 contig 时指定用哪种基因预测 |
| `--structure` | 对没有外部同源的基因做结构检索 |
| `--no-deep`、`--no-trees` | 跳过深度检索、跳过建树 |

所有子命令见 `python3 -m seeddrift --help`。

### HTTP 接口

网页用的就是这套接口。`localhost:8600` 指服务器本机，或经 SSH 隧道转发后的本地端口。

| 方法与路径 | 作用 |
|---|---|
| `GET /api/health` | 服务与数据状态 |
| `GET /api/demos` | 示例基因组列表 |
| `POST /api/jobs` | 提交任务，返回 `{"id": "..."}`。表单字段：`file`、`url`、`demo` 三选一提供输入，`file2` 是 GFF3 对应的基因组 FASTA；`name`、`self_taxon`、`domain`（`auto` / `prokaryote` / `eukaryote`）、`home_rank`；`deep`、`trees`、`structure`、`repeat_filter`（`true` / `false`，默认都是 `true`）；`max_structures`（默认 400） |
| `GET /api/jobs` | 任务列表 |
| `GET /api/jobs/{id}` | 任务状态与摘要；`status` 为 `done` 或 `failed` 时结束 |
| `DELETE /api/jobs/{id}` | 删除任务及其结果；运行中的任务不能删除 |
| `GET /api/jobs/{id}/genes?category=mammal,prokaryote` | 逐基因结果（JSON），可按类别筛选，多个类别用逗号分隔 |
| `GET /api/jobs/{id}/files/{name}` | 下载结果文件，例如 `genes.tsv`、`trees/<基因>.nwk` |

```bash
curl -F file=@proteins.faa -F name=my-genome -F structure=false http://localhost:8600/api/jobs
```

任务按提交顺序一次运行一个。

## 准确性

用四组带真值的构造数据测过：两组蛋白掺入、一组 DNA 移植、一组污染组装。掺入和移植的外源基因都来自参考库没有收录的物种。完整表格在 [docs/benchmark-results.md](docs/benchmark-results.md)。

| 测试 | 召回率 | 阴性对照被判外来 | 高置信档 |
|---|---|---|---|
| 大肠杆菌，掺入 635 个外源蛋白 | 89% / 86% / 76% / 48% | 0 / 60 | 精确率 98%，召回率 56% |
| 溶组织内阿米巴，掺入 635 个外源蛋白 | 92% / 92% / 86% / 48% | 13%（高置信档 0.6%） | 精确率不低于 56%，召回率 58% |
| 嗜肺军团菌，DNA 层面移植 240 个基因 | 82% / 92% / 93% | 0 / 48 | 精确率 89%，召回率 72% |
| 微孢子虫组装，混入 6 段细菌 contig | — | 真核 contig 误标 0 | 6 段污染全部识别，其上的候选无一进入高置信档 |

怎么读这张表：

- **召回率**：被找回并归对来源类别的外源基因所占的比例。前两行的四个数依次对应外源蛋白被随机替换 0%、30%、50%、70% 位点后的结果，替换越多越接近年代久远、已经高度分化的转移。第三行的三个数对应插入基因的密码子用法向受体同化的三个程度；同化只换同义密码子，不改变蛋白序列，所以召回率不随它下降。
- **阴性对照**：受体自身的核糖体蛋白和 RNA 聚合酶亚基，它们不应被判为外来。
- **高置信档的精确率**：把所有不是人为加入的基因都算作阴性。真实基因组里本来就有转移基因，所以它是下限。
- 置信度的阈值是参照前两组数据定的，这两组的数字因此偏乐观；后两组没有参与调参。

## 适用范围与局限

- **给出的是候选，不是定论。** 筛查以序列相似性为主，不能完全区分转移、古老保守、谱系特异丢失和参考库污染。重要的候选请结合系统发育树和基因组上下文判断。
- **默认只报远距离的转移。** 原核基因组默认只报来自其他门的基因；动物、真菌、植物默认只报来自本界以外的基因，原生生物只报来自所在真核超类群以外的基因。更近的转移要把本谱系的范围改窄（例如到科）才会出现。
- **准确性数字只覆盖远距离的转移。** 上面的测试用例都是来自本谱系以外的外源基因。原噬菌体、基因组岛、同一门内的转移等场景还没有评估，表里的数字不能外推到这些场景。
- **尚未用文献报道的真实转移验证，也没有和其他工具（如 HGTector、AvP）比较。**
- **高度分化的外源基因容易漏掉。** 测试中，序列被随机替换掉 70% 位点的外源蛋白只找回约一半。
- **采样稀疏、演化快的真核谱系误报偏高**，例如阿米巴；这类基因组请以高置信档为主。
- **未注释的真核基因组**用 MetaEuk 预测基因，没有任何已知同源蛋白的基因不会被预测出来。
- **参考库里没有同属物种的基因组**：自身类群是根据与参考库几乎完全相同（一致度 ≥ 95%）的命中自动推断的，没有同属物种时推断不可靠，请手动指定。
- **污染检查只对真核基因组做**，检查的是混入的细菌 contig。
- **结构检索**只对没有外部同源、长度在 50 到 1,000 个氨基酸之间的基因做，每个任务最多 400 个；相似的折叠说明的是折叠或功能，不代表来源。

## 工作原理

每个基因依次经过：

1. **比对参考库**：用 DIAMOND 比对约 4,000 万条参考蛋白（约 8,500 个 UniProt 参考蛋白组：细菌、古菌、病毒和哺乳动物每属一个，其他真核每科一个）。
2. **深度检索**：对没有外部同源的基因，关闭重复屏蔽再用高灵敏度模式检索一次（有 GPU 时用 MMseqs2，否则用 DIAMOND）。
3. **重复过滤**：去掉主要落在重复结构（Pfam 的重复型家族、串联重复、卷曲螺旋）上、且一致度低于 40% 的命中。
4. **判定类别**：计算 Alien Index，AI = ln(本谱系最佳命中的 E 值) − ln(本谱系以外最佳命中的 E 值)。外部命中比本谱系命中好得越多，AI 越大；判为外来要求 AI ≥ 15，即外部命中的 E 值约小 300 万倍。
5. **本谱系补搜**：对没有本谱系命中的候选，直接到本谱系的参考蛋白组里再搜一次，避免采样稀疏的谱系被误判。
6. **建树验证**：对外来候选用 MAFFT 和 FastTree 建树，看最近的亲缘是否在本谱系之外；树里至少要有 2 条本谱系序列才给结论。
7. **补充证据**：有编码序列时检查 GC 和 GC3 偏离；有基因坐标时检查外来候选是否相邻成簇（基因岛）；真核基因组还会检查所在 contig 是否像细菌污染。

## 配置

设置放在 `deploy/seeddrift.env`。运行 `up` 之前先复制模板再编辑；不改的话 `up` 会按默认值自动生成：

```bash
cp deploy/seeddrift.env.example deploy/seeddrift.env
```

可以设置数据目录、线程数、端口、代理、是否复用系统自带的 PyTorch。国内网络可以用镜像：`SD_CONDA_MIRROR=tuna`，`HF_ENDPOINT=https://hf-mirror.com`。

| 资源 | 需求 |
|---|---|
| 磁盘 | 下载约 60 GB，建好索引后共约 120 GB，另需临时空间 |
| 显存 | ESMFold 服务常驻约 8 GB；深度检索的 GPU 版 MMseqs2 另外占用一些，不可用时自动改在 CPU 上用 DIAMOND |
| 内存 | 可用内存超过 64 GB 时检索明显更快（DIAMOND 自动换用更大的分块，160 GB 和 400 GB 以上再各快一档）；更小的内存用 DIAMOND 的默认分块 |

已经下载过的数据（另一份 SeedDrift 的数据目录，或 AlphaFold Swiss-Prot、Pfam-A、Swiss-Prot、NCBI taxdump、ESMFold 权重这些标准文件）可以链接过来，不必重下。数据集名称和各自需要的文件见 `data/datasets.json`：

```bash
SEEDDRIFT_DATA=$PWD/store python3 -m seeddrift data link alphafold-swissprot --from /path/to/afdb
```

之后照常运行 `./seeddrift.sh up`，已有的部分会跳过。

也可以用 Docker 代替 `./seeddrift.sh up`（需要 NVIDIA Container Toolkit）。第一条命令把数据下载并索引到 `store/`，第二条构建并启动三个服务；接口和网页同样在本机的 8600 端口：

```bash
docker compose -f deploy/docker/compose.yml --profile data run --rm data
```

```bash
docker compose -f deploy/docker/compose.yml up -d --build
```

服务管理：`./seeddrift.sh start|stop|restart|status [api|tools|esmfold]`。

## 架构

```
   你的电脑                         服务器
 ┌──────────────┐   SSH 隧道   ┌────────────────────────────────────────────────────┐
 │ 浏览器       │ ───────────▶ │ 接口服务 :8600   网页、任务队列、判定、证据汇总    │
 │ localhost:   │              │      ├──▶ 工具服务 :8601   DIAMOND · MMseqs2 ·     │
 │ 8600         │              │      │                     Foldseek · Pfam · 建树  │
 └──────────────┘              │      └──▶ ESMFold 服务 :8602（GPU）                │
                               │                参考数据 $SEEDDRIFT_DATA            │
                               └────────────────────────────────────────────────────┘
```

| 目录 | 内容 |
|---|---|
| `seeddrift/` | 算法：单基因组筛查、重复过滤、多基因组扫描、结构检索 |
| `services/` | 三个 HTTP 服务：接口、工具、ESMFold |
| `web/` | 纯静态网页，不需要构建 |
| `data/` | 每个数据集的来源、许可证和获取方式；数据本身不在仓库里 |
| `deploy/` | 运行环境（锁定版本）、Docker 文件、配置模板 |
| `tests/` | 单元测试 |
| `bench/`、`docs/` | 基准测试的脚本、结果和报告 |
| `examples/` | 多基因组扫描的示例脚本 |
| `research/` | 实验代码，不属于当前发布的功能 |

## 版本与路线图

- **当前版本**：[v0.1.0](https://github.com/eight-acres-lab/SeedDrift/releases/tag/v0.1.0)。`main` 分支在它之后只多了一处修复（ESMFold 服务的显存占用不再随任务增长），部署请用 `main`。
- **计划中**：在原噬菌体、基因组岛、同一门内的转移等更难的场景上评估准确性；用学习模型改进判定。相关代码在 `bench/hard/` 和 `research/`，还在开发，不属于当前发布的功能。

## 开发

```bash
./seeddrift.sh test
```

基准结果可以重新生成：先按 `bench/benchmark.json` 里每个数据集的 `build` 命令生成测试数据，再在部署了服务的机器上运行 `bench/run_benchmark.py`，它会重写 `docs/benchmark-results.md`。

欢迎通过 [issue](https://github.com/eight-acres-lab/SeedDrift/issues) 反馈问题和建议，也欢迎提交 pull request。报告判定问题时，附上输入的来源和 `genes.tsv` 里相关的行会很有帮助。

## 数据与许可

代码以 [MIT](LICENSE) 许可发布。

SeedDrift 不随仓库分发任何第三方数据、模型或软件，它们由脚本在运行的机器上从原始来源获取，各有自己的许可证：UniProt 参考蛋白组和 Swiss-Prot（CC BY 4.0）、AlphaFold DB（CC BY 4.0）、Pfam（CC0）、NCBI Taxonomy、ESMFold（MIT），以及 DIAMOND、MMseqs2、Foldseek、MAFFT、FastTree、pyrodigal、MetaEuk 等工具。完整清单、版本和引用方式见 [REFERENCES.md](REFERENCES.md)，使用时请遵守并引用它们。
