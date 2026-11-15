# SeedDrift

**看看基因组里哪些基因像是从别的田里飘来的种子。**

SeedDrift 检测水平基因转移（HGT）：给一个基因组，找出哪些基因可能来自哺乳动物、来自原核生物，或来自其他远缘谱系。给一组近缘基因组，还能做基因树与物种树调和，配合共线性、移动元件和组成偏离，给出全基因组的转移事件和 P(HGT)。

本仓库包含全部代码、算法、服务和网页。数据和第三方工具都不在仓库里，由脚本在运行的机器上从原始来源获取，来源与许可证见 [REFERENCES.md](REFERENCES.md)。

## 结构

```
   你的电脑                         GPU 服务器（或任何一台 Linux 机器）
 ┌──────────────┐   SSH 隧道   ┌──────────────────────────────────────────────────────────┐
 │ 网页 web/    │ ───────────▶ │ SeedDrift API :8600   自己的算法：任务、分类、AI、重复过滤 │
 │ localhost:   │              │      │                                                    │
 │ 8700         │              │      ├──▶ 工具服务 :8601   DIAMOND · Foldseek · Pfam · COILS │
 └──────────────┘              │      └──▶ ESMFold 服务 :8602（GPU）                        │
                               │                    数据 $SEEDDRIFT_DATA（约 45 GB）         │
                               └──────────────────────────────────────────────────────────┘
```

| 部分 | 位置 | 内容 |
|---|---|---|
| 自己的算法 | `seeddrift/` | 单基因组筛查（`single.py`、`pipeline.py`）、重复过滤（`repeats.py`）、多基因组 HGT 扫描（`scan.py`、`reconcile.py` 等）、供体检索、结构检索 |
| 依赖服务 | `services/tools`、`services/esmfold` | 第三方工具，以 HTTP 服务的形式对外，绑定数据 |
| 算法服务 | `services/api` | 任务队列和结果接口；网页只和它通信 |
| 网页 | `web/` | 纯静态页面，不需要构建 |
| 数据 | `data/datasets.json` | 每个数据集的来源、许可证、获取与索引方式；数据本身放在 `$SEEDDRIFT_DATA` |
| 部署 | `seeddrift.sh`、`deploy/` | 一条命令安装和启动；micromamba 环境（锁定版本）、Docker compose |

## 一条命令复刻

在一台有 NVIDIA GPU（CUDA 12 驱动）的 Linux 机器上：

```bash
git clone https://github.com/eight-acres-lab/SeedDrift.git && cd SeedDrift && ./seeddrift.sh up
```

`up` 依次完成以下四步，每一步都可以单独执行，中断后重跑会接着做：

1. `install`：用 micromamba 建三个环境，即 `sd-app`、`sd-tools`、`sd-esmfold`，版本锁定在 `deploy/envs/`。
2. `data`：从原始来源下载约 45 GB 数据并建索引。数据全部下载到这台机器上；参考蛋白组要逐个下载约 5,300 个，需要几个小时。
3. `start`：启动三个服务，只监听 127.0.0.1。
4. `status`：检查服务和数据。

需要改的设置放在 `deploy/seeddrift.env` 里，第一次运行时会从 `deploy/seeddrift.env.example` 复制一份：

- 数据目录、线程数、端口；
- 国内用清华镜像：`SD_CONDA_MIRROR=tuna`，`HF_ENDPOINT=https://hf-mirror.com`；
- 代理；
- 复用系统自带的 PyTorch：`SD_ESMFOLD_TORCH=system`。

如果机器上已经有这些数据，不用重新下载，可以直接链接过来：

```bash
.mamba/envs/sd-tools/bin/python -m seeddrift data link reference-proteomes --from /path/to/ref
.mamba/envs/sd-tools/bin/python -m seeddrift data status
```

有 Docker 的机器（需要 NVIDIA Container Toolkit）：

```bash
docker compose -f deploy/docker/compose.yml --profile data run --rm data
docker compose -f deploy/docker/compose.yml up -d --build
```

## 在自己的电脑上打开网页

网页只需要系统自带的 `python3`，不用装别的东西。

1. 把远程 API 转到本地 8600 端口：

   ```bash
   ./seeddrift.sh tunnel -p 31902 root@your-gpu-host
   ```

2. 再开一个终端，打开网页：

   ```bash
   ./seeddrift.sh web
   ```

浏览器访问 http://localhost:8700。

网页上可以：
- 上传文件：
  - 蛋白 FASTA 或 CDS FASTA；
  - GenBank / EMBL 注释文件；
  - GFF3，基因组序列可以内嵌在里面，也可以作为第二个文件上传；
  - 基因组 contig。原核基因组用 pyrodigal 预测基因，真核基因组用 MetaEuk 预测基因；不确定是哪一类时，会先抽样比对，自动判断；
  - gz、bz2、xz 压缩包都可以；
- 填一个 URL，由服务器去下载，不占你本地的流量；
- 直接选示例基因组。

每个基因经过以下几步：
1. **比对参考库：** 用 DIAMOND 比对约 4,000 万条参考蛋白。
2. **深度检索：** 没有外部同源的基因，关闭重复屏蔽，再用 ultra-sensitive 模式检索一次。DIAMOND 默认的屏蔽会把富含重复的蛋白整个藏起来。
3. **重复过滤：** 去掉落在重复结构上的命中。
4. **判定类别：** 按 Alien Index 归类。判为外来要求 AI ≥ 15，也就是外部命中的 E 值比本谱系好约 300 万倍；得分接近平手不算证据。
5. **本谱系补搜：** 对没有本谱系命中的候选，直接到本谱系的参考蛋白组里再搜一次。参考库有 4,000 万条序列，采样稀疏的谱系（例如阿米巴所在的 Amoebozoa）里的近亲，常常排不进前 100 个命中。
6. **建树验证：** 对外来候选，用 MAFFT 和 FastTree 建树，看最近的亲缘是不是在本谱系之外。树里至少要有 2 条本谱系参考序列，否则不给结论。
7. **补充证据：** 有基因组序列时，再看 GC 和 GC3 是否偏离、相邻基因里是否成簇（基因组岛），以及所在 contig 是否像污染。

每个基因会被归入以下几类之一：

| 判定 | 含义 |
|---|---|
| 疑似哺乳动物来源 / 疑似原核来源 / 疑似其他真核来源 / 疑似病毒来源 | Alien Index ≥ 15：最近的命中明显在本谱系之外。供体类群取本谱系以外前 10 个命中，按 bit 分加权得出 |
| 本谱系 | 最近的命中在自己所在的谱系内。原核按门；真核按动物、真菌、植物，原生生物按所在的真核超类群（如 Amoebozoa）。范围可以在网页上改窄，例如改成“科”，这样同一门内其他科来的基因也会被判出来 |
| 外部无同源 | 排除自身所在的属之后，深度检索也没有命中。可以选择对这类基因做结构检索（ESMFold + Foldseek） |
| 仅重复结构命中 | 主要命中都落在重复结构上（Pfam 重复型家族、串联重复、卷曲螺旋），且一致度低于 40%：不做判定 |

外来候选还会给出置信度：

| 置信度 | 条件 |
|---|---|
| 高 | AI ≥ 45，至少有两个供体物种；并且补搜后本谱系仍然没有同源，或建树支持外来且一致度比本谱系高 ≥ 10 个百分点 |
| 中 | AI ≥ 45 或建树支持外来；建树没有落在本谱系；并且本谱系没有同源，或一致度高 ≥ 5 个百分点 |
| 低 | 其余候选，以及位于疑似污染 contig 上的候选 |

这些阈值是在掺入数据上定的。实际使用时建议以“高”和“中”两档为主，结果见 [docs/benchmark-results.md](docs/benchmark-results.md)。

“支持物种”一栏只有 1 时要小心。这类真核来源候选往往是该真核基因组组装混入了细菌序列。

## 命令行

```bash
python3 -m seeddrift genome proteins.faa --out result/ [--self-taxon Legionella] [--structure]
python3 -m seeddrift serve api|tools|esmfold
python3 -m seeddrift data status|fetch|build|link
python3 -m seeddrift prepare / scan / donors / structure ...
```

最后一行是多基因组 HGT 扫描，说明见 [docs/hgt-scan.md](docs/hgt-scan.md)。

`genome` 命令设了 `SEEDDRIFT_TOOLS_URL` 和 `SEEDDRIFT_ESMFOLD_URL` 时调用这两个服务，没有设就在本进程里直接运行工具。

## 测试与基准

```bash
./seeddrift.sh test
```

上面这条是单元测试。下面是基准测试，在服务器上运行，数据由脚本从原始来源下载，真值表与数据放在一起：

```bash
python bench/spikein.py --recipient Bacteria/UP000000625_83333 --out bench/work/spike_ecoli
python bench/contamination.py --out bench/work/contam_ecuniculi
python bench/run_benchmark.py --api http://127.0.0.1:8600
```

- `spikein.py`：在真实受体蛋白组里掺入已知来源的外源蛋白，按 0%、30%、50%、70% 的位点替换率设置难度，供体选参考库没收录的物种。
- `contamination.py`：往真核组装里混入细菌 contig。
- `run_benchmark.py`：跑完整流程和各项消融，结果写进 [docs/benchmark-results.md](docs/benchmark-results.md)。

## 局限

- 单基因组筛查只看相似性，不建树。它给出的是候选，不能区分转移、古老保守、谱系特异丢失和参考库污染。
- 自身类群默认由近乎相同的最佳命中推断。如果基因组属于参考库中没有的新属，推断会退到科，这时应手动指定。
- 结构检索只对“外部无同源”的基因做，相似的折叠说明的是折叠或功能，不是供体。
- 参考库每科或每目只取一个代表，近缘供体会被归到它所在的更大类群。

## 许可

[MIT](LICENSE)。第三方数据、模型和软件各自的许可证见 [REFERENCES.md](REFERENCES.md)。
