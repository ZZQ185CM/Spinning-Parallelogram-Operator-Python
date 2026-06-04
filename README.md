# SPO (Spinning Parallelogram Operator) - Python Implementation

这是一个用于光场深度估计的 SPO 算法 Python 版本，核心流程包括 EPI 构建、代价体生成、水平/垂直方向可靠性融合，以及 guided filter 后处理。

## 来源说明与引用

本仓库是基于原始 MATLAB 版本
[`shuozh/Spinning-Parallelogram-Operator`](https://github.com/shuozh/Spinning-Parallelogram-Operator)
整理和重写的 Python 实现，用于在 Python / NumPy / SciPy / CuPy 环境下复现和使用 SPO
(Spinning Parallelogram Operator) 光场深度估计算法。

需要区分的关系如下：

- 原始算法、论文和最早公开代码来自 Shuo Zhang 等作者
- 本仓库不是原作者发布的官方 Python 版本，而是基于其 MATLAB 实现思路改写的 Python 版本
- 如果你的科研工作使用了本仓库或对应算法，建议在论文正文或项目说明中明确说明：本实现是基于原始 MATLAB 代码重写的 Python 版本

原始仓库 README 中明确建议：科研使用应引用其 CVIU 2016 论文；商业使用请联系原作者。

### 建议引用的原始论文

Shuo Zhang, Hao Sheng, Chao Li, Jun Zhang and Zhang Xiong.  
Robust depth estimation for light field via spinning parallelogram operator.  
Computer Vision and Image Understanding, 145(C):148-159, 2016.

### BibTeX

```bibtex
@article{Zhang2016Robust,
  title={Robust depth estimation for light field via spinning parallelogram operator},
  author={Zhang, Shuo and Sheng, Hao and Li, Chao and Zhang, Jun and Xiong, Zhang},
  journal={Computer Vision and Image Understanding},
  volume={145},
  pages={148-159},
  year={2016}
}
```

### 推荐致谢写法

如果你需要在论文、报告或仓库中说明来源，可以使用类似表述：

> This repository is a Python reimplementation of the SPO method based on the
> original MATLAB code released by Zhang et al. We cite the original CVIU 2016
> paper for the algorithm and acknowledge the upstream repository:
> https://github.com/shuozh/Spinning-Parallelogram-Operator

## 目录结构

```text
SPO-Python-cleaned/
|-- config.json                 # 全局计算参数
|-- main.py                     # 程序入口
|-- spo.py                      # SPO 主流程
|-- full_to_epi.py              # 光场拼接图到 EPI 的转换
|-- depth_integration.py        # 代价融合与 guided filter 后处理
|-- optimization/               # guided filter 等模块
|-- input/
|   `-- boxes/
|       |-- lf.png              # 输入光场拼接图
|       `-- depth_opt.py        # 当前场景的视差范围和视角数
`-- result/
    `-- boxes/                  # 输出结果
```

## 依赖

CPU 模式：

```bash
pip install numpy pillow scipy
```

GPU 模式可额外安装 CuPy，程序会自动检测：

```bash
pip install cupy-cuda12x
```

请根据你的 CUDA 版本替换为对应的 CuPy 包。

## 使用方法

1. 准备输入数据：
   - 将光场拼接图放到 `input/boxes/lf.png`
   - 在 `input/boxes/depth_opt.py` 中设置该场景的 `disp_min`、`disp_max`、`NumView`
2. 编辑 `config.json`：
   - 这里放的是全局计算参数，例如 `scale`、`bins`、`nD`、`sigma`
   - guided filter 的 `guided_filter_radius`、`guided_filter_eps` 也在这里修改
3. 运行：

```bash
python main.py
```

程序会自动读取：

- `config.json`：全局计算参数
- `input/boxes/depth_opt.py`：当前光场场景的视差范围和视角数

## 参数来源说明

### `config.json`

以下参数是全局运行参数，用户直接编辑这个文件即可：

- `scale`: SPO 窗口宽度
- `bins`: 直方图 bin 数量
- `nD`: 深度标签数量
- `sigma`: 水平/垂直方向可靠性融合参数
- `guided_filter_radius`: guided filter 半径
- `guided_filter_eps`: guided filter 正则项
- `use_gpu`: 是否优先使用 GPU

默认内容如下：

```json
{
  "scale": 1.0,
  "bins": 64,
  "nD": 64,
  "sigma": 0.3,
  "guided_filter_radius": 10,
  "guided_filter_eps": 0.0001,
  "use_gpu": true
}
```

### `input/<scene>/depth_opt.py`

以下参数继续按每个具体光场场景单独配置：

- `disp_min`
- `disp_max`
- `NumView`

也就是说，视差范围仍然不从 `config.json` 读取，而是从每个场景自己的 `depth_opt.py` 读取。

## 输出结果

运行完成后会在 `result/boxes/` 下生成：

- `depth_initial.bmp`: 融合前的初始深度图
- `depth_filtering.bmp`: guided filter 后的深度图

## 算法流程

1. 从光场拼接图中构建水平/垂直 EPI
2. 对不同深度标签生成代价体
3. 计算水平/垂直方向的可靠性权重
4. 融合代价体并得到初始深度图
5. 使用 guided filter 对每个深度切片进行后处理
6. 输出过滤后的深度结果

## 说明

- 当前入口默认处理 `input/boxes/`
- 若 `use_gpu=true` 但环境中没有 CuPy，程序会自动退回 CPU
- `nD` 会显著影响计算时间和显存占用
- `disp_min/disp_max` 的配置方式请以各场景 `depth_opt.py` 中的注释为准
