# SPO (Spinning Parallelogram Operator) - Python Implementation

This repository provides a Python implementation of the SPO (Spinning Parallelogram Operator) method for light field depth estimation. The main pipeline includes EPI construction, cost-volume generation, reliability-based horizontal/vertical cost fusion, and guided-filter post-processing.

## Attribution and Citation

This repository is a Python reimplementation based on the original MATLAB version:
[`shuozh/Spinning-Parallelogram-Operator`](https://github.com/shuozh/Spinning-Parallelogram-Operator).

Please note the distinction:

- The original algorithm, paper, and MATLAB implementation are from Shuo Zhang et al.
- This repository is not an official Python release by the original authors.
- If you use this repository or the SPO algorithm in research, please cite the original paper and mention that this is a Python reimplementation based on the MATLAB code.

The original repository states that research use should cite the CVIU 2016 paper, and commercial use should contact the original authors.

### Original Paper

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

### Suggested Acknowledgement

```text
This repository is a Python reimplementation of the SPO method based on the
original MATLAB code released by Zhang et al. We cite the original CVIU 2016
paper for the algorithm and acknowledge the upstream repository:
https://github.com/shuozh/Spinning-Parallelogram-Operator
```

## Repository Structure

```text
SPO-Python-cleaned/
|-- config.json                 # Global runtime parameters
|-- main.py                     # Entry point
|-- spo.py                      # Main SPO pipeline
|-- full_to_epi.py              # Light field mosaic to EPI conversion
|-- depth_integration.py        # Cost fusion and guided-filter post-processing
|-- optimization/               # Guided filter and related modules
|-- input/
|   `-- boxes/
|       |-- lf.png              # Input light field mosaic image
|       `-- depth_opt.py        # Scene-specific disparity range and view count
`-- result/
    `-- boxes/                  # Output directory
```

## Dependencies

CPU mode:

```bash
pip install numpy pillow scipy
```

GPU acceleration is optional. If CuPy is installed, the code will use it automatically when `use_gpu` is enabled:

```bash
pip install cupy-cuda12x
```

Install the CuPy package that matches your CUDA version.

## Usage

1. Prepare input data:
   - Put the light field mosaic image at `input/boxes/lf.png`.
   - Set the scene-specific `disp_min`, `disp_max`, and `NumView` in `input/boxes/depth_opt.py`.
2. Edit `config.json`:
   - This file stores global runtime parameters such as `scale`, `bins`, `nD`, and `sigma`.
   - Guided filter parameters, `guided_filter_radius` and `guided_filter_eps`, are also configured here.
3. Run the program:

```bash
python main.py
```

The program reads:

- `config.json`: global runtime parameters
- `input/boxes/depth_opt.py`: scene-specific disparity range and number of views

## Parameter Sources

### `config.json`

These global runtime parameters should be edited directly in `config.json`:

- `scale`: SPO window-width parameter
- `bins`: number of histogram bins
- `nD`: number of depth labels
- `sigma`: reliability-fusion parameter for horizontal and vertical EPI costs
- `guided_filter_radius`: guided filter radius
- `guided_filter_eps`: guided filter regularization term
- `use_gpu`: whether to prefer GPU acceleration

Default configuration:

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

The following parameters remain scene-specific and are loaded from each scene's `depth_opt.py`:

- `disp_min`
- `disp_max`
- `NumView`

In other words, the disparity range is not read from `config.json`; it is read from the current light field scene configuration.

## Outputs

After running the program, outputs are written to `result/boxes/`:

- `depth_initial.bmp`: initial depth map before guided filtering
- `depth_filtering.bmp`: depth map after guided filtering

## Algorithm Pipeline

1. Build horizontal and vertical EPIs from the light field mosaic.
2. Generate a matching cost volume over discrete depth labels.
3. Estimate reliability weights for horizontal and vertical directions.
4. Fuse the directional cost volumes and produce an initial depth map.
5. Apply guided filtering to each depth-cost slice.
6. Save the filtered depth result.

## Notes

- The default entry point processes `input/boxes/`.
- If `use_gpu` is `true` but CuPy is not installed, the program automatically falls back to CPU mode.
- `nD` has a large impact on runtime and memory usage.
- Configure `disp_min` and `disp_max` according to the comments in each scene's `depth_opt.py`.
