# SPO Parameter Configuration Guide

## Parameter Scope

The parameters are divided into two groups:

1. `config.json`
   - Stores global runtime parameters.
   - Users can edit this file directly.
2. `input/<scene>/depth_opt.py`
   - Stores scene-specific geometric parameters.
   - The disparity range is still loaded from this file, not from `config.json`.

Default `config.json`:

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

## Parameters Configured in `depth_opt.py`

These parameters are tied to each specific light field scene:

- `disp_min`: minimum scene disparity
- `disp_max`: maximum scene disparity
- `NumView`: number of angular views per dimension

Different light field scenes usually have different disparity ranges and angular sampling. Keeping these parameters in each scene folder reduces the risk of running a scene with the wrong geometry.

## Parameters Configured in `config.json`

### 1. `scale`

Purpose: controls the width of the SPO weighting window.

Suggested values:

- Rich texture and fine details: `0.6 ~ 1.0`
- General scenes: `1.0 ~ 1.5`
- Smooth regions or stronger noise: `1.5 ~ 2.0`

Symptoms and adjustment:

- Blurry depth boundaries: decrease `scale`
- Noisy or unstable depth maps: increase `scale`

### 2. `bins`

Purpose: controls the quantization precision of the color histogram.

Suggested values:

- Rich color and texture variations: `128 ~ 256`
- General scenes: `64 ~ 128`
- Strong noise or low color diversity: `16 ~ 64`

Symptoms and adjustment:

- Color information is underused: increase `bins`
- Too sensitive to image noise: decrease `bins`

### 3. `nD`

Purpose: number of depth labels. This directly affects depth resolution and computational cost.

Suggested values:

- Fast testing: `32 ~ 64`
- Regular use: `96 ~ 128`
- Higher precision: `160 ~ 256`

Symptoms and adjustment:

- Visible stair-step artifacts in the depth map: increase `nD`
- Runtime is too long or GPU memory is insufficient: decrease `nD`

This is one of the most important runtime parameters.

### 4. `sigma`

Purpose: controls the reliability fusion strength between horizontal and vertical EPI costs.

Suggested values:

- Strong directional texture: `0.2 ~ 0.4`
- Balanced horizontal and vertical textures: `0.4 ~ 0.7`
- Weak texture directionality and more stable fusion: `0.7 ~ 1.0`

Symptoms and adjustment:

- One directional texture dominates strongly: keep `sigma` relatively small
- Horizontal and vertical costs should be blended more smoothly: increase `sigma`

### 5. `guided_filter_radius`

Purpose: spatial radius of the guided filter. It controls the smoothing range during post-processing.

Suggested values:

- Preserve details and boundaries: `5 ~ 10`
- General scenes: `10 ~ 15`
- Stronger smoothing and denoising: `15 ~ 20`

Symptoms and adjustment:

- Depth boundaries are oversmoothed: decrease `guided_filter_radius`
- Depth map has many noisy points or holes: increase `guided_filter_radius`

### 6. `guided_filter_eps`

Purpose: regularization term of the guided filter. It balances edge preservation and smoothing strength.

Suggested values:

- Stronger edge preservation: `1e-5 ~ 5e-5`
- General scenes: `1e-4 ~ 5e-4`
- Stronger smoothing: `1e-3 ~ 1e-2`

Symptoms and adjustment:

- Boundaries should be sharper: decrease `guided_filter_eps`
- Boundaries have artifacts or noise: increase `guided_filter_eps`

### 7. `use_gpu`

Purpose: whether to prefer GPU acceleration.

Behavior:

- `true`: try to use CuPy acceleration
- `false`: force CPU mode
- If `true` but CuPy is unavailable, the program automatically falls back to CPU mode

## Recommended Starting Configurations

### General Scene

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

### High-Texture Scene

```json
{
  "scale": 1.0,
  "bins": 256,
  "nD": 128,
  "sigma": 0.3,
  "guided_filter_radius": 7,
  "guided_filter_eps": 0.00005,
  "use_gpu": true
}
```

### High-Noise Scene

```json
{
  "scale": 1.0,
  "bins": 32,
  "nD": 64,
  "sigma": 0.3,
  "guided_filter_radius": 10,
  "guided_filter_eps": 0.0001,
  "use_gpu": true
}
```

For high-noise scenes, `bins` can also be reduced to `16` when the color histogram is too sensitive to noise.

## Recommended Tuning Order

1. First set `disp_min`, `disp_max`, and `NumView` correctly in `depth_opt.py`.
2. Tune `nD` in `config.json`.
3. Tune `scale` and `sigma` according to the visual result.
4. Finally tune `guided_filter_radius` and `guided_filter_eps`.

This order is safer because an incorrect disparity range cannot usually be fixed by tuning global runtime parameters.
