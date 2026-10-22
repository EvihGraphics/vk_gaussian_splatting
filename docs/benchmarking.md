# Profiling and Benchmarking

To properly assess the performance of the pipelines, you should **deactivate vertical synchronization (V-Sync)**. This can be done from the **View > V-Sync** menu, the **Renderer Properties** panel, or the **Profiler** window. If V-Sync is not deactivated, the system does not run at optimal performance, and the reported timings in the **Window Top Bar** and **Profiler** window are generally higher (and fps lower) than the achievable performance. The V-Sync option is enabled by default to save energy.

The system also provides a means to run automatic benchmarks, which are detailed in the **Performance Results** sections of the documentation sub-pages. In Benchmarking mode, V-Sync is automatically disabled.

## Automated benchmarks (`benchmark.py`)

The repository includes `benchmark.py` to drive the application through scripted parameter sequences (`.cfg` files), collect timing logs, and—for some datasets—generate PSNR/FPS charts and comparison images against a raster reference.

### Python dependencies

Install the required packages:

``` sh
python -m pip install matplotlib numpy pillow
```

### Running a benchmark

Build the application first (see [Getting Started](getting-started.md#building-from-source)), then run from the repository root:

``` sh
python benchmark.py <benchmark.cfg> <dataset> <dataset_path> <output.csv> [options]
```

Example (billboard bounding-mode study):

``` sh
python benchmark.py benchmark_billboards.cfg 3DGS_BILLBOARDS <path_to_vkgs_dataset> benchmark_billboards.csv --headless
```

Other dataset-specific configuration files include `benchmark_3dgs.cfg`, `benchmark_3dgrt.cfg`, and `benchmark_3dgut.cfg`. Their usage and published charts are described in the **Performance Results** sections of the corresponding [deep dives](index.md).

### Useful options

| Flag | Description |
|------|-------------|
| `--headless` | Run without a window (recommended for batch benchmarks) |
| `--verbose` / `-v` | Stream application output to the console |
| `--forcegpu ID` | Forward `--forcegpu` to the app (Vulkan physical device index) |
| `--skip-exec` | Skip launching the app; re-analyze existing logs and screenshots in `_benchmark/` |

Outputs are written under `_benchmark/` (logs, screenshots, CSV, and PNG charts).

For `benchmark_billboards.cfg` / `3DGS_BILLBOARDS`, charts are grouped by sort strategy and geometry:

| Charts | Sort | Geometry |
|--------|------|----------|
| `14–17_icosa_*` | Stochastic any-hit | Icosahedron billboards |
| `14–17_aabb_*` | Stochastic any-hit | AABB billboards |
| `18–21_aabb_allpass_*` | All-pass | AABB billboards (stochastic AABB modes except NoShorten; icosa does not support all-pass) |

All-pass timing sequences use `--sequenceframes 64` / `--sequenceaverages 32` / `--sequenceresetframes 0` so the profiler writes `Timeline` lines into the log. Shorter sequences (for example 5 frames) produce empty `ParameterSequence` blocks and the all-pass charts are skipped.

`13_summary_table.csv` lists timing and PSNR for all modes.

### Gaussian Point Splatting (CUDA) benchmark

`benchmark_gps.cfg` / dataset `GPS` compares the Gaussian Point Splatting CUDA pipeline (`--pipeline 6`, requires a build configured with `-DUSE_GPS_CUDA=ON`) with the Vulkan 3DGS mesh-shader rasterizer (GPU radix sort, and stochastic splat) on the INRIA 3DGS scenes:

``` sh
python benchmark.py benchmark_gps.cfg GPS <path_to_3dgs_dataset_root> benchmark_results_gps.csv --headless
# optional: --scenes bonsai,bicycle to run a subset
```

Both the nested (`bicycle/bicycle/point_cloud/...`) and flat (`bicycle/point_cloud/...`) INRIA dataset layouts are accepted.

GPU timings mix two clocks in the same profiler log:

* `GPS CUDA` (GPS algorithm) and `GPS CUDA output` (copy into the shared buffer): **CUDA events** recorded around the GPS work, reported in the GPU column of the profiler through a dedicated time provider. `--gpsIsolateGpuTiming 1` (set by the cfg) waits for the Vulkan queue before the CUDA work, so that work from the previous frame does not overlap the measurement.
* All other timers (`GPU Dist`, `GPU Sort`, `Rasterization`, `GPS resolve`, `GPS depth`, ...): Vulkan timestamps.

The `Frame` GPU time therefore covers the full GPU work of every pipeline. Note that the CUDA event span includes the short GPU idle gaps caused by the host synchronizations inside GPS (point-count readbacks between its kernels): an Nsight Systems trace of bonsai at 1920x1280 measured a 10.24 ms event span for 9.32 ms of GPU busy time (about 9% gaps).

Outputs (in `_benchmark/`):

| File | Content |
|------|---------|
| `10_gps_histogram_splatting_gpu_time.png` | Splatting GPU time: dist + sort + raster for Vulkan, CUDA + resolve/depth copies for GPS |
| `11_gps_histogram_frame_gpu_time.png` | Full frame GPU time (includes temporal accumulation for stochastic pipelines) |
| `12_gps_histogram_fps.png` | Frame rate from the CPU frame time (GPS runs synchronously) |
| `13_gps_summary_table.csv` | Per scene and pipeline: GPU/CPU times, fps, splatting speed-up and PSNR vs. the sorted mesh reference |
| `*_compare_<pipeline>_tiled.png` | Reference / test / difference images (screenshots after 512 accumulated frames) |

Each rendering sequence stays below the 1000-frame temporal accumulation limit: once converged, accumulating pipelines stop rendering and would no longer produce timings.
