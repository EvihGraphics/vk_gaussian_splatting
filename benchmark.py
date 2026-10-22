import re
import csv
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os
import sys
import argparse
from collections import defaultdict
from matplotlib.ticker import AutoMinorLocator
import matplotlib.patches as mpatches
import shutil
from PIL import Image, ImageDraw, ImageFont


# ============================================================================
#
#   SECTION 1 — BENCHMARK EXECUTION
#
#   Functions for running the benchmark application, generating multi-camera
#   configuration files, and parsing benchmark log output into structured data.
#
# ============================================================================


def generate_multi_camera_cfg(template_cfg_path, camera_presets, output_dir="."):
    """Generate a temporary cfg file that repeats the test sequences for each camera preset.
    
    IMPORTANT: The template cfg MUST have its first SEQUENCE be a "load/init" block
    (e.g. 'SEQUENCE "Load scene and common settings"') that ensures the project is fully
    loaded before any --activateCameraPreset command is issued. Camera presets stored in
    .vkgs projects are only available AFTER the scene finishes loading, so the activation
    is injected between the first (load) sequence and the remaining (test) sequences.
    
    For each camera preset ID, the generated cfg contains:
      1. The load sequence (first SEQUENCE from template) — ensures scene is loaded
      2. A camera activation sequence — activates the preset by index
      3. All remaining test sequences — with screenshot filenames prefixed by camN_
    
    Args:
        template_cfg_path: Path to the template .cfg file
        camera_presets: List of integer camera preset IDs (e.g. [0, 1, 2])
        output_dir: Directory where the temp cfg is written (default: current dir)
    
    Returns:
        Path to the generated temporary cfg file.
    """
    with open(template_cfg_path, "r") as f:
        template_content = f.read()
    
    # Remove any existing --activateCameraPreset lines from template
    lines = template_content.split("\n")
    lines = [l for l in lines if "--activateCameraPreset" not in l]
    clean_template = "\n".join(lines)
    
    # Strip "_benchmark/" prefix from paths since the temp cfg is written inside _benchmark/
    clean_template = clean_template.replace('"_benchmark/', '"')
    
    # Split into "load" part (first SEQUENCE block) and "test" part (all subsequent sequences).
    # The split point is the start of the SECOND 'SEQUENCE "..."' line.
    match = re.search(r'\n(SEQUENCE\s+")', clean_template[1:])  # skip first char to find 2nd SEQUENCE
    if match:
        split_pos = match.start() + 1
        load_part = clean_template[:split_pos]
        test_part = clean_template[split_pos:]
    else:
        load_part = ""
        test_part = clean_template
    
    # Build the combined cfg: for each camera, emit load + activation + tests
    combined = ""
    for preset_id in camera_presets:
        combined += load_part + "\n"
        
        # Camera activation — runs after scene load so presets are available
        combined += f'SEQUENCE "--- Camera Preset {preset_id}"\n'
        combined += f'--activateCameraPreset {preset_id}\n'
        combined += '--sequenceframes 1\n'
        combined += '--sequenceaverages 1\n'
        combined += '--sequenceresetframes 0\n\n'
        
        # Test sequences with camera-indexed screenshot filenames
        cam_tests = test_part.replace("screenshot_", f"screenshot_cam{preset_id}_")
        combined += cam_tests + "\n\n"
    
    # Write to temp file in the output directory
    temp_cfg = os.path.join(output_dir, "_benchmark_temp.cfg")
    with open(temp_cfg, "w") as f:
        f.write(combined)
    return temp_cfg


def run_benchmark(executable, benchmark_file, scene_path, output_log, camera_presets_path="", verbose=False, headless=False, forcegpu=None):
    """Launch the benchmark executable and capture output to a log file."""
    command = [os.path.abspath(executable), "--size", "1920", "1080", "--benchmark", "1", "--sequencefile", os.path.abspath(benchmark_file)]
    if headless:
        command += ["--headless", "1"]
    if forcegpu is not None:
        command += ["--forcegpu", str(forcegpu)]
    if camera_presets_path:
        command += ["--loadCameraPresets", camera_presets_path]
    command.append(scene_path)
    if verbose:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
        with open(output_log, "w", encoding="utf-8") as log_file:
            for line in process.stdout:
                sys.stdout.write(line)
                log_file.write(line)
        process.wait()
    else:
        with open(output_log, "w", encoding="utf-8") as log_file:
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, shell=True)


def parse_benchmark(log_text, scene_name):
    """Parse a benchmark log file into a list of result dicts with timers and memory."""
    benchmark_pattern = re.compile(r'ParameterSequence\s+(\d+)\s+"([^"]+)"\s*=')
    timer_pattern = re.compile(r'Timer\s+"([^"]+)"\s*;\s*GPU;\s*avg\s+(\d+);.*?CPU;\s*avg\s+(\d+);')
    benchmark_adv_pattern = re.compile(r'BENCHMARK_ADV (\d+) {')
    memory_pattern = re.compile(r'Memory (\w+); Host used\s+(\d+); Device Used\s+(\d+); Device Allocated\s+(\d+);')

    benchmarks = []
    
    # Extract benchmark sections
    benchmark_sections = re.split(benchmark_pattern, log_text)[1:]
    
    benchmark_data = {}
    for i in range(0, len(benchmark_sections), 3):
        benchmark_id = int(benchmark_sections[i])
        benchmark_name = benchmark_sections[i+1].strip()
        benchmark_content = benchmark_sections[i+2]
        
        timers = {}
        for match in timer_pattern.finditer(benchmark_content):
            stage = match.group(1).strip()
            vk_time = float(match.group(2)) / 1000.0
            cpu_time = float(match.group(3)) / 1000.0
            timers[stage] = {'VK': vk_time, 'CPU': cpu_time}

        benchmark_data[benchmark_id] = {
            "scene": scene_name,
            "id": benchmark_id,
            "name": benchmark_name,
            "timers": timers,
            "memory": {}
        }

    # Extract BENCHMARK_ADV sections
    benchmark_adv_sections = re.split(benchmark_adv_pattern, log_text)[1:]

    for i in range(0, len(benchmark_adv_sections), 2):
        benchmark_id = int(benchmark_adv_sections[i])
        benchmark_content = benchmark_adv_sections[i+1]

        memory_data = {}
        for match in memory_pattern.finditer(benchmark_content):
            memory_type = match.group(1).strip()  # "Scene", "Rasterization" or "Raytracing"
            host_used = int(match.group(2))
            device_used = int(match.group(3))
            device_allocated = int(match.group(4))

            memory_data[memory_type] = {
                "Host Used": host_used,
                "Device Used": device_used,
                "Device Allocated": device_allocated
            }
        
        # Attach to corresponding BENCHMARK
        if benchmark_id in benchmark_data:
            benchmark_data[benchmark_id]["memory"] = memory_data

    return list(benchmark_data.values())


def parse_benchmark_multi_camera(log_text, scene_name, camera_presets):
    """Parse benchmark log that contains multiple camera preset runs.
    Splits results by '--- Camera Preset N' markers and labels each group."""
    all_results = parse_benchmark(log_text, scene_name)
    
    # Find the camera marker sequences and assign each result to a camera group
    current_cam_idx = -1
    for result in all_results:
        if result["name"].startswith("--- Camera Preset"):
            current_cam_idx += 1
        if current_cam_idx >= 0 and current_cam_idx < len(camera_presets):
            result["scene"] = f"{scene_name} (Cam {camera_presets[current_cam_idx]})"
    
    # Remove the camera marker entries themselves
    all_results = [r for r in all_results if not r["name"].startswith("--- Camera Preset")]
    return all_results


def save_to_csv(benchmarks, filename="benchmark_results.csv"):
    """Export benchmark results to a CSV file with timers and memory columns."""
    # Collect all possible timer stages
    stages = sorted({stage for b in benchmarks for stage in b["timers"]})
    
    # Define CSV field names
    fieldnames = ["Scene", "Benchmark ID", "Benchmark Name"]
    fieldnames += [f"{stage} VK" for stage in stages] + [f"{stage} CPU" for stage in stages]
    fieldnames += ["Scene Host Used", "Scene Device Used", "Scene Device Allocated"]
    fieldnames += ["Rendering Host Used", "Rendering Device Used", "Rendering Device Allocated"]

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for benchmark in benchmarks:
            row = {
                "Scene": benchmark["scene"],
                "Benchmark ID": benchmark["id"],
                "Benchmark Name": benchmark["name"],
            }
            
            # Add timer values
            for stage in stages:
                row[f"{stage} VK"] = benchmark["timers"].get(stage, {}).get("VK", "N/A")
                row[f"{stage} CPU"] = benchmark["timers"].get(stage, {}).get("CPU", "N/A")

            # Add memory values (default to "N/A" if missing)
            row["Scene Host Used"] = benchmark["memory"].get("Scene", {}).get("Host Used", "N/A")
            row["Scene Device Used"] = benchmark["memory"].get("Scene", {}).get("Device Used", "N/A")
            row["Scene Device Allocated"] = benchmark["memory"].get("Scene", {}).get("Device Allocated", "N/A")

            row["Rendering Host Used"] = benchmark["memory"].get("Rendering", {}).get("Host Used", "N/A")
            row["Rendering Device Used"] = benchmark["memory"].get("Rendering", {}).get("Device Used", "N/A")
            row["Rendering Device Allocated"] = benchmark["memory"].get("Rendering", {}).get("Device Allocated", "N/A")

            writer.writerow(row)


# ============================================================================
#
#   SECTION 2 — ANALYSIS AND VISUALIZATION
#
#   Functions for computing image quality metrics, generating comparison images,
#   and producing charts from benchmark results.
#
# ============================================================================


# Branding color palette
color_set = {
    "green": "#76B900",
    "dark_green": "#2A7F00",
    "light_green": "#8BFF00",
    "teal": "#00A88E",
    "dark_teal": "#007A6A",
    "blue": "#4A90D9",
    "dark_blue": "#3A6EA8",
    "purple": "#7B5EA7",
    "black": "#000000",
    "white": "#FFFFFF",
    "orange1": "#ffb366",
    "orange2": "#ff8829",
    "orange3": "#fe6b40",
    "red": "#D32F2F",
}


def legend_layout(num_items, max_cols=5):
    """Compute ncol and vertical space for a legend that wraps to multiple rows."""
    ncol = min(num_items, max_cols)
    nrows = (num_items + ncol - 1) // ncol
    row_height = 0.035
    return ncol, nrows, nrows * row_height


# ----------------------------------------------------------------------------
#   2.1 — Image Quality (PSNR)
# ----------------------------------------------------------------------------


def load_raw_image(path):
    """Load a .raw binary image file as float64 numpy array (H, W, 3).
    Format: 16-byte header (width, height, channels, bytesPerChannel as uint32) + raw float32 RGBA data."""
    with open(path, 'rb') as f:
        header = np.fromfile(f, dtype=np.uint32, count=4)
        width, height, channels, bytes_per_channel = header
        pixel_count = int(width) * int(height) * int(channels)
        data = np.fromfile(f, dtype=np.float32, count=pixel_count)
    img = data.reshape(int(height), int(width), int(channels))
    return img[:, :, :3].astype(np.float64)


def compute_psnr_hdr(ref_arr, test_arr):
    """Compute PSNR between two HDR float arrays in [0,1] range.
    Uses peak=1.0 to match the in-app GPU computation."""
    mse = np.mean((ref_arr - test_arr) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_psnr(img_ref, img_test):
    """Compute PSNR between two PIL images (8-bit). Returns PSNR in dB."""
    ref = np.array(img_ref, dtype=np.float64)
    test = np.array(img_test, dtype=np.float64)
    mse = np.mean((ref - test) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(255.0 ** 2 / mse)


def generate_comparison_image(ref_path, test_path, output_path, label="", raw_ref_path="", raw_test_path=""):
    """Generate benchmark comparison outputs for one test vs. the raster reference.
    
    Saves:
        - output_path: test (distorted) render only, copied from test_path
        - <stem>_tiled<ext>: 2x2 composite grid (reference | test; diff panels below)
        - <stem>_diff<ext>, _diff_red<ext>, _diff_red_lum<ext>: full-resolution standalone diffs
    
    Tiled layout (2 rows x 2 cols):
        Row 1: Reference | Distorted
        Row 2: Diff (amplified) | Luminance diff as red only
    
    If .raw paths are provided, use them for lossless float32 PSNR computation."""
    ref_img = Image.open(ref_path).convert("RGB")
    test_img = Image.open(test_path).convert("RGB")
    
    w, h = ref_img.size
    ref_arr = np.array(ref_img, dtype=np.float64)
    test_arr = np.array(test_img, dtype=np.float64)
    
    # Compute PSNR from raw float32 if available, otherwise from PNG
    if raw_ref_path and raw_test_path and os.path.exists(raw_ref_path) and os.path.exists(raw_test_path):
        raw_ref = load_raw_image(raw_ref_path)
        raw_test = load_raw_image(raw_test_path)
        print(f"    RAW ref range: [{raw_ref.min():.6f}, {raw_ref.max():.6f}], test range: [{raw_test.min():.6f}, {raw_test.max():.6f}]")
        psnr = compute_psnr_hdr(raw_ref, raw_test)
    else:
        psnr = compute_psnr(ref_img, test_img)
    
    # Amplified absolute difference
    diff_arr = np.abs(ref_arr - test_arr)
    diff_amplified = np.clip(diff_arr * 5, 0, 255).astype(np.uint8)
    diff_img = Image.fromarray(diff_amplified)
    
    # Red only: luminance difference shown as red channel on black
    diff_intensity = np.dot(diff_arr / 255.0, [0.299, 0.587, 0.114])
    diff_intensity = np.clip(diff_intensity * 5, 0.0, 1.0)
    red_only = np.zeros_like(ref_arr)
    red_only[:, :, 0] = (diff_intensity * 255.0)
    red_only_img = Image.fromarray(np.clip(red_only, 0, 255).astype(np.uint8))
    
    # Red on reference luminance (matches in-app eDifferenceRedGray mode)
    ref_gray = np.dot(ref_arr / 255.0, [0.299, 0.587, 0.114])
    red_on_lum = np.zeros_like(ref_arr)
    red_on_lum[:, :, 0] = ((1.0 - diff_intensity) * ref_gray + diff_intensity * 1.0) * 255.0
    red_on_lum[:, :, 1] = ((1.0 - diff_intensity) * ref_gray) * 255.0
    red_on_lum[:, :, 2] = ((1.0 - diff_intensity) * ref_gray) * 255.0
    red_on_lum_img = Image.fromarray(np.clip(red_on_lum, 0, 255).astype(np.uint8))
    
    # Build 2x2 composite with label bar
    bar_h = 36
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except (IOError, OSError):
            font = ImageFont.load_default()
    
    comparison = Image.new("RGB", (w * 2, h * 2 + bar_h * 2), (0, 0, 0))
    comparison.paste(ref_img, (0, bar_h))
    comparison.paste(test_img, (w, bar_h))
    comparison.paste(diff_img, (0, h + bar_h * 2))
    comparison.paste(red_only_img, (w, h + bar_h * 2))
    
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 6), "Reference (Mesh)", fill=(255, 255, 255), font=font)
    draw.text((w + 10, 6), f"{label} (PSNR: {psnr:.2f} dB)", fill=(255, 255, 255), font=font)
    draw.text((10, h + bar_h + 6), "Diff x5", fill=(255, 255, 255), font=font)
    draw.text((w + 10, h + bar_h + 6), "Luminance difference x5 (red)", fill=(255, 255, 255), font=font)
    
    stem, ext = os.path.splitext(output_path)
    tiled_path = f"{stem}_tiled{ext}"

    test_img.save(output_path)
    comparison.save(tiled_path)

    diff_img.save(f"{stem}_diff{ext}")
    red_only_img.save(f"{stem}_diff_red{ext}")
    red_on_lum_img.save(f"{stem}_diff_red_lum{ext}")
    
    return psnr


# ----------------------------------------------------------------------------
#   2.3 — Gaussian Point Splatting (CUDA) vs. Vulkan rasterization (dataset GPS)
# ----------------------------------------------------------------------------

# Sequence names of benchmark_gps.cfg. The first entry is the quality reference.
GPS_REFERENCE_PIPELINE = "3DGS Mesh"
GPS_PIPELINES = [
    "3DGS Mesh",
    "3DGS Mesh stochastic",
    "GPS CUDA ss2",
    "GPS CUDA ss1",
    "GPS CUDA ss2 no culling",
]
# (pipeline, screenshot base name) — files are <scene_prefix>_screenshot_<base>_main.{png,raw}
GPS_SCREENSHOT_DEFS = [
    ("3DGS Mesh", "3dgs"),
    ("3DGS Mesh stochastic", "3dgs_stochastic"),
    ("GPS CUDA ss2", "gps_ss2"),
    ("GPS CUDA ss1", "gps_ss1"),
    ("GPS CUDA ss2 no culling", "gps_ss2_noculling"),
]
# Splatting work only (no UBO update, temporal accumulation, tone mapping, readbacks):
#  - Vulkan raster: distance + sort + rasterization (Vulkan timestamps)
#  - GPS: CUDA algorithm + output conversion (CUDA events) + Vulkan resolve/depth copies
GPS_RENDER_STAGES = ["GPU Dist", "GPU Sort", "Rasterization", "GPS CUDA", "GPS CUDA output", "GPS resolve", "GPS depth"]
GPS_PIPELINE_COLORS = [color_set["dark_green"], color_set["light_green"], color_set["blue"],
                       color_set["teal"], color_set["purple"]]


def _gpu_ms(timers, stages):
    """Sum of the GPU ("VK" column) times of the given stages, in ms."""
    return sum(timers.get(stage, {}).get("VK", 0.0) for stage in stages)


def gps_compute_psnr(full_scene_paths):
    """PSNR of every pipeline screenshot vs. the sorted 3DGS mesh reference, per scene."""
    psnr_data = {}
    ref_base = dict(GPS_SCREENSHOT_DEFS)[GPS_REFERENCE_PIPELINE]
    for scene_name in full_scene_paths:
        scene_prefix = f"benchmark_{scene_name.replace(' ', '_')}"
        ref_file = f"{scene_prefix}_screenshot_{ref_base}_main.png"
        ref_raw = f"{scene_prefix}_screenshot_{ref_base}_main.raw"
        if not os.path.exists(ref_file):
            print(f"Warning: reference screenshot not found: {ref_file}")
            continue
        psnr_data[scene_name] = {}
        for pipeline, base in GPS_SCREENSHOT_DEFS:
            if pipeline == GPS_REFERENCE_PIPELINE:
                continue
            test_file = f"{scene_prefix}_screenshot_{base}_main.png"
            test_raw = f"{scene_prefix}_screenshot_{base}_main.raw"
            if not os.path.exists(test_file):
                print(f"Warning: screenshot not found: {test_file}")
                continue
            comp_file = f"{scene_prefix}_compare_{base}.png"
            psnr = generate_comparison_image(ref_file, test_file, comp_file, label=pipeline,
                                             raw_ref_path=ref_raw, raw_test_path=test_raw)
            psnr_data[scene_name][pipeline] = psnr
            print(f"  {scene_name} | {pipeline}: PSNR = {psnr:.2f} dB")
    return psnr_data


def gps_write_summary(all_results, psnr_data, filename="13_gps_summary_table.csv"):
    """Per scene and pipeline: frame GPU time, splatting GPU time, CPU frame time, fps, speed-up, PSNR."""
    by_key = {(r["scene"], r["name"]): r for r in all_results}
    scenes = list(dict.fromkeys(r["scene"] for r in all_results))
    rows = []
    for scene in scenes:
        ref = by_key.get((scene, GPS_REFERENCE_PIPELINE))
        ref_render = _gpu_ms(ref["timers"], GPS_RENDER_STAGES) if ref else 0.0
        for pipeline in GPS_PIPELINES:
            result = by_key.get((scene, pipeline))
            if not result:
                continue
            timers = result["timers"]
            frame_gpu = timers.get("Frame", {}).get("VK", 0.0)
            frame_cpu = timers.get("Frame", {}).get("CPU", 0.0)
            render_gpu = _gpu_ms(timers, GPS_RENDER_STAGES)
            cuda_ms = timers.get("GPS CUDA", {}).get("VK", 0.0)
            if pipeline == GPS_REFERENCE_PIPELINE:
                psnr = "ref"
            else:
                value = psnr_data.get(scene, {}).get(pipeline)
                psnr = f"{value:.2f}" if value is not None else ""
            rows.append([
                scene, pipeline,
                f"{frame_gpu:.3f}", f"{render_gpu:.3f}", f"{cuda_ms:.3f}" if cuda_ms > 0 else "",
                f"{frame_cpu:.3f}", f"{1000.0 / frame_cpu:.1f}" if frame_cpu > 0 else "",
                f"{ref_render / render_gpu:.2f}" if render_gpu > 0 and ref_render > 0 else "",
                psnr,
            ])
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Scene", "Pipeline", "Frame GPU (ms)", "Splatting GPU (ms)", "of which CUDA (ms)",
                         "Frame CPU (ms)", "FPS", "Splatting speed-up vs Mesh", "PSNR vs Mesh (dB)"])
        writer.writerows(rows)
    print(f"Summary table saved as {filename}")

    # Console summary: average over scenes
    print("\nAverage over scenes (splatting GPU time, PSNR vs mesh):")
    for pipeline in GPS_PIPELINES:
        times = [_gpu_ms(by_key[(s, pipeline)]["timers"], GPS_RENDER_STAGES) for s in scenes if (s, pipeline) in by_key]
        psnrs = [psnr_data[s][pipeline] for s in psnr_data if pipeline in psnr_data[s]]
        if times:
            line = f"  {pipeline:<26} {np.mean(times):8.3f} ms"
            if psnrs:
                line += f"   {np.mean(psnrs):6.2f} dB"
            print(line)


# ----------------------------------------------------------------------------
#   2.2 — Charts: Billboard Bounding Mode Comparison
# ----------------------------------------------------------------------------

# 3DGRT ellipsoid baselines (icosahedron / AABB max-density depth) — fixed chart color
BILLBOARD_ELLIPSOID_BASELINE_PIPELINES = frozenset({
    "3DGRT Icosa Elipsoid",
    "3DGRT AABB Max Opacity",
    "3DGRT AABB Max Opacity All pass",
})

# Icosa billboard modes: stochastic any-hit only (all-pass is not supported with icosa geometry)
ICOSA_STOCHASTIC_PIPELINES = [
    "3DGRT Icosa Elipsoid",
    "3DGRT Icosa Fitted NoShorten",
    "3DGRT Icosa Uniform NoShorten",
    "3DGRT Icosa Fitted",
    "3DGRT Icosa Uniform",
    "3DGRT Icosa Uniform 3/4",
    "3DGRT Icosa Uniform 2/3",
    "3DGRT Icosa Uniform 1/2",
    "3DGRT Icosa Uniform 1/3",
    "3DGRT Icosa Uniform 1/4",
    "3DGRT Icosa Optimal",
]

# (sequence label, screenshot base without cam prefix) for PSNR / compare images
RTX_ICOSA_STOCHASTIC_TEST_DEFS = [
    ("3DGRT Icosa Elipsoid", "3dgrt_icosa_ellipsoid"),
    ("3DGRT Icosa Fitted NoShorten", "3dgrt_icosa_fitted_noshorten"),
    ("3DGRT Icosa Uniform NoShorten", "3dgrt_icosa_uniform_noshorten"),
    ("3DGRT Icosa Fitted", "3dgrt_icosa_fitted"),
    ("3DGRT Icosa Uniform", "3dgrt_icosa_uniform"),
    ("3DGRT Icosa Uniform 3/4", "3dgrt_icosa_uniform34"),
    ("3DGRT Icosa Uniform 2/3", "3dgrt_icosa_uniform23"),
    ("3DGRT Icosa Uniform 1/2", "3dgrt_icosa_uniform12"),
    ("3DGRT Icosa Uniform 1/3", "3dgrt_icosa_uniform13"),
    ("3DGRT Icosa Uniform 1/4", "3dgrt_icosa_uniform14"),
    ("3DGRT Icosa Optimal", "3dgrt_icosa_optimal"),
]

AABB_STOCHASTIC_PIPELINES = [
    "3DGRT AABB Max Opacity",
    "3DGRT AABB Fitted NoShorten",
    "3DGRT AABB Uniform NoShorten",
    "3DGRT AABB Fitted",
    "3DGRT AABB Uniform",
    "3DGRT AABB Uniform 3/4",
    "3DGRT AABB Uniform 2/3",
    "3DGRT AABB Uniform 1/2",
    "3DGRT AABB Uniform 1/3",
    "3DGRT AABB Uniform 1/4",
    "3DGRT AABB Optimal",
]
RTX_AABB_TEST_DEFS = [
    ("3DGRT AABB Max Opacity", "3dgrt_aabb_maxopacity"),
    ("3DGRT AABB Fitted NoShorten", "3dgrt_aabb_fitted_noshorten"),
    ("3DGRT AABB Uniform NoShorten", "3dgrt_aabb_uniform_noshorten"),
    ("3DGRT AABB Fitted", "3dgrt_aabb_fitted"),
    ("3DGRT AABB Uniform", "3dgrt_aabb_uniform"),
    ("3DGRT AABB Uniform 3/4", "3dgrt_aabb_uniform34"),
    ("3DGRT AABB Uniform 2/3", "3dgrt_aabb_uniform23"),
    ("3DGRT AABB Uniform 1/2", "3dgrt_aabb_uniform12"),
    ("3DGRT AABB Uniform 1/3", "3dgrt_aabb_uniform13"),
    ("3DGRT AABB Uniform 1/4", "3dgrt_aabb_uniform14"),
    ("3DGRT AABB Optimal", "3dgrt_aabb_optimal"),
]
# Shorten ray is N/A for all-pass; exclude Fitted/Uniform NoShorten variants.
AABB_ALLPASS_BASE_PIPELINES = [n for n in AABB_STOCHASTIC_PIPELINES if "NoShorten" not in n]
AABB_ALLPASS_PIPELINES = [f"{name} All pass" for name in AABB_ALLPASS_BASE_PIPELINES]
RTX_AABB_ALLPASS_TEST_DEFS = [
    (f"{label} All pass", f"{base}_allpass")
    for label, base in RTX_AABB_TEST_DEFS
    if "noshorten" not in base
]


def _frame_time_ms(timers):
    """Frame time in ms from parsed log timers (CPU preferred, else GPU)."""
    frame = timers.get("Frame", {})
    return frame.get("CPU", 0) or frame.get("VK", 0)


def _billboard_chart_scenes(all_results, psnr_data, pipelines):
    """Scene labels that have PSNR and frame timing for at least one pipeline in the list."""
    fps_data = {}
    for result in all_results:
        if result["name"] not in pipelines:
            continue
        frame_ms = _frame_time_ms(result["timers"])
        if frame_ms > 0:
            fps_data.setdefault(result["scene"], {})[result["name"]] = 1000.0 / frame_ms

    scenes = []
    for scene in psnr_data:
        if scene not in fps_data:
            continue
        if any(p in psnr_data.get(scene, {}) and p in fps_data.get(scene, {}) for p in pipelines):
            scenes.append(scene)
    return scenes, fps_data


def _warn_skip_billboard_chart(filename, all_results, psnr_data, pipelines):
    n_psnr = sum(1 for s in psnr_data for p in pipelines if p in psnr_data.get(s, {}))
    n_timers = sum(1 for r in all_results if r["name"] in pipelines and _frame_time_ms(r["timers"]) > 0)
    print(f"Warning: Skipping {filename} — no scenes with both PSNR and frame timing "
          f"({n_psnr} PSNR entries, {n_timers} timed sequences). "
          f"If all-pass modes show 0 timed sequences, re-run the benchmark (not --skip-exec) "
          f"after increasing --sequenceframes in benchmark_billboards.cfg.")


def build_billboard_pipeline_color_map(pipelines):
    """Assign colors per pipeline; ellipsoid baselines use dark blue."""
    base_colors = [color_set["orange1"], color_set["purple"], color_set["green"], color_set["dark_green"],
                   color_set["teal"], color_set["dark_teal"], color_set["blue"],
                   color_set["orange3"], color_set["black"], color_set["red"]]
    color_map = {}
    color_idx = 0
    for pipeline in pipelines:
        if pipeline in BILLBOARD_ELLIPSOID_BASELINE_PIPELINES:
            color_map[pipeline] = color_set["dark_blue"]
        else:
            color_map[pipeline] = base_colors[color_idx % len(base_colors)]
            color_idx += 1
    return color_map


def plot_dual_axis_chart(all_results, psnr_data, pipelines, sort_by="psnr", filename="14_dual_axis.png"):
    """Plot a dual-axis chart with PSNR and FPS side by side per mode.
    
    For each scene/camera group, modes are sorted by the primary metric (descending).
    Primary metric = solid colored bars (left Y axis).
    Secondary metric = hatched grey bars (right Y axis).
    
    Args:
        all_results: list of benchmark result dicts (with "scene", "name", "timers")
        psnr_data: dict of {scene_label: {pipeline_name: psnr_value}}
        pipelines: list of pipeline names to include (excluding "3DGS Mesh")
        sort_by: "psnr" (Chart A) or "fps" (Chart B)
        filename: output PNG filename
    """
    scenes, fps_data = _billboard_chart_scenes(all_results, psnr_data, pipelines)
    if not scenes:
        _warn_skip_billboard_chart(filename, all_results, psnr_data, pipelines)
        return

    color_map = build_billboard_pipeline_color_map(pipelines)

    num_modes = len(pipelines)
    bar_width = 0.035
    pair_width = bar_width * 2 + 0.005
    group_width = num_modes * (pair_width + 0.02)
    index = np.arange(len(scenes)) * (group_width + 0.2)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax2 = ax.twinx()

    legend_added = set()

    for scene_idx, scene in enumerate(scenes):
        # Gather (pipeline, psnr, fps) tuples
        mode_data = []
        for p in pipelines:
            psnr_val = psnr_data.get(scene, {}).get(p, 0)
            fps_val = fps_data.get(scene, {}).get(p, 0)
            mode_data.append((p, psnr_val, fps_val))

        # Sort by primary metric descending
        if sort_by == "psnr":
            mode_data.sort(key=lambda x: x[1], reverse=True)
        else:
            mode_data.sort(key=lambda x: x[2], reverse=True)

        for j, (pipeline, psnr_val, fps_val) in enumerate(mode_data):
            center = index[scene_idx] + (j - (num_modes - 1) / 2) * (pair_width + 0.02)

            if sort_by == "psnr":
                primary_val = psnr_val
                secondary_val = fps_val
                primary_ax = ax
                secondary_ax = ax2
            else:
                primary_val = fps_val
                secondary_val = psnr_val
                primary_ax = ax
                secondary_ax = ax2

            # Solid colored bar (primary metric, left axis)
            label = pipeline if pipeline not in legend_added else None
            primary_ax.bar(center - bar_width / 2 - 0.002, primary_val, bar_width,
                           color=color_map[pipeline], label=label)
            legend_added.add(pipeline)

            # Hatched grey bar (secondary metric, right axis)
            secondary_ax.bar(center + bar_width / 2 + 0.002, secondary_val, bar_width,
                             color='#AAAAAA', hatch='//', edgecolor='#666666', linewidth=0.5)

    # Axis labels
    if sort_by == "psnr":
        ax.set_ylabel("PSNR (dB) — sorted", color=color_set["dark_green"])
        ax2.set_ylabel("FPS", color='#666666')
        ax.set_ylim(bottom=30)
        title_text = "Quality vs Speed (sorted by PSNR)\nNVIDIA RTX 6000 Ada - 1920x1080"
    else:
        ax.set_ylabel("FPS — sorted", color=color_set["dark_green"])
        ax2.set_ylabel("PSNR (dB)", color='#666666')
        ax2.set_ylim(bottom=30)
        title_text = "Quality vs Speed (sorted by FPS)\nNVIDIA RTX 6000 Ada - 1920x1080"

    ax.set_xlabel("Scene")
    ax.set_xticks(index)
    ax.set_xticklabels(scenes, rotation=45, ha="right")
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(axis='y', which='major', alpha=0.2)

    # Legend: colored entries for modes + one grey hatched entry for secondary metric
    secondary_label = "FPS" if sort_by == "psnr" else "PSNR (dB)"
    legend_handles = [mpatches.Patch(color=color_map[p], label=p) for p in pipelines]
    legend_handles.append(mpatches.Patch(facecolor='#AAAAAA', edgecolor='#666666', hatch='//', label=secondary_label))

    ncol, nrows, legend_height = legend_layout(len(legend_handles))
    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - legend_height - 0.12)
    fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.05, 1.0 - 0.10),
               ncol=ncol, fontsize=7, frameon=False, borderaxespad=0)
    fig.suptitle(title_text, fontsize=11, y=0.99)
    plt.savefig(filename, bbox_inches='tight')
    print(f"Dual-axis chart saved as {filename}")


def plot_scatter_psnr_vs_fps(all_results, psnr_data, pipelines, filename="16_scatter_psnr_vs_fps.png"):
    """Scatter plot: X = FPS, Y = PSNR. Each point is one (scene/camera, bounding mode).
    Points are colored by bounding mode, with different markers per scene/camera.
    Points of the same scene are connected with grey segments sorted by FPS."""
    scenes, fps_data = _billboard_chart_scenes(all_results, psnr_data, pipelines)
    if not scenes:
        _warn_skip_billboard_chart(filename, all_results, psnr_data, pipelines)
        return

    color_map = build_billboard_pipeline_color_map(pipelines)
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'd']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each (scene, mode) as a point, connect per scene sorted by FPS
    for scene_idx, scene in enumerate(scenes):
        marker = markers[scene_idx % len(markers)]
        points = []
        for pipeline in pipelines:
            fps_val = fps_data.get(scene, {}).get(pipeline, None)
            psnr_val = psnr_data.get(scene, {}).get(pipeline, None)
            if fps_val is not None and psnr_val is not None:
                points.append((fps_val, psnr_val, pipeline))
                ax.scatter(fps_val, psnr_val, color=color_map[pipeline], marker=marker,
                           s=80, edgecolors='black', linewidths=0.5, zorder=3)
        # Connect points with grey segments sorted by FPS (left to right)
        if len(points) > 1:
            points.sort(key=lambda p: p[0])
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color='#999999', linewidth=0.8, linestyle='-', zorder=1)

    # Two stacked legends placed in figure coordinates (above the chart)
    from matplotlib.lines import Line2D
    mode_handles = [mpatches.Patch(color=color_map[p], label=p) for p in pipelines]
    scene_handles = [Line2D([0], [0], marker=markers[i % len(markers)], color='grey',
                            markerfacecolor='grey', markersize=8, linestyle='None',
                            label=scene) for i, scene in enumerate(scenes)]

    ax.set_xlabel("FPS (frames per second)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(bottom=30)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(alpha=0.3)

    ncol_scenes, nrows_scenes, scene_legend_height = legend_layout(len(scene_handles))
    ncol_modes, nrows_modes, mode_legend_height = legend_layout(len(mode_handles))
    total_legend_height = scene_legend_height + mode_legend_height

    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - total_legend_height - 0.12)
    fig.legend(handles=scene_handles, loc='upper left', bbox_to_anchor=(0.05, 1.0 - 0.10),
               ncol=ncol_scenes, fontsize=7, frameon=False, borderaxespad=0)
    fig.legend(handles=mode_handles, loc='upper left', bbox_to_anchor=(0.05, 1.0 - 0.10 - scene_legend_height),
               ncol=ncol_modes, fontsize=7, frameon=False, borderaxespad=0)
    fig.suptitle("Quality vs Speed: PSNR vs FPS\nNVIDIA RTX 6000 Ada - 1920x1080",
                 fontsize=11, y=0.99)
    plt.savefig(filename, bbox_inches='tight')
    print(f"Scatter plot saved as {filename}")


def plot_scatter_fps_vs_psnr(all_results, psnr_data, pipelines, filename="17_scatter_fps_vs_psnr.png"):
    """Scatter plot: X = PSNR, Y = FPS. Each point is one (scene/camera, bounding mode).
    Points are colored by bounding mode, with different markers per scene/camera.
    Points of the same scene are connected with grey segments sorted by PSNR."""
    scenes, fps_data = _billboard_chart_scenes(all_results, psnr_data, pipelines)
    if not scenes:
        _warn_skip_billboard_chart(filename, all_results, psnr_data, pipelines)
        return

    color_map = build_billboard_pipeline_color_map(pipelines)
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'h', 'd']

    fig, ax = plt.subplots(figsize=(12, 7))

    for scene_idx, scene in enumerate(scenes):
        marker = markers[scene_idx % len(markers)]
        points = []
        for pipeline in pipelines:
            fps_val = fps_data.get(scene, {}).get(pipeline, None)
            psnr_val = psnr_data.get(scene, {}).get(pipeline, None)
            if fps_val is not None and psnr_val is not None:
                points.append((psnr_val, fps_val, pipeline))
                ax.scatter(psnr_val, fps_val, color=color_map[pipeline], marker=marker,
                           s=80, edgecolors='black', linewidths=0.5, zorder=3)
        if len(points) > 1:
            points.sort(key=lambda p: p[0])
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color='#999999', linewidth=0.8, linestyle='-', zorder=1)

    from matplotlib.lines import Line2D
    mode_handles = [mpatches.Patch(color=color_map[p], label=p) for p in pipelines]
    scene_handles = [Line2D([0], [0], marker=markers[i % len(markers)], color='grey',
                            markerfacecolor='grey', markersize=8, linestyle='None',
                            label=scene) for i, scene in enumerate(scenes)]

    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("FPS (frames per second)")
    ax.set_xlim(left=30)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(alpha=0.3)

    ncol_scenes, nrows_scenes, scene_legend_height = legend_layout(len(scene_handles))
    ncol_modes, nrows_modes, mode_legend_height = legend_layout(len(mode_handles))
    total_legend_height = scene_legend_height + mode_legend_height

    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - total_legend_height - 0.12)
    fig.legend(handles=scene_handles, loc='upper left', bbox_to_anchor=(0.05, 1.0 - 0.10),
               ncol=ncol_scenes, fontsize=7, frameon=False, borderaxespad=0)
    fig.legend(handles=mode_handles, loc='upper left', bbox_to_anchor=(0.05, 1.0 - 0.10 - scene_legend_height),
               ncol=ncol_modes, fontsize=7, frameon=False, borderaxespad=0)
    fig.suptitle("Speed vs Quality: FPS vs PSNR\nNVIDIA RTX 6000 Ada - 1920x1080",
                 fontsize=11, y=0.99)
    plt.savefig(filename, bbox_inches='tight')
    print(f"Scatter plot saved as {filename}")


# ----------------------------------------------------------------------------
#   2.3 — Charts: General Benchmark Timers and Memory
# ----------------------------------------------------------------------------


def plot_cumulative_histogram_timers(
    benchmarks, 
    title,
    ylabel,
    xlabel,
    pipelines, 
    pipeline_names,  
    stages=["GPU Dist", "GPU Sort", "Rasterization"], 
    filename="histogram_timers.png",
    stage_colors = [color_set["black"], color_set["dark_green"], color_set["green"], color_set["light_green"]],
    legend = [],
    device="VK",
    mstofps=False,
    legend_title="Stage"
):
    """Plot cumulative (stacked) bar chart for timer stages across pipelines."""
    scene_groups = defaultdict(list)

    # Group the results by scene and pipeline
    for benchmark in benchmarks:
        scene_name = benchmark["scene"]
        benchmark_name = benchmark["name"]
        timers = benchmark["timers"]
        
        if benchmark_name in pipelines and isinstance(timers, dict):
            scene_groups[scene_name].append((benchmark_name, timers))
    
    all_data = []
    x_labels = []
    width = 0.35  # Base bar width

    # Flatten data for all scenes and pipelines
    for scene, results in scene_groups.items():
        scene_data = {pipeline: {stage: 0 for stage in stages} for pipeline in pipelines}
        for benchmark_name, timers in results:
            for stage in stages:
                if mstofps:
                    stage_ms = timers.get(stage, {}).get(device, 0)
                    scene_data[benchmark_name][stage] += 1000.0 / stage_ms if stage_ms > 0 else 0
                else:
                    scene_data[benchmark_name][stage] += timers.get(stage, {}).get(device, 0)
        all_data.append(scene_data)
        x_labels.append(scene)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Adjust the space between groups by reducing the range of index values
    index = np.arange(len(x_labels)) * 0.5
    num_pipelines = len(pipelines)
    
    # Adjust bar width to ensure space between bars within each group
    bar_width = width * 0.8 / num_pipelines
    
    # Stack bars
    bottom_values = {pipeline: np.zeros(len(x_labels)) for pipeline in pipelines}

    for i, stage in enumerate(stages):
        for j, pipeline in enumerate(pipelines):
            values = [scene_data[pipeline].get(stage, 0) for scene_data in all_data]

            # Adjust the offset for each bar within a group so that bars are well spaced
            position_offset = (j - (num_pipelines - 1) / 2) * (bar_width + 0.05)

            ax.bar(index + position_offset, values, bar_width, color=stage_colors[i], bottom=bottom_values[pipeline])
            bottom_values[pipeline] += np.array(values)

    # Customize plot
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.set_title(title)   

    # Add minor ticks automatically (e.g., 4 minor ticks between major ticks)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='y', which='minor', length=4, width=1)
    ax.grid(axis='y', which='major', alpha=0.2)

    # Format x-ticks with pipeline short names
    ax.set_xticks(index)
    ax.set_xticklabels([f"{scene}\n({', '.join(pipeline_names)})" for scene in x_labels], 
                       rotation=45, ha="right")
    
    # Create legend for stages    
    if len(legend) == 0:
        legend = stages
    
    legend_handles = [mpatches.Patch(color=stage_colors[i], label=stage) for i, stage in enumerate(stages)]
    ax.legend(handles=legend_handles, title=legend_title, loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")


def plot_histogram_timers(
    benchmarks, 
    title,
    ylabel,
    xlabel,
    pipelines, 
    pipeline_names,  
    stages=["GPU Dist", "GPU Sort", "Rasterization"],
    filename="histogram_timers.png",
    stage_colors = [color_set["light_green"], color_set["green"], color_set["dark_green"], color_set["black"] ],
    legend = [],
    device="VK",
    mstofps=False,
    legend_title="Setup",
    sort_bars=False
):
    """Plot grouped bar chart for timer stages, one bar per pipeline per scene.
    Supports sorting bars within each group by value."""
    scene_groups = defaultdict(list)

    # Group the results by scene and pipeline
    for benchmark in benchmarks:
        scene_name = benchmark["scene"]
        benchmark_name = benchmark["name"]
        timers = benchmark["timers"]
        
        if benchmark_name in pipelines and isinstance(timers, dict):
            scene_groups[scene_name].append((benchmark_name, timers))
    
    all_data = []
    x_labels = []
    width = 0.35  # Base bar width

    # Flatten data for all scenes and pipelines
    for scene, results in scene_groups.items():
        scene_data = {pipeline: {stage: 0 for stage in stages} for pipeline in pipelines}
        for benchmark_name, timers in results:
            for stage in stages:
                if mstofps:
                    stage_ms = timers.get(stage, {}).get(device, 0)
                    scene_data[benchmark_name][stage] += 1000.0 / stage_ms if stage_ms > 0 else 0
                else:
                    scene_data[benchmark_name][stage] += timers.get(stage, {}).get(device, 0)
        all_data.append(scene_data)
        x_labels.append(scene)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    num_pipelines = len(pipelines)
    bar_width = 0.08
    bar_gap = 0.01
    group_width = num_pipelines * (bar_width + bar_gap)
    index = np.arange(len(x_labels)) * (group_width + 0.15)
    
    # Fixed color per pipeline (consistent across scenes even when order changes)
    color_map = {pipeline: stage_colors[j % len(stage_colors)] for j, pipeline in enumerate(pipelines)}
    
    # Track which labels have been added to legend
    legend_added = set()

    for scene_idx, scene_data in enumerate(all_data):
        # Get total value per pipeline (sum of all stages) for sorting
        pipeline_totals = []
        for pipeline in pipelines:
            total = sum(scene_data[pipeline].get(stage, 0) for stage in stages)
            pipeline_totals.append((pipeline, total))
        
        if sort_bars:
            # For fps: sort descending (highest first). For ms: sort ascending (lowest first).
            pipeline_totals.sort(key=lambda x: x[1], reverse=mstofps)
        
        for j, (pipeline, _) in enumerate(pipeline_totals):
            position_offset = (j - (num_pipelines - 1) / 2) * (bar_width + bar_gap)
            bottom = 0
            for i, stage in enumerate(stages):
                value = scene_data[pipeline].get(stage, 0)
                label = pipeline if pipeline not in legend_added else None
                ax.bar(index[scene_idx] + position_offset, value, bar_width,
                       color=color_map[pipeline], bottom=bottom, label=label)
                bottom += value
            legend_added.add(pipeline)

    # Customize plot
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 

    # Add minor ticks automatically (e.g., 4 minor ticks between major ticks)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='y', which='minor', length=4, width=1)
    ax.grid(axis='y', which='major', alpha=0.2)

    # Format x-ticks
    ax.set_xticks(index)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # Create legend from pipeline colors (horizontal, above chart)
    if len(legend) == 0:
        legend = pipelines
    legend_handles = [mpatches.Patch(color=color_map[p], label=l) for p, l in zip(pipelines, legend)]

    ncol, nrows, legend_height = legend_layout(len(legend_handles))
    # Save the plot
    plt.tight_layout()
    fig.subplots_adjust(top=1.0 - legend_height - 0.12)
    fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.05, 1.0 - 0.10),
               ncol=ncol, fontsize=7, frameon=False, borderaxespad=0)
    fig.suptitle(title, fontsize=11, y=0.99)
    plt.savefig(filename, bbox_inches='tight')
    print(f"Histogram saved as {filename}")


def plot_cumulative_histogram_memory(
    benchmarks, 
    title,
    ylabel,
    xlabel,
    pipelines, 
    pipeline_names,  
    stages=["Scene", "Rendering"], 
    filename="histogram_memory.png",
    stage_colors = [color_set["dark_green"], color_set["green"], color_set["light_green"]]
):
    """Plot cumulative (stacked) bar chart for VRAM memory usage across pipelines."""
    scene_groups = defaultdict(list)

    # Group the results by scene and pipeline
    for benchmark in benchmarks:
        scene_name = benchmark["scene"]
        benchmark_name = benchmark["name"]
        memory = benchmark["memory"]
        
        if benchmark_name in pipelines and isinstance(memory, dict):
            scene_groups[scene_name].append((benchmark_name, memory))
    
    all_data = []
    x_labels = []
    width = 0.35  # Base bar width

    # Flatten data for all scenes and pipelines
    for scene, results in scene_groups.items():
        scene_data = {pipeline: {stage: 0 for stage in stages} for pipeline in pipelines}
        for benchmark_name, memory in results:
            for stage in stages:
                scene_data[benchmark_name][stage] += memory.get(stage, {}).get("Device Allocated", 0)

        all_data.append(scene_data)
        x_labels.append(scene)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Adjust the space between groups by reducing the range of index values
    index = np.arange(len(x_labels)) * 0.5
    num_pipelines = len(pipelines)
    
    # Adjust bar width to ensure space between bars within each group
    bar_width = width * 0.8 / num_pipelines
    
    # Stack bars
    bottom_values = {pipeline: np.zeros(len(x_labels)) for pipeline in pipelines}

    for i, stage in enumerate(stages):
        for j, pipeline in enumerate(pipelines):
            values = [scene_data[pipeline].get(stage, 0) / 1024 / 1024 for scene_data in all_data]

            # Adjust the offset for each bar within a group so that bars are well spaced
            position_offset = (j - (num_pipelines - 1) / 2) * (bar_width + 0.05)

            ax.bar(index + position_offset, values, bar_width, color=stage_colors[i], bottom=bottom_values[pipeline])
            bottom_values[pipeline] += np.array(values)

    # Customize plot
    ax.set_xlabel(xlabel) 
    ax.set_ylabel(ylabel) 
    ax.set_title(title)   

    # Format x-ticks with pipeline short names
    ax.set_xticks(index)
    ax.set_xticklabels([f"{scene}\n({', '.join(pipeline_names)})" for scene in x_labels], 
                       rotation=45, ha="right")
    
    # Create legend for stages
    legend_handles = [mpatches.Patch(color=stage_colors[i], label=stage) for i, stage in enumerate(stages)]
    ax.legend(handles=legend_handles, title="VRAM Cost", loc='upper right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Histogram saved as {filename}")


# ============================================================================
#
#   SECTION 3 — MAIN ENTRY POINT
#
#   CLI argument parsing, scene/dataset definitions, benchmark orchestration,
#   and analysis output generation.
#
# ============================================================================


if __name__ == "__main__":   
    # Setup argument parsing for the base dataset path
    parser = argparse.ArgumentParser(description="Run benchmarks for 3D scenes.")
    parser.add_argument("benchmark", type=str, help="path to the benchmark .cfg file")
    parser.add_argument("dataset", type=str, choices=["3DGS", "3DGRT", "3DGUT", "3DGS_BILLBOARDS", "GPS"],
                        help="Dataset to use (GPS: INRIA 3DGS scenes, Gaussian Point Splatting CUDA vs. Vulkan raster, "
                             "requires a -DUSE_GPS_CUDA=ON build)")
    parser.add_argument("dataset_path", type=str, help="Base path to the dataset")
    parser.add_argument("csv_name", type=str, help="Name of the output csv file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print benchmark output to console in real-time")
    parser.add_argument("--headless", action="store_true", help="Run the app in headless mode (no window)")
    parser.add_argument("--forcegpu", type=int, metavar="ID", default=None,
                        help="Forward --forcegpu ID to the app (Vulkan physical device index)")
    parser.add_argument("--skip-exec", action="store_true", help="Skip execution, re-analyze existing results in _benchmark/")
    parser.add_argument("--scenes", type=str, default="",
                        help="Comma-separated substrings to select a subset of the dataset scenes (e.g. 'bonsai,bicycle')")

    args = parser.parse_args()

    # Find executable (only needed when actually running benchmarks)
    executable = None
    if not args.skip_exec:
        paths = ["./_bin/Release/vk_gaussian_splatting.exe", "../_bin/Release/vk_gaussian_splatting.exe","./_bin/Release/vk_gaussian_splatting_app", "../_bin/Release/vk_gaussian_splatting_app"]
        for path in paths:
            if os.path.exists(path):
                executable = os.path.abspath(path)
                break
        if executable:
            print(f"Using executable: {executable}")
        else:
            print("Executable not found.")
            sys.exit(1)
    else:
        print("--skip-exec: Skipping execution, analyzing existing results.")

    # path to the benchmark definition
    benchmark_file = os.path.abspath(args.benchmark)

    # Define the scenes from 3DGS with the relative paths
    # Each value is a tuple: (ply_relative_path, camera_json_relative_path)
    # Use "" for camera_json_relative_path if no camera presets are available
    scenes = {
        "bicycle 6.13M Splats": ("bicycle/bicycle/point_cloud/iteration_30000/point_cloud.ply", "bicycle/bicycle/cameras.json")
        ,"bonsai 1.24M Splats": ("bonsai/bonsai/point_cloud/iteration_30000/point_cloud.ply", "bonsai/bonsai/cameras.json")
        ,"counter 1.22M Splats": ("counter/point_cloud/iteration_30000/point_cloud.ply", "counter/cameras.json")
        ,"drjohnson 3.41M Splats": ("drjohnson/point_cloud/iteration_30000/point_cloud.ply", "drjohnson/cameras.json")
        ,"flowers 3.64M Splats": ("flowers/point_cloud/iteration_30000/point_cloud.ply", "flowers/cameras.json")
        ,"garden 5.83M Splats": ("garden/point_cloud/iteration_30000/point_cloud.ply", "garden/cameras.json")
        ,"kitchen 1.85M Splats": ("kitchen/point_cloud/iteration_30000/point_cloud.ply", "kitchen/cameras.json")
        ,"playroom 2.55M Splats": ("playroom/point_cloud/iteration_30000/point_cloud.ply", "playroom/cameras.json")
        ,"room 1.59M Splats": ("room/point_cloud/iteration_30000/point_cloud.ply", "room/cameras.json")
        ,"stump 4.96M Splats": ("stump/point_cloud/iteration_30000/point_cloud.ply", "stump/cameras.json")
        ,"train 1.03M Splats": ("train/point_cloud/iteration_30000/point_cloud.ply", "train/cameras.json")
        ,"treehill 3.78M Splats": ("treehill/point_cloud/iteration_30000/point_cloud.ply", "treehill/cameras.json")
        ,"truck 2.54M Splats": ("truck/point_cloud/iteration_30000/point_cloud.ply", "truck/cameras.json")
    }
    
    # Define the scenes from 3DGUT with the relative paths
    rt_scenes = {
        "bicycle 1M Splats": ("bicycle_exported.ply", "")
       ,"bonsai 1M Splats": ("bonsai_exported.ply", "")
       ,"counter 1M Splats": ("counter_exported.ply", "")
       ,"flowers 1M Splats": ("flowers_exported.ply", "")
       ,"garden 1M Splats": ("garden_exported.ply", "")
       ,"kitchen 1M Splats": ("kitchen_exported.ply", "")
       ,"room 1M Splats": ("room_exported.ply", "")
       ,"stump 1M Splats": ("stump_exported.ply", "")
       ,"treehill 1M Splats": ("treehill_exported.ply", "")
    }

    # Define the scenes for 3DGS_BILLBOARDS dataset
    # Third element is a list of camera preset IDs to test (optional)
    billboard_scenes = {
        "City 100M Splats": ("test_billboard_city.vkgs", "", [0, 1, 2])
        ,"Winter Garden 16M Splats": ("test_billboard_winter_garden.vkgs", "", [0])
    }

    # Select the appropriate scenes based on the dataset argument
    if args.dataset in ("3DGS", "GPS"):
        selected_scenes = scenes
    elif args.dataset == "3DGS_BILLBOARDS":
        selected_scenes = billboard_scenes
    else:
        selected_scenes = rt_scenes

    def resolve_dataset_path(base, relative):
        """Join base and relative path. If missing, also accept the flat INRIA layout
        ("bicycle/point_cloud/..." instead of "bicycle/bicycle/point_cloud/...")."""
        path = os.path.join(base, relative)
        if os.path.exists(path):
            return path
        parts = relative.replace("\\", "/").split("/")
        if len(parts) > 2 and parts[0] == parts[1]:
            flat = os.path.join(base, *parts[1:])
            if os.path.exists(flat):
                return flat
        return path

    # Build the full paths by combining the dataset path and scene relative paths
    full_scene_paths = {}
    for scene_name, scene_tuple in selected_scenes.items():
        ply_relative = scene_tuple[0]
        cam_relative = scene_tuple[1] if len(scene_tuple) > 1 else ""
        camera_presets = scene_tuple[2] if len(scene_tuple) > 2 else []
        ply_path = resolve_dataset_path(args.dataset_path, ply_relative)
        cam_path = resolve_dataset_path(args.dataset_path, cam_relative) if cam_relative else ""
        if not args.skip_exec and not os.path.exists(ply_path):
            print(f"Warning: scene file not found, skipping '{scene_name}': {ply_path}")
            continue
        if args.scenes and not any(s.lower() in scene_name.lower() for s in args.scenes.split(",")):
            continue
        full_scene_paths[scene_name] = (ply_path, cam_path, camera_presets)
    
    # Prepare the benchmark directory and output files
    benchmark_dir = "_benchmark"
    os.makedirs(benchmark_dir, exist_ok=True)
    os.chdir(benchmark_dir)

    all_results = []
    for scene_name, (scene_path, camera_path, camera_presets) in full_scene_paths.items():

        output_prefix = f"benchmark_{scene_name.replace(' ', '_')}"
        output_log = f"{output_prefix}.log"

        if not args.skip_exec:
            # Generate temp cfg with camera presets if needed
            effective_cfg = benchmark_file
            if camera_presets:
                effective_cfg = generate_multi_camera_cfg(benchmark_file, camera_presets)

            print(f"Running benchmark for {scene_name} at {scene_path}...")
            run_benchmark(executable, effective_cfg, scene_path, output_log, camera_path,
                          verbose=args.verbose, headless=args.headless, forcegpu=args.forcegpu)

            # Rename all files starting with "screenshot" (any extension)
            for img_file in os.listdir("."):
                if img_file.startswith("screenshot"):
                    new_name = f"{output_prefix}_{img_file}"
                    shutil.move(img_file, new_name)

        # Read and process the benchmark log
        if not os.path.exists(output_log):
            print(f"Warning: Log file not found: {output_log} — skipping scene '{scene_name}'")
            continue

        with open(output_log, "r", encoding="utf-8") as file:
            log_text = file.read()
        
        # Parse results — split by camera preset if multiple presets
        if len(camera_presets) > 1:
            results = parse_benchmark_multi_camera(log_text, scene_name, camera_presets)
        else:
            results = parse_benchmark(log_text, scene_name)
        all_results.extend(results)
    
    save_to_csv(all_results, args.csv_name)

    if args.dataset == "3DGS":
        plot_cumulative_histogram_timers(
            all_results, 
            xlabel="Scene",
            ylabel="Cumulative VK Time (milliseconds)",
            title="Pipeline Performance Comparison - Mesh vs. Vertex - SH Format uint 8",
            pipelines = ["Mesh pipeline uint8", "Vert pipeline uint8"], 
            pipeline_names= ["Mesh", "Vert"],
            stages=["GPU Dist", "GPU Sort", "Rasterization"], 
            filename="05_histogram_shader_timers_uint8.png")

        plot_cumulative_histogram_timers(
            all_results, 
            xlabel="Scene",
            ylabel="Cumulative VK Time (milliseconds)",
            title="Pipeline Performance Comparison - Mesh vs. Vertex - SH Format float 16",
            pipelines = ["Mesh pipeline fp16", "Vert pipeline fp16"], 
            pipeline_names= ["Mesh", "Vert"],
            stages=["GPU Dist", "GPU Sort", "Rasterization"], 
            filename="04_histogram_shader_timers_fp16.png")

        plot_cumulative_histogram_timers(
            all_results, 
            xlabel="Scene",
            ylabel="Cumulative VK Time (milliseconds)",
            title="Pipeline Performance Comparison - Mesh vs. Vertex - SH Format float 32",
            pipelines = ["Mesh pipeline fp32", "Vert pipeline fp32"], 
            pipeline_names= ["Mesh", "Vert"],
            stages=["GPU Dist", "GPU Sort", "Rasterization"], 
            filename="03_histogram_shader_timers_fp32.png")

        plot_cumulative_histogram_timers(
            all_results, 
            xlabel="Scene",
            ylabel="Cumulative VK Time (milliseconds)",
            title="Mesh Pipeline Performance Comparison - SH storage formats in float 32, float 16 and uint 8",
            pipelines = ["Mesh pipeline fp32", "Mesh pipeline fp16", "Mesh pipeline uint8"],
            pipeline_names= ["fp32", "fp16", "uint8"],
            stages=["GPU Dist", "GPU Sort", "Rasterization"], 
            filename="00_histogram_format_timers_mesh.png")

        plot_cumulative_histogram_timers(
            all_results, 
            xlabel="Scene",
            ylabel="Cumulative VK Time (milliseconds)",
            title="Vertex Pipeline Performance Comparison - SH storage formats in float 32, float 16 and uint 8",
            pipelines = ["Vert pipeline fp32", "Vert pipeline fp16", "Vert pipeline uint8"],
            pipeline_names= ["fp32", "fp16", "uint8"],
            stages=["GPU Dist", "GPU Sort", "Rasterization"], 
            filename="01_histogram_format_timers_vertex.png")

        plot_cumulative_histogram_memory(
            all_results, 
            xlabel="Scene",
            ylabel="Cumulative VRAM usage (Mega Bytes)",
            title="Memory Consumption Comparison - SH storage formats in float 32, float 16 and uint 8",
            pipelines = ["Mesh pipeline fp32", "Mesh pipeline fp16", "Mesh pipeline uint8"],
            pipeline_names= ["fp32", "fp16", "uint8"],
            stages=["Scene", "Rasterization"], 
            filename="06_histogram_format_memory.png")

    if args.dataset == "GPS":
        gps_legend = ["A - VK3DGS mesh, GPU radix sort (reference)",
                      "B - VK3DGS mesh, stochastic splat",
                      "C - GPS CUDA, supersampling 2x2",
                      "D - GPS CUDA, supersampling 1x1",
                      "E - GPS CUDA 2x2, no occlusion/hierarchical culling"]

        plot_histogram_timers(
            all_results,
            xlabel="Scene",
            ylabel="Splatting GPU time (milliseconds)",
            title="Gaussian Point Splatting (CUDA) vs. VK3DGS rasterization - splatting GPU time (ms)\n"
                  "Vulkan timestamps (dist + sort + raster / resolve) + CUDA events (GPS) - 1920x1080",
            pipelines=GPS_PIPELINES,
            pipeline_names=GPS_PIPELINES,
            stages=GPS_RENDER_STAGES,
            stage_colors=GPS_PIPELINE_COLORS,
            legend=gps_legend,
            filename="10_gps_histogram_splatting_gpu_time.png")

        plot_histogram_timers(
            all_results,
            xlabel="Scene",
            ylabel="Frame GPU time (milliseconds)",
            title="Gaussian Point Splatting (CUDA) vs. VK3DGS rasterization - full frame GPU time (ms)\n"
                  "includes temporal accumulation for the stochastic pipelines - 1920x1080",
            pipelines=GPS_PIPELINES,
            pipeline_names=GPS_PIPELINES,
            stages=["Frame"],
            stage_colors=GPS_PIPELINE_COLORS,
            legend=gps_legend,
            filename="11_gps_histogram_frame_gpu_time.png")

        plot_histogram_timers(
            all_results,
            xlabel="Scene",
            ylabel="Frames per second",
            title="Gaussian Point Splatting (CUDA) vs. VK3DGS rasterization - framerate (fps, CPU frame time)\n"
                  "GPS runs synchronously (CUDA work + host readbacks are on the frame's critical path) - 1920x1080",
            pipelines=GPS_PIPELINES,
            pipeline_names=GPS_PIPELINES,
            stages=["Frame"],
            stage_colors=GPS_PIPELINE_COLORS,
            legend=gps_legend,
            device="CPU",
            mstofps=True,
            filename="12_gps_histogram_fps.png")

        psnr_data = gps_compute_psnr(full_scene_paths)
        gps_write_summary(all_results, psnr_data)

    if args.dataset == "3DGRT":
        plot_histogram_timers(
            all_results, 
            xlabel="Scene",
            ylabel="Raytracing VK Time (milliseconds)",
            title="Raytracing (3DGRT) Pipeline Performance Comparison Using Acceleration Structures Variants",
            pipelines = ["3DGRT - sh uint8 - inst. off - comp. on", "3DGRT - sh uint8 - inst. off - comp. off", "3DGRT - sh uint8 - inst. on - comp. on", "3DGRT - sh uint8 - inst. on - comp. on - useAABB"],
            pipeline_names= ["A", "B", "C", "D"],
            stages=["Raytracing"], 
            legend=["A - TLAS inst. off, BLAS comp. on", "B - TLAS inst. off, BLAS comp. off", "C - TLAS inst. on, BLAS comp. on", "D - TLAS inst. on, Use AABB"],
            filename="07_histogram_as_format_timers_3dgrt.png")
        
        plot_cumulative_histogram_memory(
            all_results, 
            xlabel="Scene (A - TLAS inst. off, BLAS comp. on; B - TLAS inst. off, BLAS comp. off; C - TLAS inst. on, BLAS comp. on; D - TLAS inst. on, Use AABB)",
            ylabel="Cumulative VRAM usage (Mega Bytes)",
            title="Raytracing (3DGRT) Memory Consumption Comparison Using Acceleration Structures Variants",
            pipelines = ["3DGRT - sh uint8 - inst. off - comp. on", "3DGRT - sh uint8 - inst. off - comp. off", "3DGRT - sh uint8 - inst. on - comp. on", "3DGRT - sh uint8 - inst. on - comp. on - useAABB"],
            pipeline_names= ["A", "B", "C", "D"],
            stages=["Scene", "Raytracing"], 
            filename="08_histogram_as_format_memory_3dgrt.png")

    if args.dataset == "3DGUT":

        plot_cumulative_histogram_timers(
            all_results, 
            xlabel="Scene (Extent method)",
            ylabel="Cumulative VK Time (milliseconds)",
            title="Comparison of VK3DGUT rendering performances on 3DGUT dataset - GPU time (ms) \n NVIDIA RTX 6000 Ada - 1920x1080",
            pipelines = ["3DGUT - conic", "3DGUT - eigen"],
            pipeline_names= ["conic", "eigen"],
            stages=["GPU Dist", "GPU Sort", "Rasterization"], 
            legend=["conic", "eigen"],
            legend_title="Extent method",
            filename="09_histogram_timers_3dgut.png")

        plot_histogram_timers(
            all_results, 
            xlabel="Scene (Extent method)",
            ylabel="Frames per second (fps)",
            title="Comparison of VK3DGUT rendering performances on 3DGUT dataset - framerate (fps) \n NVIDIA RTX 6000 Ada - 1920x1080",
            pipelines = ["3DGUT - conic", "3DGUT - eigen"],
            pipeline_names= ["conic", "eigen"],
            stages=["Frame"], 
            legend=["conic", "eigen"],
            mstofps=True,
            legend_title="Extent method",
            stage_colors = [color_set["green"], color_set["dark_green"]],
            filename="09_histogram_timers_3dgut_fps.png")

    # using 3DGUT datset (informative but fair comparison is not possible using a single dataset, 
    # since each rendering depent on an appropriate training)
    if args.dataset == "COMPARE": 

        plot_histogram_timers(
            all_results, 
            xlabel="Scene (Pipelines)",
            ylabel="Frames per second (fps)",
            title="Comparison of VK3DGS, VK3DGUT and VK3DGRT rendering performances on 3DGUT dataset \n NVIDIA RTX 6000 Ada - 1920x1080 - 3DGRT TLAS inst. off",
            pipelines = ["3DGS - sh fp32", "3DGUT - sh fp32", "3DGRT - sh fp32"],
            pipeline_names= ["A", "B", "C"],
            stages=["Frame"], 
            legend=["A - VK3DGS", "B - VK3DGUT", "C - VK3DGRT"],
            device="CPU",
            mstofps=True,
            legend_title="Pipeline",
            filename="09_histogram_timers_3dgs_3dgut_3dgrt_fps.png")

    if args.dataset == "3DGS_BILLBOARDS":

        # Image comparison: compute PSNR of each 3DGRT render vs raster reference
        # Each entry: (label, png_base, raw_base) — "screenshot_" prefix is added per camera
        rtx_test_defs = RTX_ICOSA_STOCHASTIC_TEST_DEFS + RTX_AABB_TEST_DEFS + RTX_AABB_ALLPASS_TEST_DEFS
        
        psnr_data = {}
        for scene_name, (_, _, camera_presets) in full_scene_paths.items():
            scene_prefix = f"benchmark_{scene_name.replace(' ', '_')}"
            
            # Determine which camera passes to iterate over
            cam_ids = camera_presets if camera_presets else [None]
            for cam_id in cam_ids:
                cam_prefix = f"cam{cam_id}_" if cam_id is not None else ""
                label = f"{scene_name} (Cam {cam_id})" if len(cam_ids) > 1 else scene_name
                
                ref_file = f"{scene_prefix}_screenshot_{cam_prefix}3dgs_main.png"
                ref_raw_file = f"{scene_prefix}_screenshot_{cam_prefix}3dgs_main.raw"
                
                if not os.path.exists(ref_file):
                    print(f"Warning: Reference screenshot not found: {ref_file}")
                    continue
                
                psnr_data[label] = {}
                for test_name, test_base in rtx_test_defs:
                    test_file = f"{scene_prefix}_screenshot_{cam_prefix}{test_base}_main.png"
                    test_raw_file = f"{scene_prefix}_screenshot_{cam_prefix}{test_base}_main.raw"
                    
                    if not os.path.exists(test_file):
                        print(f"Warning: Test screenshot not found: {test_file}")
                        continue
                    
                    comp_file = f"{scene_prefix}_compare_{cam_prefix}{test_base}.png"
                    psnr = generate_comparison_image(ref_file, test_file, comp_file, label=test_name,
                                                     raw_ref_path=ref_raw_file, raw_test_path=test_raw_file)
                    psnr_data[label][test_name] = psnr
                    print(f"  {label} | {test_name}: PSNR = {psnr:.2f} dB")
        
        # --- Icosahedron charts (stochastic any-hit) ---
        if psnr_data:
            plot_dual_axis_chart(all_results, psnr_data, ICOSA_STOCHASTIC_PIPELINES,
                                sort_by="psnr", filename="14_icosa_quality_vs_speed_by_psnr.png")
            plot_dual_axis_chart(all_results, psnr_data, ICOSA_STOCHASTIC_PIPELINES,
                                sort_by="fps", filename="15_icosa_quality_vs_speed_by_fps.png")
            plot_scatter_psnr_vs_fps(all_results, psnr_data, ICOSA_STOCHASTIC_PIPELINES,
                                     filename="16_icosa_scatter_psnr_vs_fps.png")
            plot_scatter_fps_vs_psnr(all_results, psnr_data, ICOSA_STOCHASTIC_PIPELINES,
                                     filename="17_icosa_scatter_fps_vs_psnr.png")

        # --- AABB charts (stochastic any-hit) ---
        if psnr_data:
            plot_dual_axis_chart(all_results, psnr_data, AABB_STOCHASTIC_PIPELINES,
                                sort_by="psnr", filename="14_aabb_quality_vs_speed_by_psnr.png")
            plot_dual_axis_chart(all_results, psnr_data, AABB_STOCHASTIC_PIPELINES,
                                sort_by="fps", filename="15_aabb_quality_vs_speed_by_fps.png")
            plot_scatter_psnr_vs_fps(all_results, psnr_data, AABB_STOCHASTIC_PIPELINES,
                                     filename="16_aabb_scatter_psnr_vs_fps.png")
            plot_scatter_fps_vs_psnr(all_results, psnr_data, AABB_STOCHASTIC_PIPELINES,
                                     filename="17_aabb_scatter_fps_vs_psnr.png")

        # --- AABB all-pass charts (separate from stochastic icosa and stochastic AABB) ---
        if psnr_data:
            print("Generating AABB all-pass charts (18-21)...")
            plot_dual_axis_chart(all_results, psnr_data, AABB_ALLPASS_PIPELINES,
                                sort_by="psnr", filename="18_aabb_allpass_quality_vs_speed_by_psnr.png")
            plot_dual_axis_chart(all_results, psnr_data, AABB_ALLPASS_PIPELINES,
                                sort_by="fps", filename="19_aabb_allpass_quality_vs_speed_by_fps.png")
            plot_scatter_psnr_vs_fps(all_results, psnr_data, AABB_ALLPASS_PIPELINES,
                                     filename="20_aabb_allpass_scatter_psnr_vs_fps.png")
            plot_scatter_fps_vs_psnr(all_results, psnr_data, AABB_ALLPASS_PIPELINES,
                                     filename="21_aabb_allpass_scatter_fps_vs_psnr.png")

        # Generate summary CSV combining timing and PSNR data
        summary_pipelines = ["3DGS Mesh"] + ICOSA_STOCHASTIC_PIPELINES + AABB_STOCHASTIC_PIPELINES + AABB_ALLPASS_PIPELINES
        # Collect unique scene labels from results (includes camera-labeled names)
        scene_labels = list(dict.fromkeys(r["scene"] for r in all_results))
        summary_file = "13_summary_table.csv"
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Scene", "Mode", "Frame Time (ms)", "FPS", "PSNR (dB)"])
            for label in scene_labels:
                for pipeline in summary_pipelines:
                    ms = ""
                    fps = ""
                    psnr = "ref" if pipeline == "3DGS Mesh" else ""
                    # Look up timing from all_results
                    for result in all_results:
                        if result["scene"] == label and result["name"] == pipeline:
                            frame_ms = result["timers"].get("Frame", {}).get("CPU", 0)
                            if frame_ms > 0:
                                ms = f"{frame_ms:.2f}"
                                fps = f"{1000.0 / frame_ms:.2f}"
                            break
                    # Look up PSNR
                    if pipeline != "3DGS Mesh" and label in psnr_data:
                        psnr_val = psnr_data[label].get(pipeline, None)
                        if psnr_val is not None:
                            psnr = f"{psnr_val:.2f}"
                    writer.writerow([label, pipeline, ms, fps, psnr])
        print(f"Summary table saved as {summary_file}")

    print("CSV and histogram generation complete.")
