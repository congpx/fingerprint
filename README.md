# FingerPrint

FingerPrint is a research-oriented repository for fingerprint retrieval experiments, covering baseline training, MixStyle-based domain generalization, hyperparameter ablation, triplet fine-tuning, full-gallery evaluation, error analysis, and computational cost measurement.

## Repository Purpose

This project organizes the complete experimental workflow for a multi-phase fingerprint retrieval study. The repository includes:

- Python entry points for training, evaluation, cost measurement, and figure generation
- Shell scripts for running each experimental phase reproducibly
- split definition files
- experiment outputs and summaries
- a patch file used during checkpoint/path handling

## Recommended Repository Layout

```bash
fingerprint/
├── scripts/                  # all .py and .sh files
├── outputs/                  # all outputs_* directories merged here
├── data/
│   └── splits/               # CSV split definitions
├── patches/                  # patch files
├── docs/                     # optional notes, images, PDFs
├── requirements.txt
├── README.md
└── .gitignore
```

Current Workflow Components

The original repository contains the following main Python and shell entry points:

```
run_socofing.py
run_socofing_v2.py
run_socofing_v3.py
run_socofing_v3_aug.py
run_socofing_v3_aug1.py
run_socofing_v3_mixstyle.py
run_socofing_v3_mixstyle_triplet.py
eval_fullgallery_compat.py
eval_v2_fullgallery.py

measure_computational_cost.py

draw_figure2_cr_hard.py

draw_figure2_cr_hard_v2.py

draw_pipeline.py

run_phase2.sh

run_phase3_ckpt.sh

run_phase3_ckpt_fixed.sh

run_phase4_mix_tuning.sh

run_phase5_final_confirm.sh

run_phase6_final_eval.sh

run_phase6_triplet.sh

final_eval_one_command.sh

error_analysis_cr_hard.sh

phase6_error_analysis_and_figure2.sh
```

The repository also includes split files and multiple phase-specific output directories such as outputs_phase2, outputs_phase3, outputs_phase4_mix_tuning, outputs_phase5_final, outputs_phase6_triplet, outputs_phase6_final_eval, outputs_final_eval, and others. In the cleaned layout, these should be merged under a single outputs/ directory.

## File-by-File Responsibilities

# Training Python files
```
run_socofing.py
Baseline training entry point for the original SOCOFing retrieval setup.

run_socofing_v2.py
Second-generation training script, typically used to refine data flow, training logic, or experiment settings compared with the original baseline.

run_socofing_v3.py
Third-generation baseline training script, usually serving as the stronger reference pipeline before advanced augmentation or domain-generalization techniques are added.

run_socofing_v3_aug.py
Version of the v3 training pipeline with an explicit augmentation strategy added.

run_socofing_v3_aug1.py
Alternative augmentation variant for v3, likely used for controlled augmentation ablation.

run_socofing_v3_mixstyle.py
Main training script for the MixStyle-based model. This is the core entry point for the domain-generalization stage of the project.

run_socofing_v3_mixstyle_triplet.py
Final training/fine-tuning script that adds batch-hard triplet or metric-learning refinement on top of the best MixStyle-based checkpoint.
```

# Evaluation and analysis Python files
```
eval_fullgallery_compat.py
Compatibility-oriented full-gallery evaluation script. Likely used to maintain consistency with earlier checkpoint formats or older result layouts.

eval_v2_fullgallery.py
Main full-gallery evaluation script for the newer pipeline. Use this when generating final retrieval metrics.

measure_computational_cost.py
Measures model complexity or runtime-related indicators such as parameter count, FLOPs, or inference cost.

draw_figure2_cr_hard.py
Generates the Figure 2 visualization for CR-hard analysis.

draw_figure2_cr_hard_v2.py
Revised version of the Figure 2 generator, likely with improved formatting or corrected filtering logic.

draw_pipeline.py
Draws the overall method or workflow diagram used in the paper/report.
```
# Shell scripts for phase execution
```
run_phase2.sh
Runs Phase 2 experiments. Usually the first major controlled phase after the earliest baseline setup.

run_phase3_ckpt.sh
Runs Phase 3 with checkpoint handling.

run_phase3_ckpt_fixed.sh
Fixed version of the Phase 3 checkpoint script, likely created to correct path or checkpoint-selection issues.

run_phase4_mix_tuning.sh
Runs MixStyle hyperparameter tuning or ablation experiments.

run_phase5_final_confirm.sh
Runs the final confirmation stage after selecting the best MixStyle setting.

run_phase6_triplet.sh
Runs the triplet fine-tuning stage on top of the best selected checkpoint.

run_phase6_final_eval.sh
Runs the final evaluation after triplet fine-tuning.

final_eval_one_command.sh
Convenience wrapper to execute the final evaluation pipeline in one command.

error_analysis_cr_hard.sh
Runs the CR-hard error analysis workflow.

phase6_error_analysis_and_figure2.sh
End-to-end script for running Phase 6 error analysis and then generating the Figure 2 output.
```
Data and artifact files
```
splits/index.csv

splits/index_v2.csv

splits/index_v3.csv
Split definition files that specify how the dataset is partitioned for each protocol or training version.

multi_best_ckpt_relative.patch
Patch file used to fix or adapt checkpoint path handling, likely for relative-path compatibility.
```
# Outputs

The original repository contains many phase-specific output directories, including:
```
outputs

outputs_computational_cost

outputs_error_analysis_cr_hard

outputs_final_eval

outputs_phase2

outputs_phase3

outputs_phase3_ckpt

outputs_phase4_mix_tuning

outputs_phase5_final

outputs_phase6_error_analysis

outputs_phase6_final_eval

outputs_phase6_triplet
```
In the cleaned repository layout, move all of them under:

outputs/

For example:
```
outputs/
├── computational_cost/
├── error_analysis_cr_hard/
├── final_eval/
├── phase2/
├── phase3/
├── phase3_ckpt/
├── phase4_mix_tuning/
├── phase5_final/
├── phase6_error_analysis/
├── phase6_final_eval/
└── phase6_triplet/

```
# Dataset Note

The raw dataset is not included in this repository because of its size. You must prepare the fingerprint dataset locally and ensure that dataset paths inside the Python and shell scripts point to your local storage.

The split CSV files should remain available, for example:
```
data/splits/index.csv
data/splits/index_v2.csv
data/splits/index_v3.csv
```
# Environment Setup
1. Create virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

If needed, install PyTorch separately for your CUDA version first.

How to Run the Entire Project From Scratch

The full experimental workflow is phase-based.

# Step 1. Prepare dataset paths

Before running anything, open the Python and shell scripts and update:

dataset root path

split CSV path

checkpoint output path

evaluation output path

If you have reorganized the repository, scripts should be called from scripts/.

# Step 2. Run baseline or earlier reference training

Depending on your intended starting point, run one of:
```
python scripts/run_socofing.py
python scripts/run_socofing_v2.py
python scripts/run_socofing_v3.py
```
If your current study starts from the v3-based pipeline, run_socofing_v3.py is the practical starting point.

# Step 3. Run augmentation or MixStyle stage

For augmentation ablations:
```
python scripts/run_socofing_v3_aug.py
python scripts/run_socofing_v3_aug1.py
```
For the main MixStyle stage:
```
python scripts/run_socofing_v3_mixstyle.py
```
Or use the phase wrapper:
```
bash scripts/run_phase4_mix_tuning.sh
```
# Step 4. Select best MixStyle setting

Use the outputs from the tuning phase to identify the best configuration, then confirm it with:
```
bash scripts/run_phase5_final_confirm.sh
```
# Step 5. Run triplet fine-tuning
python scripts/run_socofing_v3_mixstyle_triplet.py

Or:
```
bash scripts/run_phase6_triplet.sh
```
# Step 6. Run final evaluation
```
python scripts/eval_v2_fullgallery.py
```
Or:
```
bash scripts/run_phase6_final_eval.sh
bash scripts/final_eval_one_command.sh
```
Use eval_fullgallery_compat.py only if you need compatibility with older checkpoints or result formatting.

# Step 7. Run error analysis
```
bash scripts/error_analysis_cr_hard.sh
bash scripts/phase6_error_analysis_and_figure2.sh
```
# Step 8. Draw figures and measure cost
```
python scripts/draw_figure2_cr_hard.py
python scripts/draw_figure2_cr_hard_v2.py
python scripts/draw_pipeline.py
python scripts/measure_computational_cost.py
```

Minimal Reproducible End-to-End Order

If you want a simple full order to follow:
```
python scripts/run_socofing_v3.py
python scripts/run_socofing_v3_mixstyle.py
bash scripts/run_phase4_mix_tuning.sh
bash scripts/run_phase5_final_confirm.sh
python scripts/run_socofing_v3_mixstyle_triplet.py
bash scripts/run_phase6_triplet.sh
bash scripts/run_phase6_final_eval.sh
bash scripts/error_analysis_cr_hard.sh
python scripts/measure_computational_cost.py
python scripts/draw_figure2_cr_hard_v2.py
python scripts/draw_pipeline.py
```

## Important Notes

Do not commit raw datasets.

Do not commit SSH keys, private keys, or credentials.

Keep only lightweight result summaries in outputs/, preferably JSON and CSV files.

If shell scripts still use old root-level paths, update them after moving files into scripts/.

# Author

Congpx
