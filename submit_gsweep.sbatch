#!/bin/bash
# Goal-variability sweep on the RCI cluster (CPU — no GPU needed: the search
# runs on a CPU twin, and the bottleneck is capped forward searches).
#SBATCH --job-name=gsweep
#SBATCH --partition=cpu            # CPU partition (check `sinfo`; cpufast caps at 4h)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00            # 4x4 sweep ~4-7h; 5x5 is much slower — bump if needed
#SBATCH --output=gsweep_%j.out
#SBATCH --error=gsweep_%j.err

# Board size: 4 (default) or 5. Override at submit time:
#   sbatch --export=ALL,NBOARD=5 submit_gsweep.sbatch
NBOARD="${NBOARD:-4}"

module purge
ml Anaconda3

# --- activate the conda env (auto-finds conda; edit ENV_NAME if not 'bidir') -
ENV_NAME=bidir
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
# Use the env's python EXPLICITLY. The old broken env fell back to the base
# Anaconda python (whose scipy hits a GLIBCXX error on this node); an explicit
# path cannot. The orchestrator spawns its G subprocesses with sys.executable,
# so they inherit THIS python too.
PY="$HOME/.conda/envs/${ENV_NAME}/bin/python"
# ---------------------------------------------------------------------------

echo "python: $PY  env: $CONDA_DEFAULT_ENV  board: ${NBOARD}x${NBOARD}"
"$PY" -c "from scipy.optimize import linear_sum_assignment; import torch; print('deps ok', torch.__version__)" || exit 1

cd "$SLURM_SUBMIT_DIR"             # submit from the repo root
mkdir -p gsweep_results
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH=.

RG_N="$NBOARD" GSWEEP_GS="1,4,16,256,inf" GSWEEP_NTRAIN=1000 GSWEEP_NEVAL=200 \
  GSWEEP_MAXITERS=10000 FWD_STEPS=8000 \
  "$PY" -u -m rank_forward.experiments.tiles_gsweep \
  > "gsweep_results/run_${NBOARD}x${NBOARD}.log" 2>&1

echo "DONE — table at the end of gsweep_results/run_${NBOARD}x${NBOARD}.log"
