#!/bin/sh
#BSUB -q gpuh100
#BSUB -R "select[gpu80gb]"
#BSUB -J sm_
### -- specify that the cores must be on the same host --
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -R "rusage[mem=45GB]"
#BSUB -o sm_%J.out
#BSUB -e sm_%J.err

module load python3/3.10.12
module load cuda/11.8
source /dtu/blackhole/14/189044/atde/bone_voyage_env/bin/activate

python src/train_label.py --batch-size 8 --model-depth 18 --no-debug