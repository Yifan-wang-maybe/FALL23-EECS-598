#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=I2SB_Test_Version_no_condx
#SBATCH --mail-user=wangyfan@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --account=chuan0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G

# eecs598s007f23_class

# The application(s) to execute along with its input arguments and options:

python3 train.py --name TestP2_4 --n-gpu-per-node 1 --corrupt mixture --dataset-dir Sample_Dataset --dataroot dataset/NLST_with_second_2023/image --dataroot_demo dataset/NLST_with_second_2023/demo --batch-size 16 --microbatch 2 --beta-max 1 --log-dir First_test --image-size 64 --clip-denoise


#python -m visdom.server
