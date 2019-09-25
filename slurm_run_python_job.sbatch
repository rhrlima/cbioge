#!/bin/bash

# This example makes use of user variables. These have to be passed
# when the script is launched:

# sbatch slurm_run_job.sbatch --export=n=20,evaluations=1000

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=unetexp

# Use this parameter to define the output and error files
# For job arrays, the %A represents the job ID and %a the array index
# Also, you can use your own variables to make the file name more readable

mkdir -p outputs
#SBATCH --output=outputs/tftest_%A_%a.out                                                                                            
#SBATCH --error=outputs/tftest_%A_%a.err

# Number of cores to be used. This option uses only one core                    
#SBATCH -n 4
# This option would use one node with all its cores
# #SBATCH -N 1                          
#SBATCH --ntasks=1

# Remember to ask for enough memory for your process. The default is 2Gb
#SBATCH --mem-per-cpu=2048                                                                                       
# It is mandatory to indicate a maximum time for the process. 
# If you have no idea about how much time it will take, set it at a big value
#SBATCH --time=4:00:00   
# So far there is only one queue 
#SBATCH --partition=2018allq                                                                                           

# Determine the number of repetitions of the process
#SBATCH --array=1-1
# #SBATCH --array=1-50 

# Leave these options commented                                                      
# #SBATCH --cpus-per-task=1                                                                                                 
# #SBATCH --threads-per-core=2                                                                                              
# #SBATCH --ntasks-per-core=2        

# ONLY USE TENSORFLOW ABLE NODES
#SBATCH -x nodo28,nodo29,nodo30,nodo31,nodo32,nodo33,nodo34,nodo35,nodo36,nodo37,nodo38,nodo39,nodo40,nodo41,nodo42,nodo43,nodo45,nodo46,nodo47,nodo48,nodo49,nodo50,nodo51,nodo52,nodo53,nodo54,nodo55,nodo56,nodo57,nodo58,nodo59,nodo60,nodo61,nodo62,nodo63,nodo64,nodo65,nodo66,nodo67,nodo68,nodo69,nodo70,nodo71,nodo72,nodo73,nodo74,nodo75,nodo76,nodo77,nodo78,nodo79


# Define and create a unique scratch directory for this job

SCRATCH_DIRECTORY=/var/tmp/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from
cp -r ${SLURM_SUBMIT_DIR}/../tf-deep-evolution ${SCRATCH_DIRECTORY}
cd tf-deep-evolution/

# This is where the actual work is done. In this case, the script only waits.
# R CMD BATCH --vanilla --quiet --slave '--args n="'$n'" evaluations='$evaluations''  main.R
python3 -m examples.unet_membrane

# After the job is done we copy our output back to $SLURM_SUBMIT_DIR 
# You can use SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID to avoid overwritting files when launched as an array
# cp ${SCRATCH_DIRECTORY}/results.csv ${SLURM_SUBMIT_DIR}/results/reskali_$n\_$evaluations\_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}.csv
cp -r ${SCRATCH_DIRECTORY}/tf-deep-evolution/datasets/membrane/test/pred/* ${SLURM_SUBMIT_DIR}/outputs/

# After everything is saved to the home directory, delete the work directory 
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
