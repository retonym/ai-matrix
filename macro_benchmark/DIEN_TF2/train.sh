#!/bin/bash

export TF_NUM_INTEROP_THREADS=1

# for ats
export ZE_AFFINITY_MASK=0.0
export SYCL_DEVICE_FILTER={level_zero:gpu:0}
export SYCL_PI_LEVEL_ZERO_BATCH_SIZE=1

export UseVmBind=1
export EnableDirectSubmission=1


NUM_ACCELERATORS=${NUM_ACCELERATORS:-1}
echo "NUM_ACCELERATORS=${NUM_ACCELERATORS}"
TOTAL_RECOMMDS=512000

rm -r dnn_save_path dnn_best_model
mkdir dnn_save_path dnn_best_model

if [ -d results ]; then
    mv results results_$(date +%Y%m%d%H%M%S)
fi
mkdir results

# batchs='256 512 1024'
# batchs=128
# manner=benchmark
# data_type=BF16
# use_static_rnn=True

for batch in $batchs
do
	echo "----------------------------------------------------------------"
	echo "Running training with batch size of $batch"
	echo "----------------------------------------------------------------"
	start=`date +%s%N`
    python script/train.py --mode=train --batch_size=$batch --manner=$manner --data_type=$data_type --use_static_rnn=True |& tee results/result_train_${batch}.txt
	end=`date +%s%N`
	total_time=$(((end-start)/1000000))
    #total_time=`bc <<< "scale = 3; ($end-$start)/1000000000"`
    total_images=$TOTAL_RECOMMDS
    system_performance=$((1000*$total_images/$total_time))
    echo "Total recommendations: $total_images" >> results/result_train_${batch}.txt
    echo "System time in miliseconds is: $total_time" >> results/result_train_${batch}.txt
    echo "System performance in recommendations/second is: $system_performance" >> results/result_train_${batch}.txt
done

python process_results.py --train
