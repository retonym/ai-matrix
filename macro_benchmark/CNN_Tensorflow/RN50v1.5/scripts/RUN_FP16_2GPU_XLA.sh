mpiexec --allow-run-as-root --bind-to socket -np 2 /usr/bin/python main.py --mode=training_benchmark --batch_size=256 --warmup_steps=200 --num_iter=400 --precision=fp32 --iter_unit=batch --data_dir=/data/source_data/build_imagenet_data-rebuild/ --results_dir=/log_fp16_2gpus_xla  --use_tf_amp --use_xla 
