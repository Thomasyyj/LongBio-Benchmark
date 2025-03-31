# export three env variables: HOME, MODEL_PATH, MODEL_NAME, TOKENIZER_NAME
MODEL_PATH="${MODEL_PATH}/${MODEL_NAME}"
DATA_PATH="${HOME}/data/bios_qwen"
SAVE_PATH="${HOME}/inference_results/qwen"
LOG_PATH="../logs/exp_logs"
model=${MODEL_NAME}
tokenizer_model=${TOKENIZER_NAME}
n_gpu=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)
dp=8

# Start the server
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH="." setsid python scripts/start_vllm_server.py \
    -dp ${dp}\
    -m ${MODEL_PATH}\
    -ng $n_gpu > logs/server.log &
SERVICE_PID=$!
echo "Server_pid: $SERVICE_PID"
sleep 180

position="0_100"
density="1.0"
context_lengths=(8 16 32)
n_retrievals=(1 2 5 10)
skills=("standard")
# skills=("rank" "standard")
for context_length in "${context_lengths[@]}"; do
    batch_size=$((64*${n_gpu}/${context_length}))
    if [ "$batch_size" -gt 64 ]; then
        e_batch_size=1
    else
        e_batch_size=$((64 /${batch_size}))
    fi
    for n_retrieval in "${n_retrievals[@]}"; do
        for skill in "${skills[@]}"; do
            mkdir -p "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}_2shot"
            CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ' ' ',') PYTHONPATH="." python scripts/inference.py \
                -af \
                -dp ${dp}\
                -bs $batch_size \
                -eb $e_batch_size \
                -m ${MODEL_PATH} \
                -d "${DATA_PATH}/${tokenizer_model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}/data_${skill}.json" \
                -o "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}" > "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}_2shot/terminal.log" 2>&1
            echo "${model}_${context_length}k_${skill} finished"
        done
    done
done


context_lengths=(2)
n_retrievals=(1 2)
for context_length in "${context_lengths[@]}"; do
    batch_size=$((256*${n_gpu}/${context_length}))
    if [ "$batch_size" -gt 64 ]; then
        e_batch_size=1
    else
        e_batch_size=$((64 /${batch_size}))
    fi
    for n_retrieval in "${n_retrievals[@]}"; do
        for skill in "${skills[@]}"; do
            mkdir -p "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}_2shot"
            CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ' ' ',') PYTHONPATH="." python scripts/inference.py \
                -af \
                -bs $batch_size \
                -eb $e_batch_size \
                -m ${MODEL_PATH} \
                -d "${DATA_PATH}/${tokenizer_model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}/data_${skill}.json" \
                -o "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}" > "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}_2shot/terminal.log" 2>&1
            echo "${model}_${context_length}k_${skill} finished"
        done
    done
done


position="0_100"
density="1.0"
context_lengths=(2 8 16 32)
n_retrievals=(1)
skills=("IDK")
for context_length in "${context_lengths[@]}"; do
    batch_size=$((256*${n_gpu}/${context_length}))
    if [ "$batch_size" -gt 64 ]; then
        e_batch_size=1
    else
        e_batch_size=$((64/${batch_size}))
    fi
    for n_retrieval in "${n_retrievals[@]}"; do
        for skill in "${skills[@]}"; do
            mkdir -p "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}"
            CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ' ' ',') PYTHONPATH="." python scripts/IDK/inference.py \
                -bs $batch_size \
                -eb $e_batch_size \
                -m ${MODEL_PATH} \
                -d "${DATA_PATH}/${tokenizer_model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}/data_${skill}.json" \
                -o "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}" > "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}/terminal.log" 2>&1
            echo "${model}_${context_length}k_${skill} finished"
        done
    done
done

kill -TERM -$SERVICE_PID
echo "All tasks have been completed."

