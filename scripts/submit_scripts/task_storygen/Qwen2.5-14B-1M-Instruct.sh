MODEL_PATH="/cpfs01/shared/XNLP_H800/hf_hub/Qwen2.5-14B-1M-Instruct"
EVAL_MODEL_PATH="/cpfs01/shared/XNLP_H800/hf_hub/Qwen2.5-14B-Instruct-1M"
DATA_PATH="/cpfs01/shared/XNLP_H800/zeyuhuang/longcontext/data/bios"
SAVE_PATH="/cpfs01/shared/XNLP_H800/zeyuhuang/longcontext/inference_results/1106"
model="Qwen2.5-14B-1M-Instruct"
tokenizer_model="Qwen2.5-7B-1M-Instruct"

IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"

n_gpu=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

# Start the server
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH="." python scripts/start_vllm_server.py \
    -m ${MODEL_PATH}\
    -ng $n_gpu > logs/server_${model}.log 2>&1 &
SERVICE_PID1=$!
echo "Server1_pid: $SERVICE_PID1"

sleep 180

position="0_100"
density="1.0"
skills=("storygen")
context_lengths=(2 8 16 32 64)
n_retrievals=(2 5 10)
for context_length in "${context_lengths[@]}"; do
    batch_size=$((128*${n_gpu}/${context_length}))
    e_batch_size=$((128/${batch_size}))
    for n_retrieval in "${n_retrievals[@]}"; do
        for skill in "${skills[@]}"; do
            mkdir -p "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}"
            CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ' ' ',') PYTHONPATH="." python scripts/inference.py \
                -db \
                -bs $batch_size \
                -eb $e_batch_size \
                -m ${MODEL_PATH} \
                -d "${DATA_PATH}/${tokenizer_model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}/data_${skill}.json" \
                -o "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}" > "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}/terminal.log" 2>&1
            echo "${model}_${context_length}k_${skill} finished"
        done
    done
done

pkill -P $SERVICE_PID1
pkill -P $SERVICE_PID2