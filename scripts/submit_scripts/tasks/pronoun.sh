# export three env variables: HOME, MODEL_PATH, MODEL_NAME, TOKENIZER_NAME
# check if dp and run_id are passed
if [ $# -lt 2 ]; then
    echo "dp and run_id parameters should be passed"
    echo "example: $0 8 qwen_7b"
    exit 1
fi

dp="$1"
run_id="$2"

MODEL_PATH="${MODEL_PATH}/${MODEL_NAME}"
DATA_PATH="${HOME}/data/bios"
SAVE_PATH="${HOME}/inference_results/${run_id}"
LOG_PATH="../logs/exp_logs"
model=${MODEL_NAME}
tokenizer_model=${TOKENIZER_NAME}
n_gpu=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

position="0_100"
density="1.0"
context_lengths=(2 8 16 32)
n_retrievals=(1)
skills=("pronoun")
for context_length in "${context_lengths[@]}"; do
    batch_size=$((256*${n_gpu}/${context_length}))
    if [ "$batch_size" -gt 64 ]; then
        e_batch_size=1
    else
        e_batch_size=$((64 /${batch_size}))
    fi
    for n_retrieval in "${n_retrievals[@]}"; do
        for skill in "${skills[@]}"; do
            mkdir -p "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}"
            CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ' ' ',') PYTHONPATH="." python scripts/inference.py \
                -af \
                -dp ${dp}\
                -bs $batch_size \
                -eb $e_batch_size \
                -m ${MODEL_PATH} \
                -d "${DATA_PATH}/${tokenizer_model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}/data_${skill}.json" \
                -o "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}" > "${SAVE_PATH}/${model}_${context_length}k_${n_retrieval}_retrieval_density_${density}_position_${position}_${skill}/terminal.log" 2>&1
            echo "${model}_${context_length}k_${skill} finished"
        done
    done
done

echo "Pronoun tasks finished."