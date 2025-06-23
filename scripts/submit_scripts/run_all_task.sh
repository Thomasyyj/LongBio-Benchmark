# export three env variables: HOME, MODEL_PATH, MODEL_NAME, TOKENIZER_NAME
# Example usage:
# export HOME="yyang/long_context_github/LongBio-Benchmark"
# export MODEL_PATH="yyang/models"
# export MODEL_NAME="Qwen2.5-7B-Instruct" 
# export TOKENIZER_NAME="Qwen2.5-7B" 
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/submit_scripts/run_all_tasks.sh 8 run_001 # You must specify the available gpu devices

if [ $# -lt 2 ]; then
    echo "dp and run_id parameters should be passed"
    echo "example: $0 1 run_001"
    exit 1
fi

dp="$1"
run_id="$2"
n_gpu=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ' | wc -w)

# Start the server
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PYTHONPATH="." setsid python scripts/start_vllm_server.py \
    -dp ${dp}\
    -m "${MODEL_PATH}/${MODEL_NAME}"\
    -ng $n_gpu > logs/server.log &
SERVICE_PID=$!
echo "Server_pid: $SERVICE_PID"
sleep 180

tasks=("standard_0shot" "rank_2shot" "pronoun_0shot" "paraphrase_0shot" "multihop_2shot" "IDK_0shot" "cite_2shot" "calculation_0shot" "twodiff_2shot")

for task in "${tasks[@]}"; do
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash scripts/submit_scripts/tasks/${task}.sh ${dp} ${run_id}
done  

kill -TERM -$SERVICE_PID
echo "All tasks have been completed."