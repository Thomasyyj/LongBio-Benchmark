# export MODEL_PATH="Your directory to the folder of the model (does not include the folder of the model itself)"
# export HOME="home for this project"
models=("Qwen2.5-7B")

DATA_FOLDER="${HOME}/data/bios_qwen"
density=1.0

length=(2)
num_retrievals=(1 2)
skill=("standard")
# generate the data
for model in "${models[@]}"; do
    for len in "${length[@]}"; do
        for skl in "${skill[@]}"; do
            for num_retrieval in "${num_retrievals[@]}"; do
                position_start=0
                position_end=1
                echo "Model$model, Length $len, N_retrieval $num_retrieval, Skill $skl, start $position_start, end $position_end"
                mkdir -p "logs/data/${model}"
                nohup python ./data_utils/prepare_data.py \
                    -af \
                    -l $len \
                    -tok "${MODEL_PATH}/${model}" \
                    -nr $num_retrieval \
                    -ps $position_start \
                    -pe $position_end \
                    -dd $density \
                    -o $DATA_FOLDER \
                    -s $skl\
                    > logs/data/${model}/l_${len}_retrieval_${num_retrieval}_${skl}_${position_start}_${position_end}.log 2>&1 &
            done
        done
    done
done

length=(8 16 32 64 128)
num_retrievals=(1 2 5 10)
skill=("standard")
# generate the data
for model in "${models[@]}"; do
    for len in "${length[@]}"; do
        for skl in "${skill[@]}"; do
            for num_retrieval in "${num_retrievals[@]}"; do
                position_start=0
                position_end=1
                echo "Model$model, Length $len, N_retrieval $num_retrieval, Skill $skl, start $position_start, end $position_end"
                mkdir -p "logs/data/${model}"
                nohup python ./data_utils/prepare_data.py \
                    -af \
                    -l $len \
                    -tok "${MODEL_PATH}/${model}" \
                    -nr $num_retrieval \
                    -ps $position_start \
                    -pe $position_end \
                    -dd $density \
                    -o $DATA_FOLDER \
                    -s $skl\
                    > logs/data/${model}/l_${len}_retrieval_${num_retrieval}_${skl}_${position_start}_${position_end}.log 2>&1 &
            done
        done
    done
done

length=(2 8 16 32 64 128)
num_retrievals=(1)
skill=("paraphrase" "pronoun")
# generate the data
for model in "${models[@]}"; do
    for len in "${length[@]}"; do
        for skl in "${skill[@]}"; do
            for num_retrieval in "${num_retrievals[@]}"; do
                position_start=0
                position_end=1
                echo "Model$model, Length $len, N_retrieval $num_retrieval, Skill $skl, start $position_start, end $position_end"
                mkdir -p "logs/data/${model}"
                nohup python ./data_utils/prepare_data.py \
                    -af \
                    -l $len \
                    -tok "${MODEL_PATH}/${model}" \
                    -nr $num_retrieval \
                    -ps $position_start \
                    -pe $position_end \
                    -dd $density \
                    -o $DATA_FOLDER \
                    -s $skl\
                    > logs/data/${model}/l_${len}_retrieval_${num_retrieval}_${skl}_${position_start}_${position_end}.log 2>&1 &
            done
        done
    done
done

num_retrievals=(2 5)
skill=("rank" "multihop")
# generate the data
for model in "${models[@]}"; do
    for len in "${length[@]}"; do
        for skl in "${skill[@]}"; do
            for num_retrieval in "${num_retrievals[@]}"; do
                position_start=0
                position_end=1
                echo "Model$model, Length $len, N_retrieval $num_retrieval, Skill $skl, start $position_start, end $position_end"
                mkdir -p "logs/data/${model}"
                nohup python ./data_utils/prepare_data.py \
                    -af \
                    -l $len \
                    -tok "${MODEL_PATH}/${model}" \
                    -nr $num_retrieval \
                    -ps $position_start \
                    -pe $position_end \
                    -dd $density \
                    -o $DATA_FOLDER \
                    -s $skl\
                    > logs/data/${model}/l_${len}_retrieval_${num_retrieval}_${skl}_${position_start}_${position_end}.log 2>&1 &
            done
        done
    done
    wait
done

num_retrievals=(2)
skill=("calculation" "twodiff")
# generate the data
for model in "${models[@]}"; do
    for len in "${length[@]}"; do
        for skl in "${skill[@]}"; do
            for num_retrieval in "${num_retrievals[@]}"; do
                position_start=0
                position_end=1
                echo "Model$model, Length $len, N_retrieval $num_retrieval, Skill $skl, start $position_start, end $position_end"
                mkdir -p "logs/data/${model}"
                nohup python ./data_utils/prepare_data.py \
                    -af \
                    -l $len \
                    -tok "${MODEL_PATH}/${model}" \
                    -nr $num_retrieval \
                    -ps $position_start \
                    -pe $position_end \
                    -dd $density \
                    -o $DATA_FOLDER \
                    -s $skl\
                    > logs/data/${model}/l_${len}_retrieval_${num_retrieval}_${skl}_${position_start}_${position_end}.log 2>&1 &
            done
        done
    done
    wait
done
echo "Finished"
