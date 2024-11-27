seq_len=168
model=GPT4TS
percent=100
is_time=0
lr=0.0001

for pred_len in 24 72 120 168
do
python main.py \
    --root_path ./datasets/ \
    --data_path AK_G0202610_load.csv \
    --model_id AK_G0202610_load_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 168 \
    --pred_len $pred_len \
    --batch_size 128 \
    --lradj type4 \
    --learning_rate $lr \
    --train_epochs 100 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 1 \
    --c_out 1 \
    --freq 0 \
    --patch_size 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1 \
    --target electricity \
    --features MS \
    --is_time $is_time \

done
