data_dir=./Data/
exp_dir=./Exp/

dataset=msmt
testset=cuhk03_np_detected,market
method=TransMatcher
sub_method=Demo

python main.py --testset $testset --data-dir $data_dir --exp-dir $exp_dir --method $method --sub_method $sub_method --dataset $dataset -a resnet18 -b 8 -j 1 --neck 64 --num_trans_layers 1 --dim_feedforward 128 --epochs 1 --test_gal_batch 32 --test_prob_batch 32 --gs_verbose