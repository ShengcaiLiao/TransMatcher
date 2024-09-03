
data_dir=Data/Person/
testset=market
source_dir=TransMatcher/matching/

cd ${source_dir}

exp_dir=exp

python test_matching.py --testset $testset --data-dir $data_dir --exp-dir $exp_dir --neck 512 --nhead 1 --num_trans_layers 3 --dim_feedforward 2048 --dropout 0 --test_fea_batch 256 --test_gal_batch 256 --test_prob_batch 256