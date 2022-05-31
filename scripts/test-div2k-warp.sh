echo 'div2k' &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-div2k-warp-in-scale.yaml --model $1 --gpu $2 &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-div2k-warp-out-of-scale.yaml --model $1 --gpu $2 &&

true
