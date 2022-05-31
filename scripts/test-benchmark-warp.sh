echo 'set5' &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-set5-warp-in-scale.yaml --model $1 --gpu $2 &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-set5-warp-out-of-scale.yaml --model $1 --gpu $2 &&

echo 'set14' &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-set14-warp-in-scale.yaml --model $1 --gpu $2 &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-set14-warp-out-of-scale.yaml --model $1 --gpu $2 &&

echo 'b100' &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-b100-warp-in-scale.yaml --model $1 --gpu $2 
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-b100-warp-out-of-scale.yaml --model $1 --gpu $2 &&

echo 'urban100' &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-urban100-warp-in-scale.yaml --model $1 --gpu $2 &&
CUDA_VISIBLE_DEVICES=$2 python test_ltew.py --config configs/test/test-urban100-warp-out-of-scale.yaml --model $1 --gpu $2 &&

true
