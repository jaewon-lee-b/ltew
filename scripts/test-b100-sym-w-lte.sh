echo 'b100' &&
echo 'x2' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-2.yaml --model $1 --gpu $2 &&
echo 'x3' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-3.yaml --model $1 --gpu $2 &&
echo 'x4' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-4.yaml --model $1 --gpu $2 &&
echo 'x6*' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-6.yaml --model $1 --gpu $2 &&
echo 'x8*' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-8.yaml --model $1 --gpu $2 &&

true
