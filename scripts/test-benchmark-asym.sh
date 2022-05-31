echo 'set5' &&
echo 'x1.50x4.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set5-1.50-4.00.yaml --model $1 --gpu $2 &&
echo 'x1.50x3.50' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set5-1.50-3.50.yaml --model $1 --gpu $2 &&
echo 'x1.60x3.05' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set5-1.60-3.05.yaml --model $1 --gpu $2 &&
echo 'x3.00x8.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set5-3.00-8.00.yaml --model $1 --gpu $2 &&
echo 'x3.00x7.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set5-3.00-7.00.yaml --model $1 --gpu $2 &&
echo 'x3.20x6.10' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set5-3.20-6.10.yaml --model $1 --gpu $2 &&

echo 'set14' &&
echo 'x4.00x2.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set14-4.00-2.00.yaml --model $1 --gpu $2 &&
echo 'x3.50x2.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set14-3.50-2.00.yaml --model $1 --gpu $2 &&
echo 'x3.50x1.75' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set14-3.50-1.75.yaml --model $1 --gpu $2 &&
echo 'x8.00x4.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set14-8.00-4.00.yaml --model $1 --gpu $2 &&
echo 'x7.00x4.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set14-7.00-4.00.yaml --model $1 --gpu $2 &&
echo 'x7.00x3.50' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-set14-7.00-3.50.yaml --model $1 --gpu $2 &&

echo 'b100' &&
echo 'x4.00x1.40' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-4.00-1.40.yaml --model $1 --gpu $2 &&
echo 'x1.50x3.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-1.50-3.00.yaml --model $1 --gpu $2 &&
echo 'x3.50x1.45' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-3.50-1.45.yaml --model $1 --gpu $2 &&
echo 'x8.00x2.80' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-8.00-2.80.yaml --model $1 --gpu $2 &&
echo 'x3.00x6.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-3.00-6.00.yaml --model $1 --gpu $2 &&
echo 'x7.00x2.90' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-b100-7.00-2.90.yaml --model $1 --gpu $2 &&

echo 'urban100' &&
echo 'x1.60x3.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-urban100-1.60-3.00.yaml --model $1 --gpu $2 &&
echo 'x1.60x3.80' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-urban100-1.60-3.80.yaml --model $1 --gpu $2 &&
echo 'x3.55x1.55' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-urban100-3.55-1.55.yaml --model $1 --gpu $2 &&
echo 'x3.20x6.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-urban100-3.20-6.00.yaml --model $1 --gpu $2 &&
echo 'x3.20x7.60' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-urban100-3.20-7.60.yaml --model $1 --gpu $2 &&
echo 'x7.10x3.10' &&
CUDA_VISIBLE_DEVICES=$2 python test_lte.py --config ./configs/test/test-urban100-7.10-3.10.yaml --model $1 --gpu $2 &&

true
