ROOT='./'
cd ${ROOT}

export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=1
cfg_name=EPCL_HD
cfg_file=tools/cfgs/voxel/nuscenes/${cfg_name}.yaml
extra_tag=val_${cfg_name}_nuscenes_x5
pretrained_model=/root/main/EPCL_setup/checkpoints/best_checkpoint.pth

set -x

#while true
#do
#    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#    if [ "${status}" != "0" ]; then
#        break;
#    fi
#done
#echo $PORT

python -m torch.distributed.launch \
--nproc_per_node=${NGPUS} train.py \
--launcher pytorch \
--eval \
--exp 5 \
--pretrained_model ${pretrained_model} \
--cfg_file ${cfg_file} \
--extra_tag ${extra_tag} \

echo 'dist_train finished!'