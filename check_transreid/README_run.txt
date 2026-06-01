# DukeMTMC transformer-based baseline
python train.py --config_file configs/DukeMTMC/vit_base.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC baseline + JPM
python train.py --config_file configs/DukeMTMC/vit_jpm.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC baseline + SIE
python train.py --config_file configs/DukeMTMC/vit_sie.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC TransReID (baseline + SIE + JPM)
python train.py --config_file configs/DukeMTMC/vit_transreid.yml MODEL.DEVICE_ID "('0')"
# DukeMTMC TransReID with stride size [12, 12]
python train.py --config_file configs/DukeMTMC/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# MSMT17
python train.py --config_file configs/MSMT17/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
# OCC_Duke
python train.py --config_file configs/OCC_Duke/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
# Market
python train.py --config_file configs/Market/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"
# VeRi
python train.py --config_file configs/VeRi/vit_transreid_stride.yml MODEL.DEVICE_ID "('0')"

# VehicleID (The dataset is large and we utilize 4 v100 GPUs for training )
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 66666 train.py --config_file configs/VehicleID/vit_transreid_stride.yml MODEL.DIST_TRAIN True
#  or using following commands:
Bash dist_train.sh

-----------------------------------------------------------------------------

save model:
    model = ...
    torch.save(model.state_dict(), PATH)

load model:
    model = ...
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()

-----------------------------------------------------------------------------

model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

if cfg.MODEL.DIST_TRAIN:
    torch.save(model.state_dict(),
               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
else:
    torch.save(model.state_dict(),
               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
