#
# https://kaiyangzhou.github.io/deep-person-reid/
# https://kaiyangzhou.github.io/deep-person-reid/
#

# model    : class
# ---------------------------------------------------------------------------
# resnet18 : torchreid.models.resnet.ResNet
# resnet34 : torchreid.models.resnet.ResNet
# resnet50 : torchreid.models.resnet.ResNet
# resnet101 : torchreid.models.resnet.ResNet
# resnet152 : torchreid.models.resnet.ResNet
# resnext50_32x4d : torchreid.models.resnet.ResNet
# resnext101_32x8d : torchreid.models.resnet.ResNet
# resnet50_fc512 : torchreid.models.resnet.ResNet
# se_resnet50 : torchreid.models.senet.SENet
# se_resnet50_fc512 : torchreid.models.senet.SENet
# se_resnet101 : torchreid.models.senet.SENet
# se_resnext50_32x4d : torchreid.models.senet.SENet
# se_resnext101_32x4d : torchreid.models.senet.SENet
# densenet121 : torchreid.models.densenet.DenseNet
# densenet169 : torchreid.models.densenet.DenseNet
# densenet201 : torchreid.models.densenet.DenseNet
# densenet161 : torchreid.models.densenet.DenseNet
# densenet121_fc512 : torchreid.models.densenet.DenseNet
# inceptionresnetv2 : torchreid.models.inceptionresnetv2.InceptionResNetV2
# inceptionv4 : torchreid.models.inceptionv4.InceptionV4
# xception : torchreid.models.xception.Xception
# resnet50_ibn_a : torchreid.models.resnet_ibn_a.ResNet
# resnet50_ibn_b : torchreid.models.resnet_ibn_b.ResNet
# nasnsetmobile : torchreid.models.nasnet.NASNetAMobile
# mobilenetv2_x1_0 : torchreid.models.mobilenetv2.MobileNetV2
# mobilenetv2_x1_4 : torchreid.models.mobilenetv2.MobileNetV2
# shufflenet : torchreid.models.shufflenet.ShuffleNet
# squeezenet1_0 : torchreid.models.squeezenet.SqueezeNet
# squeezenet1_0_fc512 : torchreid.models.squeezenet.SqueezeNet
# squeezenet1_1 : torchreid.models.squeezenet.SqueezeNet
# shufflenet_v2_x0_5 : torchreid.models.shufflenetv2.ShuffleNetV2
# shufflenet_v2_x1_0 : torchreid.models.shufflenetv2.ShuffleNetV2
# shufflenet_v2_x1_5 : torchreid.models.shufflenetv2.ShuffleNetV2
# shufflenet_v2_x2_0 : torchreid.models.shufflenetv2.ShuffleNetV2
# mudeep : torchreid.models.mudeep.MuDeep
# resnet50mid : torchreid.models.resnetmid.ResNetMid
# hacnn : torchreid.models.hacnn.HACNN
# pcb_p6 : torchreid.models.pcb.PCB
# pcb_p4 : torchreid.models.pcb.PCB
# mlfn : torchreid.models.mlfn.MLFN
# osnet_x1_0 : torchreid.models.osnet.OSNet
# osnet_x0_75 : torchreid.models.osnet.OSNet
# osnet_x0_5 : torchreid.models.osnet.OSNet
# osnet_x0_25 : torchreid.models.osnet.OSNet
# osnet_ibn_x1_0 : torchreid.models.osnet.OSNet
# osnet_ain_x1_0 : torchreid.models.osnet_ain.OSNet
# osnet_ain_x0_75 : torchreid.models.osnet_ain.OSNet
# osnet_ain_x0_5 : torchreid.models.osnet_ain.OSNet
# osnet_ain_x0_25 : torchreid.models.osnet_ain.OSNet
#

import sys
import torchreid
from pprint import pprint
from stdlib.qname import qualified_name

from torchreid.models.resnet import ResNet


def main(argv):
    names = list(torchreid.models.__model_factory.keys())
    pprint(names)
    for name in names:
        # print("--", name, "--")
        # ResNet | SENet
        model: ResNet = torchreid.models.build_model(name, num_classes=1000)
        print(name, ":", qualified_name(type(model)))
        pass
    pass


if __name__ == "__main__":
    main(sys.argv)