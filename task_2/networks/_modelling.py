from ._deeplab import DeepLabHeadV3Plus, DeepLabV3
from .resnet import resnet32, resnet50
from .utils import IntermediateLayerGetter

backbones_dict = {
    "resnet32": resnet32,
    "resnet50": resnet50,
}


def _segm_resnet(
    name, backbone_name, num_classes, output_stride, pretrained_backbone
):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    # some hacky ways to run resnet32
    if backbone_name == "resnet32":
        replace_stride_with_dilation = [False, False, False]
        inplanes = 512
        low_level_planes = 64
    elif backbone_name == "resnet50":
        inplanes = 2048
        low_level_planes = 256
    else:
        raise NotImplementedError

    backbone = backbones_dict[backbone_name](
        num_classes=num_classes,
        replace_stride_with_dilation=replace_stride_with_dilation,
        weights_path=None,
    )

    # inplanes = 2048
    # low_level_planes = 256

    if name == "deeplabv3plus":
        return_layers = {"layer4": "out", "layer1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    else:
        raise NotImplementedError
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(
    arch_type, backbone, num_classes, output_stride, pretrained_backbone
):
    if backbone.startswith("resnet"):
        model = _segm_resnet(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        raise NotImplementedError
    return model


def deeplabv3plus_resnet32(
    num_classes=21, output_stride=16, pretrained_backbone=False
):
    """Constructs a DeepLabV3 model with a ResNet-32 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        arch_type="deeplabv3plus",
        backbone="resnet32",
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_resnet50(
    num_classes=21, output_stride=16, pretrained_backbone=False
):
    """Constructs a DeepLabV3 model with a ResNet-32 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        arch_type="deeplabv3plus",
        backbone="resnet50",
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
