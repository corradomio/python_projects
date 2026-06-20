#
# NO: this is a 'generic' image feature extraction modules
#


from PIL import Image
from accelerate import Accelerator
from transformers import pipeline, CLIPImageProcessor, AutoModel

# AutoModels
# AutoModelForMaskGeneration(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForKeypointDetection(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForKeypointMatching(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTextEncoding(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForImageToImage(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModel(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForPreTraining(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForCausalLM(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForMaskedLM(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForSeq2SeqLM(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForSequenceClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForQuestionAnswering(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTableQuestionAnswering(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForVisualQuestionAnswering(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTokenClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForMultipleChoice(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForNextSentencePrediction(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForImageClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForZeroShotImageClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForImageSegmentation(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForSemanticSegmentation(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTimeSeriesPrediction(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForUniversalSegmentation(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForInstanceSegmentation(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForObjectDetection(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForZeroShotObjectDetection(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForDepthEstimation(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTextRecognition(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTableRecognition(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForVideoClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForImageTextToText(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForMultimodalLM(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForAudioClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForCTC(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTDT(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForSpeechSeq2Seq(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForAudioFrameClassification(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForAudioXVector(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTextToSpectrogram(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForTextToWaveform(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForMaskedImageModeling(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)
# AutoModelForAudioTokenization(_BaseAutoModelClass) (transformers.models.auto.modeling_auto)

# Huggingface "Image Feature Extraction"
#
# facebook    73
# google      7
# timm        208
# nvidia      15
# apple       35
# tiiuae      5
#
# onnx-community      33
# MahmoodLab      10
# py-feat         14
# birder-project  27
# r3gm            19
# Xenova          11
#
# nomica-ai       2
# paige-ai        2
# bioptimus       3
# PIA-SPACE-LAB   3
# NorskRegnesentralSTI    3
# TTPlanet        2
# xtxx            2
# canvit          3
# SixAILab        2
# Snarcy          7
# MiniMaxAI       3
# HopitAI         5
# AvitoTech       6
#
#
# to page 14


FEATURE_EXTRACION_MODELS = [
    "facebook/dinov2-large",
    "facebook/dinov2-small",

    # READ: https://huggingface.co/nvidia/MambaVision-T2-1K
    # "nvidia/RADIO-L",
    # "nvidia/RADIO-B",
    # "nvidia/C-RADIOv2-g",
    # "nvidia/C-RADIOv2-B",
    # "nvidia/RADIO",
    # "nvidia/MambaVision-S-1K",
    # "nvidia/MambaVision-T2-1K",

    "google/vit-base-patch16-224-in21k",
    "google/vit-base-patch32-224-in21k",
    "google/vit-large-patch32-224-in21k",
    "google/vit-large-patch16-224-in21k",
    "google/vit-huge-patch14-224-in21k",
    # "google/hear-pytorch",

    "microsoft/rad-dino",
    "microsoft/latent-zoning-networks",
    "microsoft/rad-dino-maira-2",

    "timm/vit_small_patch14_reg4_dinov2.lvd142m",
    "timm/vit_small_patch14_dinov2.lvd142m",
    "timm/vit_small_patch16_dinov3.lvd1689m",
    "timm/vit_small_patch16_224.dino",
    "timm/convnext_small.dinov3_lvd1689m",
    "timm/vit_small_patch16_dinov3_qkvb.lvd1689m",
    "timm/vit_small_plus_patch16_dinov3.lvd1689m",
    "timm/vit_small_patch8_224.dino",
    "timm/vit_small_plus_patch16_dinov3_qkvb.lvd1689m",
    "timm/vit_pe_spatial_small_patch16_512.fb",
    "timm/eva02_small_patch14_224.mim_in22k",
    "timm/sam2_hiera_small.fb_r896_2pt1",
    "timm/sam2_hiera_small.fb_r896",
    "timm/vit_pe_core_small_patch16_384.fb"
]

# pipeline tasks:
#
# - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
# - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
# - `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
# - `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
# - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
# - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
# - `"image-classification"`: will return a [`ImageClassificationPipeline`].
# - `"image-feature-extraction"`: will return an [`ImageFeatureExtractionPipeline`].
# - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
# - `"image-text-to-text"`: will return a [`ImageTextToTextPipeline`].
# - `"keypoint-matching"`: will return a [`KeypointMatchingPipeline`].
# - `"mask-generation"`: will return a [`MaskGenerationPipeline`].
# - `"object-detection"`: will return a [`ObjectDetectionPipeline`].
# - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
# - `"text-classification"` (alias `"sentiment-analysis"` available): will return a [`TextClassificationPipeline`].
# - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
# - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
# - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
# - `"video-classification"`: will return a [`VideoClassificationPipeline`].
# - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
# - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
# - `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
# - `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].
device = Accelerator().device
# print(device)


for image_feature_extraction_model in FEATURE_EXTRACION_MODELS:
    print(image_feature_extraction_model)
    try:
        if image_feature_extraction_model.startswith("nvidia"):
            hf_repo = image_feature_extraction_model
            image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
            model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True)
            model.eval().cuda()

            image = Image.open("pipeline-cat-chonk.jpeg").convert('RGB')
            pixel_values = image_processor(images=image, return_tensors='pt', do_resize=True).pixel_values
            pixel_values = pixel_values.cuda()
            summary, features = model(pixel_values)
            print("...", type(summary), type(features))
        else:
            processor = pipeline(
                task="image-feature-extraction",
                model=image_feature_extraction_model,
                device=device,
                pool=True
            )
            # ret = pipeline("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg")
            ret = processor("pipeline-cat-chonk.jpeg")

            print("...", type(ret))
            print("...",len(ret), type(ret[0]))
    except Exception as e:
        print(f"... is not supported: {e}")

# print(ret)
