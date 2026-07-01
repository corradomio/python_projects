from PIL import Image
import torch

# # typical inputs with 512 dimension
# B = 5
# embbedings = torch.randn((B, 512)).float()  # latent code
# norms = torch.norm(embbedings, 2, -1, keepdim=True)
# normalized_embedding  = embbedings / norms
# labels =  torch.randint(70722, (B,))
#
# # instantiate AdaFace
# adaface = AdaFace(embedding_size=512,
#                   classnum=70722,
#                   m=0.4,
#                   h=0.333,
#                   s=64.,
#                   t_alpha=0.01,)
#
# # calculate loss
# cosine_with_margin = adaface(normalized_embedding, norms, labels)
# loss = torch.nn.CrossEntropyLoss()(cosine_with_margin, labels)

image_path = r"D:\Projects.github\python_projects\check_face_embedding\.maurizio_dataset\2\2_0_DONE\random_crop\20260506_093640_crop_no_margin.jpg"


# from adaface.face_alignment import align
# from adaface.inference import load_pretrained_model, to_input
#
# model = load_pretrained_model('ir_50')
# image = Image.open(image_path).convert('RGB')                           # w h
# # aligned_rgb_img = align.get_aligned_face(image, rgb_pil_image=image)    # 112x112
# aligned_rgb_img = image.resize((112, 112))
# bgr_input = to_input(aligned_rgb_img)
# feature, _ = model(bgr_input)
# pass

from human.adaface import AdaFace, ADAFACE_MODEL_NAMES

for model_name in ADAFACE_MODEL_NAMES:
    feature = AdaFace.represent(image_path, model_name)
    print(feature.shape)