from gnm.shape import gnm_numpy  # Or gnm_jax, gnm_pytorch, gnm_tensorflow

# 1. Load from Hugging Face Hub (via huggingface_hub SDK or HTTPS CDN fallback)
gnm = gnm_numpy.GNM.from_huggingface(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)

# 2. Load from Kaggle Models (via kagglehub SDK)
gnm = gnm_numpy.GNM.from_kaggle(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)

# 3. Load via HTTPS CDN (no external SDK required, automatic local caching)
gnm = gnm_numpy.GNM.from_remote(
    version=gnm_numpy.GNMMajorVersion.V3,
    variant=gnm_numpy.GNMVariant.HEAD,
)

