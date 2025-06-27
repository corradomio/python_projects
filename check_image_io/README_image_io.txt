Package per leggere/scrivere immagini

    PIL (pillow)
    cv2 (opencv-python)     attenzione a NON avere 'opencv-python-headless'

    skimage (scikit-image)
    imageio (imageio)


Quale usare?


skimage si appoggia ad altre librerie mediante dei plugins

    {
        "fits": ["imread", "imread_collection"],
        "gdal": ["imread", "imread_collection"],
        "imageio": ["imread", "imsave", "imread_collection"],
        "imread": ["imread", "imsave", "imread_collection"],
        "matplotlib": ["imshow", "imread", "imshow_collection", "imread_collection"],
        "pil": ["imread", "imsave", "imread_collection"],
        "simpleitk": ["imread", "imsave", "imread_collection"],
        "tifffile": ["imread", "imsave", "imread_collection"]
    }
