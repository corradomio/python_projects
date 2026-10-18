#
#
import numpy as np


def kelvin_to_rgb(temp: int) -> np.ndarray:
    """

    :param temp: color temperature in Kelvin degree
    :return:
    """
    temp = np.clip(temp, 1000, 40000) / 100.0

    # Calcolo del Rosso
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)

    # Calcolo del Verde
    if temp <= 66:
        g = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)

    # Calcolo del Blu
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177412231 * np.log(temp - 10) - 305.0447927307

    rgb = np.array([r, g, b], dtype=float) / 255.0
    return np.clip(rgb, 0.0, 1.0)


def kelvin_to_gains(temp_target: int, temp_origin: int=6500) -> np.ndarray:
    assert 1000 <= temp_target <= 40000
    assert 1000 <= temp_origin <= 40000

    rgb_ref = kelvin_to_rgb(temp_origin)
    rgb_target = kelvin_to_rgb(temp_target)

    gains = rgb_target / rgb_ref
    return gains


def apply_color_temperature(image:np.ndarray, temp_target:int, temp_origin:int=6500):
    assert isinstance(image, np.ndarray)
    assert image.dtype in [float, np.float16, np.float32, np.float64]
    assert 0 <= image.min() <= 1.0
    assert 0 <= image.max() <= 1.0

    gains = kelvin_to_gains(temp_target, temp_origin)

    return image * gains

