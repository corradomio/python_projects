import mitsuba as mi
from matplotlib import pyplot as plt

import mitsubax as mix

# 1. Set the rendering variant (required before any Mitsuba operations)
# 'scalar_rgb' executes sequentially on the CPU.
# Use 'cuda_ad_rgb' for Nvidia GPUs or 'llvm_ad_rgb' for parallel CPU execution.
mi.set_variant('cuda_ad_rgb')


def render_xml_scene(xml_path, output_path):
    print(f"Loading scene: {xml_path}")
    # 2. Load the scene from the XML file
    scene = mi.load_file(xml_path)

    print("Rendering...")
    # 3. Render the scene using the default sensor inside the XML
    # You can optionally pass `spp=64` to change samples per pixel
    image = mix.render(scene)

    print(f"Saving image to: {output_path}")
    # 4. Convert the rendered tensor to a Bitmap and save it (e.g., .exr, .png)
    # mi.Bitmap(image).write(output_path)
    plt.imsave(output_path, image)
    print("Rendering complete!")


if __name__ == "__main__":
    # Replace with your actual paths
    xml_scene_file = r"examples\banner_05\scene.xml"
    output_image_file = r"examples\banner_05\output.png"

    render_xml_scene(xml_scene_file, output_image_file)
