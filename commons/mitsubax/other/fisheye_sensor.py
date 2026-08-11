import mitsuba as mi
import drjit as dr


class FisheyeSensor(mi.Sensor):
    """
    Custom 180-degree Equidistant Fisheye Sensor for Mitsuba 3.
    Maps square screen coordinates to a hemispherical viewing projection.
    """

    def __init__(self, props):
        super().__init__(props)
        # Fetch standard configuration parameters
        self.m_to_world = props.get('to_world', mi.ScalarTransform4f())
        pass

    def sample_ray(self, time, sample_wavelengths, position_sample, aperture_sample, active=True):
        # 1. Initialize film and sample maps
        film = self.film()
        film_size = mi.Vector2f(film.size())

        # 2. Normalize screen coordinates to [-1, 1] range
        # position_sample provides pixel coordinates from [0, width] and [0, height]
        p = (position_sample / film_size) * 2.0 - 1.0

        # Calculate radius on the image plane
        r = dr.norm(p)

        # 3. Equidistant mapping (r = f * theta)
        # Normalize the maximum field of view to a 180-degree hemisphere (pi / 2 rad half-angle)
        theta = r * (dr.pi / 2.0)
        phi = dr.atan2(p.y, p.x)

        # 4. Generate local ray direction vector (Hemispherical mapping)
        # Outward direction tracking relative to standard camera look-at (+Z forward)
        x = dr.sin(theta) * dr.cos(phi)
        y = dr.sin(theta) * dr.sin(phi)
        z = dr.cos(theta)

        local_dir = mi.Vector3f(-x, -y, z)  # Flips right-handed orientation matching standard sensors

        # 5. Transform ray to world space
        # Ray origins originate from the camera center (pinhole model)
        ray_origin = self.m_to_world.value() @ mi.Point3f(0.0, 0.0, 0.0)
        ray_direction = self.m_to_world.value() @ local_dir

        # Construct the Ray wavefront
        ray = mi.Ray3f(ray_origin, dr.normalize(ray_direction), time, sample_wavelengths)

        # Mask out-of-bounds rays falling outside the physical circular fisheye boundary (r > 1)
        valid_ray = active & (r <= 1.0)

        # Return ray, ray weight (importance), and mask validity
        return ray, mi.Spectrum(dr.select(valid_ray, 1.0, 0.0))

    def sample_ray_differential(self, time, sample_wavelengths, position_sample, aperture_sample, active=True):
        # Fallback to base ray sampling if explicit ray differentials aren't required
        ray, weight = self.sample_ray(time, sample_wavelengths, position_sample, aperture_sample, active)
        return mi.RayDifferential3f(ray), weight

    def to_string(self):
        return "FisheyeSensor[]"


# Register the Python class cleanly as a Mitsuba plugin loader
mi.register_sensor("fisheye", lambda props: FisheyeSensor(props))
