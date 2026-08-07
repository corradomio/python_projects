"""
wideangle_sensor.py
--------------------
A wide-angle / fisheye camera plugin for Mitsuba 3.

Mitsuba 3 does not ship a fisheye sensor out of the box (only
`perspective`, `thinlens`, `orthographic`, etc.), so this implements one
as a Python plugin, following the same subclassing mechanism Mitsuba
uses for custom BSDFs / emitters / integrators
(mi.Sensor -> override sample_ray / sample_ray_differential).

Projection model: equidistant ("f-theta") fisheye.
    r = f * theta
where `theta` is the angle from the optical axis (+Z in camera space)
and `r` is the radial distance from the image center, normalized so that
r = 1 at the edge of the configured field of view. This model supports
FOVs up to (and beyond) 180 degrees, which a pinhole/perspective camera
cannot represent.

Usage
-----
    import mitsuba as mi
    mi.set_variant('llvm_ad_rgb')  # or scalar_rgb, cuda_ad_rgb, ...

    from wideangle_sensor import WideAngleCamera
    mi.register_sensor('wideangle', lambda props: WideAngleCamera(props))

    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': {
            'type': 'wideangle',
            'fov': 180.0,  # degrees, full field of view
            'to_world': mi.ScalarTransform4f().look_at(
                origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            'film': {
                'type': 'hdrfilm',
                'width': 512, 'height': 512,
                'rfilter': {'type': 'gaussian'},
            },
            'sampler': {'type': 'independent', 'sample_count': 64},
        },
        # ... rest of the scene ...
    })

    image = mi.render(scene)
"""

import mitsuba as mi
import drjit as dr

# The class body below resolves `mi.Sensor` at *import* time, so a variant
# must already be active. If the importing script hasn't set one yet
# (e.g. this file is run directly), fall back to 'scalar_rgb'.
# if mi.variant() is None:
#     mi.set_variant('scalar_rgb')

assert mi.variant() is not None, "Variant must be set before importing mitsubax.wideangle_sensor"


class WideAngleCamera(mi.Sensor):
    """Equidistant (f-theta) fisheye sensor."""

    def __init__(self, props: mi.Properties):
        mi.Sensor.__init__(self, props)

        # Full field of view, in degrees (e.g. 180 = hemispherical fisheye).
        # Anything up to ~359.9 is legal since we don't rely on a
        # perspective divide.
        fov = props.get('fov', 180.0)
        if fov <= 0.0 or fov >= 360.0:
            raise RuntimeError("'fov' must lie in (0, 360) degrees.")
        self.m_theta_max = dr.deg2rad(fov * 0.5)

        # Near/far clipping planes, consistent with the built-in sensors.
        self.m_near_clip = props.get('near_clip', 1e-2)
        self.m_far_clip = props.get('far_clip', 1e4)
        if self.m_near_clip <= 0.0 or self.m_near_clip >= self.m_far_clip:
            raise RuntimeError('Invalid clipping range.')

        # This sensor only needs a 2D film sample (no aperture / lens
        # sample), same as the pinhole `perspective` plugin.
        self.m_needs_sample_3 = False

    # ------------------------------------------------------------------
    # Helper: map a normalized, aspect-corrected film-space position in
    # [-1, 1]^2 to a camera-space (local) ray direction.
    # ------------------------------------------------------------------
    def _local_direction(self, film_p: mi.Point2f):
        # film_p.x, film_p.y in [-1, 1], already square-aspect corrected
        r = dr.norm(film_p)
        r_clamped = dr.minimum(r, 1.0)

        # Equidistant fisheye mapping: theta grows linearly with radius.
        theta = r_clamped * self.m_theta_max
        phi = dr.atan2(film_p.y, film_p.x)

        sin_theta, cos_theta = dr.sin(theta), dr.cos(theta)
        sin_phi, cos_phi = dr.sin(phi), dr.cos(phi)

        d = mi.Vector3f(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta)

        # Rays whose radius fell outside the image circle (r > 1) are
        # invalid -- they don't correspond to any real incoming direction.
        valid = r <= 1.0
        return d, valid

    def _film_to_ndc(self, position_sample: mi.Point2f):
        """Map a [0,1]^2 film sample to an aspect-corrected [-1,1]^2 square,
        so the fisheye circle stays circular regardless of film aspect
        ratio (mirrors how `perspective` handles fov_axis)."""
        film_size = mi.Vector2f(self.film().size())
        aspect = film_size.x / film_size.y

        p = mi.Point2f(position_sample) * 2.0 - 1.0  # -> [-1, 1]
        if_wide = aspect > 1.0
        p.x = dr.select(if_wide, p.x * aspect, p.x)
        p.y = dr.select(if_wide, p.y, p.y / aspect)
        return p

    # ------------------------------------------------------------------
    # Required Sensor interface
    # ------------------------------------------------------------------
    def sample_ray(self, time, wavelength_sample, position_sample,
                    aperture_sample, active=True):
        wavelengths, wav_weight = self.sample_wavelengths(
            dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active)

        ndc = self._film_to_ndc(position_sample)
        d_local, valid = self._local_direction(ndc)

        ray = mi.Ray3f()
        ray.time = time
        ray.wavelengths = wavelengths
        ray.o = self.world_transform() @ mi.Point3f(0.0)
        ray.d = self.world_transform() @ d_local

        # Offset the origin along the ray by near_clip so that geometry
        # right at the camera position doesn't get erroneously clipped.
        ray.o = ray.o + ray.d * self.m_near_clip
        ray.maxt = self.m_far_clip - self.m_near_clip

        return ray, wav_weight * mi.Float(dr.select(valid, 1.0, 0.0))

    def sample_ray_differential(self, time, wavelength_sample,
                                 position_sample, aperture_sample,
                                 active=True):
        ray, weight = self.sample_ray(
            time, wavelength_sample, position_sample, aperture_sample, active)

        ray_diff = mi.RayDifferential3f(ray)

        # Finite-difference the ray direction by one pixel in x and y to
        # populate the differential (used for texture filtering / MSAA).
        film_size = mi.Vector2f(self.film().size())
        eps = mi.Point2f(1.0 / film_size.x, 1.0 / film_size.y)

        ndc_x = self._film_to_ndc(position_sample + mi.Point2f(eps.x, 0.0))
        d_x, _ = self._local_direction(ndc_x)
        ray_diff.o_x = ray.o
        ray_diff.d_x = self.world_transform() @ d_x

        ndc_y = self._film_to_ndc(position_sample + mi.Point2f(0.0, eps.y))
        d_y, _ = self._local_direction(ndc_y)
        ray_diff.o_y = ray.o
        ray_diff.d_y = self.world_transform() @ d_y

        ray_diff.has_differentials = True
        return ray_diff, weight

    def world_transform(self):
        # Convenience: fetch the (possibly animated) camera-to-world
        # transform at the sensor's reference time.
        return self.m_to_world.value if hasattr(self.m_to_world, 'value') \
            else self.m_to_world

    def to_string(self):
        return (f"WideAngleCamera[\n"
                f"  fov = {dr.rad2deg(self.m_theta_max) * 2}\n"
                f"  near_clip = {self.m_near_clip}\n"
                f"  far_clip = {self.m_far_clip}\n"
                f"]")


mi.register_sensor('wideangle', lambda props: WideAngleCamera(props))

# def register():
#     """Call once, after `mi.set_variant(...)`, to make 'wideangle'
#     available as a sensor `type` string in load_dict / XML scenes."""
#     mi.register_sensor('wideangle', lambda props: WideAngleCamera(props))
#
#
# if __name__ == '__main__':
#     # Minimal smoke test.
#     register()
#
#     scene = mi.load_dict({
#         'type': 'scene',
#         'integrator': {'type': 'path'},
#         'light': {'type': 'constant', 'radiance': 1.0},
#         'sphere': {'type': 'sphere'},
#         'sensor': {
#             'type': 'wideangle',
#             'fov': 170.0,
#             'to_world': mi.ScalarTransform4f().look_at(
#                 origin=[0, 0, 3], target=[0, 0, 0], up=[0, 1, 0]),
#             'film': {
#                 'type': 'hdrfilm', 'width': 256, 'height': 256,
#                 'rfilter': {'type': 'gaussian'},
#             },
#             'sampler': {'type': 'independent', 'sample_count': 16},
#         },
#     })
#
#     img = mi.render(scene)
#     mi.util.write_bitmap('wideangle_test.png', img)
#     print('Rendered wideangle_test.png')
