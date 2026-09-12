"""
A pure-Python reimplementation of Mitsuba 3's built-in ``perspective`` sensor.

Registered as sensor type ``"perspective_py"``.

Design note
-----------
The one piece of logic that is *not* reimplemented here is the construction of
the camera-to-sample projection matrix. Mitsuba exposes the exact same helper
that ``perspective.cpp`` uses -- ``mi.perspective_projection`` -- to Python, so
this plugin calls it directly. That guarantees bit-identical handling of the
field of view, aspect ratio, crop window and near/far remapping instead of
relying on a hand-copied matrix.

Everything else (FOV parsing, ray generation, ray differentials, direction
sampling / importance, traversal) is reimplemented and is the part you would
edit to change behaviour.

Usage
-----
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")     # variant FIRST
    import perspective_sensor         # registers the plugin

    sensor = mi.load_dict({
        "type": "perspective",
        "fov": 45,
        "fov_axis": "x",
        "to_world": mi.ScalarTransform4f().look_at(
            origin=[0, 0, -4], target=[0, 0, 0], up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": 256, "height": 256},
        "sampler": {"type": "independent", "sample_count": 16},
    })

Known deviations from the native plugin
---------------------------------------
* Because Python cannot subclass ``ProjectiveCamera``, this derives from
  ``mi.Sensor``. Anything that type-checks for a ``ProjectiveCamera`` will not
  accept it -- in particular the *projective* / *reparameterising* integrators
  (``direct_projective``, ``prb_projective``, ``prb_reparam``). Use ``path``,
  ``prb``, ``ptracer``, ``volpath``.
* ``near_clip()`` / ``far_clip()`` / ``focus_distance()`` are provided as plain
  Python methods rather than the C++ accessors.
* Not differentiable w.r.t. camera parameters in the way the native plugin is;
  ``x_fov`` and ``to_world`` are exposed via ``traverse()`` but gradients
  through Python-side matrix construction are not guaranteed.
"""

import math

import drjit as dr
import mitsuba as mi


# ---------------------------------------------------------------------------
# Reimplementation of Mitsuba's C++ `parse_fov(props, aspect)` helper
# ---------------------------------------------------------------------------

# Standard 35mm film gate, in millimetres. `focal_length` is interpreted as a
# 35mm-equivalent focal length and converted to a *diagonal* field of view.
FILM_35MM_WIDTH = 36.0
FILM_35MM_HEIGHT = 24.0


def parse_fov(props, aspect):
    """Return the horizontal field of view in degrees.

    ``aspect`` is ``film.size().x / film.size().y`` (the *full* film size, not
    the crop window). Mirrors the semantics of Mitsuba's ``parse_fov``:

    * ``fov`` and ``focal_length`` are mutually exclusive.
    * If neither is given, the default is ``focal_length = "50mm"``.
    * ``fov_axis`` is only meaningful together with ``fov`` and may be one of
      ``x`` (default), ``y``, ``diagonal``, ``smaller``, ``larger``.
    """
    has_fov = props.has_property("fov")
    has_focal = props.has_property("focal_length")

    if has_fov and has_focal:
        raise RuntimeError(
            "Please specify either a focal length ('focal_length') or a "
            "field of view ('fov')!"
        )

    if has_fov:
        fov = float(props["fov"])
        fov_axis = str(props.get("fov_axis", "x")).lower()
    else:
        raw = str(props.get("focal_length", "50mm")).strip()
        if raw.endswith("mm"):
            raw = raw[:-2]
        try:
            focal_length = float(raw)
        except ValueError:
            raise RuntimeError(
                "The focal length must be specified as a floating point "
                "value in millimeters (e.g. '50mm')!"
            )
        diagonal_mm = math.hypot(FILM_35MM_WIDTH, FILM_35MM_HEIGHT)
        fov = 2.0 * math.degrees(math.atan(diagonal_mm / (2.0 * focal_length)))
        fov_axis = "diagonal"

    # Resolve the relative axis names against the actual image shape.
    if fov_axis == "smaller":
        fov_axis = "x" if aspect < 1.0 else "y"
    elif fov_axis == "larger":
        fov_axis = "x" if aspect > 1.0 else "y"

    if fov_axis == "x":
        # Already what we want.
        return fov

    if fov_axis == "y":
        # half_h = tan(fov_y/2); half_w = aspect * half_h
        half_h = math.tan(math.radians(fov) * 0.5)
        return math.degrees(2.0 * math.atan(half_h * aspect))

    if fov_axis == "diagonal":
        # At unit distance: w = 2*half_w, h = w/aspect,
        # diag = sqrt(w^2 + h^2) = w * sqrt(1 + 1/aspect^2)
        diagonal = 2.0 * math.tan(0.5 * math.radians(fov))
        width = diagonal / math.sqrt(1.0 + 1.0 / (aspect * aspect))
        return math.degrees(2.0 * math.atan(width * 0.5))

    raise RuntimeError(
        "Invalid parameter 'fov_axis': expected one of 'x', 'y', 'diagonal', "
        f"'smaller', 'larger' -- got {fov_axis!r}"
    )


# ---------------------------------------------------------------------------
# The sensor
# ---------------------------------------------------------------------------

class PerspectiveCamera(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)

        # `mi.Sensor.__init__` has already extracted the film, the sampler,
        # shutter_open / shutter_close and to_world from `props`. We re-read
        # to_world because the protected C++ member `m_to_world` is not
        # reliably writable from a Python subclass.

        to_world = props.get("to_world", mi.ScalarTransform4f())
        if isinstance(to_world, mi.ScalarTransform4d):
            to_world = mi.ScalarAffineTransform4f(to_world.matrix)
            # to_world = mi.ScalarAffineTransform4f(list(to_world.matrix))
            # matrix_4f = mi.ScalarMatrix4f(list(to_world.matrix))
            # to_world = mi.ScalarAffineTransform4f(matrix_4f)

        self.m_to_world = mi.Transform4f(
            # props.get("to_world", mi.ScalarTransform4f())
            to_world
        )

        if self.m_to_world.has_scale():
            raise RuntimeError(
                "Scale factors in the camera-to-world transformation are not "
                "allowed!"
            )

        # ProjectiveCamera would normally parse these for us.
        self.m_near_clip = float(props.get("near_clip", 1e-2))
        self.m_far_clip = float(props.get("far_clip", 1e4))
        self.m_focus_distance = float(props.get("focus_distance", 0.0))

        if self.m_near_clip <= 0.0:
            raise RuntimeError("near_clip must be greater than zero!")
        if self.m_far_clip <= self.m_near_clip:
            raise RuntimeError("far_clip must be greater than near_clip!")

        film_size = self.film().size()
        aspect = film_size[0] / film_size[1]
        self.m_x_fov = parse_fov(props, aspect)

        # Principal point offset, expressed as a *fraction of the image*
        # (not pixels) -- the cx/cy of a pinhole intrinsics matrix.
        self.m_principal_point_offset = mi.ScalarPoint2f(
            float(props.get("principal_point_offset_x", 0.0)),
            float(props.get("principal_point_offset_y", 0.0)),
        )

        # A pinhole needs no aperture sample. `needs_aperture_sample()`
        # returns False by default, so nothing to do here.

        self.update_camera_transforms()

    # -- transform bookkeeping ---------------------------------------------

    def update_camera_transforms(self):
        film = self.film()

        # Identical to the native plugin: reuse Mitsuba's own helper. Note the
        # first argument is the FULL film size, the second the crop size --
        # passing crop_size twice is a common and silent bug.
        self.m_camera_to_sample = mi.perspective_projection(
            film.size(),
            film.crop_size(),
            film.crop_offset(),
            self.m_x_fov,
            self.m_near_clip,
            self.m_far_clip,
        )

        # Shift the principal point in sample space. A positive
        # principal_point_offset_x moves the image content in +x sample space.
        ppo = self.m_principal_point_offset
        if ppo[0] != 0.0 or ppo[1] != 0.0:
            shift = mi.Transform4f().translate(
                [-ppo[0], -ppo[1], 0.0]
            )
            self.m_camera_to_sample = shift @ self.m_camera_to_sample

        self.m_sample_to_camera = self.m_camera_to_sample.inverse()

        # Resolution of the *crop window*; this is what film positions in
        # [0,1]^2 are measured against.
        res = film.crop_size()
        self.m_resolution = mi.ScalarVector2f(float(res[0]), float(res[1]))

        # Position differentials on the near plane, i.e. the camera-space
        # offset corresponding to a one-pixel step in x and y.
        origin = self.m_sample_to_camera @ mi.Point3f(0.0, 0.0, 0.0)
        self.m_dx = (
            self.m_sample_to_camera
            @ mi.Point3f(1.0 / self.m_resolution[0], 0.0, 0.0)
        ) - origin
        self.m_dy = (
            self.m_sample_to_camera
            @ mi.Point3f(0.0, 1.0 / self.m_resolution[1], 0.0)
        ) - origin

        # Precomputation for importance()/sample_direction(): the visible
        # rectangle of the z=1 plane, in camera space, and 1/its area.
        p_min = self.m_sample_to_camera @ mi.Point3f(0.0, 0.0, 0.0)
        p_max = self.m_sample_to_camera @ mi.Point3f(1.0, 1.0, 0.0)

        ax = float(p_min.x[0]) / float(p_min.z[0])
        ay = float(p_min.y[0]) / float(p_min.z[0])
        bx = float(p_max.x[0]) / float(p_max.z[0])
        by = float(p_max.y[0]) / float(p_max.z[0])

        self.m_rect_min = mi.ScalarPoint2f(min(ax, bx), min(ay, by))
        self.m_rect_max = mi.ScalarPoint2f(max(ax, bx), max(ay, by))

        area = (self.m_rect_max[0] - self.m_rect_min[0]) * (
            self.m_rect_max[1] - self.m_rect_min[1]
        )
        self.m_normalization = 1.0 / area

    # -- ray generation ----------------------------------------------------

    def sample_ray(self, time, wavelength_sample, position_sample,
                   aperture_sample, active=True):
        wavelengths, wav_weight = self.sample_wavelengths(
            dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active
        )

        ray = mi.Ray3f()
        ray.time = time
        ray.wavelengths = wavelengths

        # Sample position on the near plane, in local camera space.
        near_p = self.m_sample_to_camera @ mi.Point3f(
            position_sample.x, position_sample.y, 0.0
        )

        d = dr.normalize(mi.Vector3f(near_p))

        # Push the origin out to the near plane and shorten maxt accordingly,
        # so that the [near_clip, far_clip] range is respected.
        inv_z = dr.rcp(d.z)
        near_t = self.m_near_clip * inv_z
        far_t = self.m_far_clip * inv_z

        ray.o = self.m_to_world @ mi.Point3f(0.0, 0.0, 0.0)
        ray.d = self.m_to_world @ d
        ray.o += ray.d * near_t
        ray.maxt = far_t - near_t

        return ray, wav_weight

    def sample_ray_differential(self, time, wavelength_sample, position_sample,
                                aperture_sample, active=True):
        wavelengths, wav_weight = self.sample_wavelengths(
            dr.zeros(mi.SurfaceInteraction3f), wavelength_sample, active
        )

        ray = mi.RayDifferential3f()
        ray.time = time
        ray.wavelengths = wavelengths

        near_p = self.m_sample_to_camera @ mi.Point3f(
            position_sample.x, position_sample.y, 0.0
        )

        d = dr.normalize(mi.Vector3f(near_p))

        inv_z = dr.rcp(d.z)
        near_t = self.m_near_clip * inv_z
        far_t = self.m_far_clip * inv_z

        ray.o = self.m_to_world @ mi.Point3f(0.0, 0.0, 0.0)
        ray.d = self.m_to_world @ d
        ray.o += ray.d * near_t
        ray.maxt = far_t - near_t

        # Neighbouring pixels share the pinhole origin; only directions differ.
        ray.o_x = ray.o
        ray.o_y = ray.o
        ray.d_x = self.m_to_world @ dr.normalize(mi.Vector3f(near_p) + self.m_dx)
        ray.d_y = self.m_to_world @ dr.normalize(mi.Vector3f(near_p) + self.m_dy)
        ray.has_differentials = True

        return ray, wav_weight

    # -- direction sampling (needed by particle tracing / ptracer) ---------

    def importance(self, d):
        """Solid-angle density of the film-uniform sampling strategy.

        Derivation: a hypothetical image plane at unit distance from the
        pinhole has visible area A' (accounting for fov, aspect and the crop
        window). Uniform sampling in screen space has area density 1/A' there.
        Converting to a solid angle density gives
            dOmega = 1/A' * dist(P, o)^2 / cos(theta)
        and because P lies on the z=1 plane, dist(P, o)^2 = 1/cos^2(theta),
        hence dOmega = 1/A' * 1/cos^3(theta).
        """
        ct = d.z
        inv_ct = dr.rcp(ct)

        # Project onto the z=1 plane.
        p = mi.Point2f(d.x * inv_ct, d.y * inv_ct)

        valid = (ct > 0.0)
        valid &= (p.x >= self.m_rect_min[0]) & (p.x <= self.m_rect_max[0])
        valid &= (p.y >= self.m_rect_min[1]) & (p.y <= self.m_rect_max[1])

        return dr.select(
            valid, self.m_normalization * inv_ct * inv_ct * inv_ct, 0.0
        )

    def sample_direction(self, it, sample, active=True):
        trafo = self.m_to_world

        # Reference point in local camera space.
        ref_p = trafo.inverse() @ it.p

        ds = dr.zeros(mi.DirectionSample3f)

        active = mi.Mask(active)
        active &= (ref_p.z >= self.m_near_clip) & (ref_p.z <= self.m_far_clip)

        screen_sample = self.m_camera_to_sample @ ref_p
        ds.uv = mi.Point2f(screen_sample.x, screen_sample.y)
        active &= (ds.uv.x >= 0.0) & (ds.uv.x <= 1.0)
        active &= (ds.uv.y >= 0.0) & (ds.uv.y <= 1.0)
        ds.uv = ds.uv * self.m_resolution

        local_d = mi.Vector3f(ref_p)
        dist = dr.norm(local_d)
        inv_dist = dr.rcp(dist)
        local_d = local_d * inv_dist

        ds.p = trafo @ mi.Point3f(0.0, 0.0, 0.0)
        ds.d = (ds.p - it.p) * inv_dist
        ds.dist = dist
        ds.n = trafo @ mi.Vector3f(0.0, 0.0, 1.0)
        ds.pdf = dr.select(active, mi.Float(1.0), mi.Float(0.0))

        weight = mi.Spectrum(self.importance(local_d) * inv_dist * inv_dist)
        weight = dr.select(active, weight, mi.Spectrum(0.0))

        return ds, weight

    def pdf_direction(self, it, ds, active=True):
        # Delta-distributed in direction: sample_direction always returns a
        # single valid direction with unit discrete probability.
        return ds.pdf


    # ------------------------------------------------------------------
    # Extensions: forward projection (world/scene space -> sensor space)
    # ------------------------------------------------------------------

    def project_point(self, p, normalized=False, active=True):
        """Project a world-space 3D point onto the sensor.

        This is the inverse of :meth:`sample_ray`: feeding the returned
        (normalized) position back into ``sample_ray`` yields a ray through
        the original point.

        Not part of the native ``perspective`` plugin -- this is a
        convenience addition. It reuses exactly the same
        ``m_camera_to_sample`` transform that ray generation and
        :meth:`sample_direction` use, so it is consistent with them by
        construction, including fov, aspect ratio, crop window and the
        principal point offset.

        Parameters
        ----------
        p : mi.Point3f
            Point in world (scene) space. May be a batched Dr.Jit array.
        normalized : bool
            If ``True``, return sample-space coordinates in ``[0, 1]^2``.
            If ``False`` (default), return **pixel** coordinates measured
            relative to the crop window origin, i.e. in
            ``[0, crop_size.x] x [0, crop_size.y]``. Add
            ``film.crop_offset()`` to obtain absolute film pixels.
        active : mi.Mask
            Execution mask.

        Returns
        -------
        (mi.Point2f, mi.Mask)
            The projected position, and a mask that is ``True`` only where
            the point lies in front of the camera, within
            ``[near_clip, far_clip]``, and inside the image rectangle.
            Entries where the mask is ``False`` are forced to zero rather
            than left as inf/NaN, so a point behind the pinhole cannot
            poison downstream arithmetic -- always consult the mask, since
            an invalid result is indistinguishable from a genuine hit at the
            upper-left corner.

        Notes
        -----
        The y axis points *down* in the returned coordinates (row 0 is the
        top image row), matching Mitsuba's film convention.

        Camera-space depth, if you need it for a z-buffer or visibility
        test, is ``(self.m_to_world.inverse() @ mi.Point3f(p)).z``.
        """
        active = mi.Mask(active)

        # World -> local camera space.
        p_local = self.m_to_world.inverse() @ mi.Point3f(p)

        # Reject points behind the pinhole or outside the clip range. This
        # must happen before relying on the projective divide by z.
        valid = active & (p_local.z >= self.m_near_clip) \
                       & (p_local.z <= self.m_far_clip)

        # Camera space -> sample space ([0,1]^2 over the crop window).
        screen = self.m_camera_to_sample @ p_local
        uv = mi.Point2f(screen.x, screen.y)

        valid &= (uv.x >= 0.0) & (uv.x <= 1.0)
        valid &= (uv.y >= 0.0) & (uv.y <= 1.0)

        if not normalized:
            uv = uv * self.m_resolution

        uv = mi.Point2f(dr.select(valid, uv.x, 0.0),
                        dr.select(valid, uv.y, 0.0))

        return uv, valid

    def eval(self, si, active=True):
        # A pinhole has zero measure, so the radiance function is identically
        # zero -- all contribution flows through sample_ray / sample_direction.
        return mi.Spectrum(0.0)

    def eval_direction(self, it, ds, active=True):
        return mi.Spectrum(0.0)

    # -- misc --------------------------------------------------------------

    def bbox(self):
        p = self.m_to_world @ mi.Point3f(0.0, 0.0, 0.0)
        p = mi.ScalarPoint3f(float(p.x[0]), float(p.y[0]), float(p.z[0]))
        return mi.ScalarBoundingBox3f(p, p)

    def near_clip(self):
        return self.m_near_clip

    def far_clip(self):
        return self.m_far_clip

    def focus_distance(self):
        return self.m_focus_distance

    def x_fov(self):
        return self.m_x_fov

    def traverse(self, callback):
        try:
            callback.put_parameter("x_fov", self.m_x_fov,
                                   mi.ParamFlags.Differentiable)
            callback.put_parameter("to_world", self.m_to_world,
                                   mi.ParamFlags.NonDifferentiable)
            callback.put_parameter("near_clip", self.m_near_clip,
                                   mi.ParamFlags.NonDifferentiable)
            callback.put_parameter("far_clip", self.m_far_clip,
                                   mi.ParamFlags.NonDifferentiable)
        except TypeError:
            # Older binding without the flags argument.
            callback.put_parameter("x_fov", self.m_x_fov)
            callback.put_parameter("to_world", self.m_to_world)

    def parameters_changed(self, keys=None):
        keys = keys or []
        touched = (
            not keys
            or any(k in keys for k in
                   ("x_fov", "to_world", "near_clip", "far_clip",
                    "principal_point_offset_x", "principal_point_offset_y"))
        )
        if touched:
            if self.m_to_world.has_scale():
                raise RuntimeError(
                    "Scale factors in the camera-to-world transformation are "
                    "not allowed!"
                )
            self.update_camera_transforms()

    def to_string(self):
        return (
            f"PerspectiveCamera[\n"
            f"  x_fov = {self.m_x_fov},\n"
            f"  near_clip = {self.m_near_clip},\n"
            f"  far_clip = {self.m_far_clip},\n"
            f"  film = {self.film()},\n"
            f"  sampler = {self.sampler()},\n"
            f"  resolution = {self.m_resolution},\n"
            f"  principal_point_offset = {self.m_principal_point_offset},\n"
            f"  to_world = {self.m_to_world}\n"
            f"]"
        )


mi.register_sensor("perspective_py", lambda props: PerspectiveCamera(props))
