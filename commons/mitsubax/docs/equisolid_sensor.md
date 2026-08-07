Now let's actually run and verify it, including checking it behaves distinctly from the other three:Confirmed working — renders correctly at FOV=220° (beyond hemisphere) on the JIT backend, with the expected circular fisheye frame and no NaNs.That completes the full classic fisheye set:

| File | Type string | Formula | FOV range | Notes |
|---|---|---|---|---|
| `equidistant_fisheye_sensor.py` | `equidistant_fisheye` | `r = f·θ` | up to 360° | linear angle-to-radius |
| `stereographic_sensor.py` | `stereographic` | `r = 2f·tan(θ/2)` | up to 360° | conformal, preserves local shape |
| `orthographic_fisheye_sensor.py` | `orthographic_fisheye` | `r = f·sin(θ)` | up to 180° only | strongest center magnification |
| `equisolid_angle_sensor.py` | `equisolid_angle` | `r = 2f·sin(θ/2)` | up to 360° | equal-area, what most real fisheye lenses use |

This one (`equisolid_angle`) is the one you'd reach for if the goal is downstream lighting/environment-map work — because it's equal-area, uniformly sampling sensor pixels gives you (nearly) uniform solid-angle sampling of the scene, which is what you want when e.g. building an HDRI or importance-sampling incident light from the render. All four share the same `mi.Sensor` interface and constructor arguments (`fov`, `near_clip`, `far_clip`, `to_world`), so you can swap between them freely in a scene dict just by changing `'type'`.