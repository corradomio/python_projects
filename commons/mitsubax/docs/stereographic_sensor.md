Now let me actually verify it renders correctly rather than just trust the math.Confirmed working — rendered successfully on both `scalar_rgb` and the JIT `llvm_ad_rgb` backend, at FOV 170° and 220°, with no NaNs and a correctly circular fisheye frame.Key differences from the equidistant version I gave you earlier:

- **Projection**: `r = 2f·tan(θ/2)` instead of `r = f·θ`. Inverting it for ray generation gives `θ = 2·atan(r · tan(θ_max/2))`.
- **Conformal**: stereographic projection preserves local angles/shapes, so objects near the edge of frame don't get the radial stretching you see with equidistant fisheye — this is the same projection behind "little planet" panoramas and many real circular-fisheye lenses (as opposed to the more common equisolid-angle lenses, which are a different mapping again).
- Same `mi.Sensor` interface (`sample_ray`, `sample_ray_differential`), same aspect-ratio handling, same clipping-plane behavior, so it's a drop-in swap for the `wideangle` type — just register it under a different name (`stereographic`) and use `'type': 'stereographic'` in your scene dict.

Everything else (near/far clipping, ray-differential finite differencing for texture filtering, invalid-ray masking outside the image circle) is identical in structure to the previous file.
