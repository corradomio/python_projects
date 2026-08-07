That's actually the same projection I gave you in my very first response — the `wideangle_sensor.py` file already implements the equidistant (f-θ) fisheye, where `r = f·θ` grows linearly with angle from the optical axis. To keep this consistent with the naming pattern of the other two (`stereographic_sensor.py`, `orthographic_fisheye_sensor.py`), let me rename it accordingly and re-verify it still runs correctly.Now re-verify the renamed file actually still runs (didn't want to just trust a sed rename):Good — identical mean to the very first render I did (0.5455345), confirming the rename didn't change behavior. Let me view it once more and hand it over.This is the same equidistant (f-θ) projection from my first response, just renamed (`EquidistantFisheyeCamera`, registered as `'equidistant_fisheye'`) to match the naming convention of the `stereographic` and `orthographic_fisheye` files so all three can live side by side without collisions. Re-verified end-to-end after the rename — identical render output to the original (same pixel mean, same visual result).

You now have all three classic fisheye mappings as consistent, drop-in `mi.Sensor` plugins:

| File | Type string | Formula | FOV range |
|---|---|---|---|
| `equidistant_fisheye_sensor.py` | `equidistant_fisheye` | `r = f·θ` | up to 360° |
| `stereographic_sensor.py` | `stereographic` | `r = 2f·tan(θ/2)` | up to 360° |
| `orthographic_fisheye_sensor.py` | `orthographic_fisheye` | `r = f·sin(θ)` | up to 180° only |

If you want, I could also add the fourth classic mapping (equisolid-angle, `r = 2f·sin(θ/2)` — the most common one on real consumer fisheye lenses) to round out the set.