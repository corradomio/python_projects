from random import uniform

from matplotlib import patches
from matplotlib import pyplot as plt

COORDS = tuple[float, float]
POINT2D = tuple[float, float]

HANGAR_SIZE = (80, 12)
CAMERAS_LAYOUT = (8, 2)
CAMERA_SIZE = (11.2, 6.3)
INITIAL_CHICKENS: int = 20000


class ChickenType:
    def __init__(self, chicken_types: dict):
        assert isinstance(chicken_types, dict)

        self.types: dict[str, float] = {}
        for k, v in chicken_types.items():
            self.types[k] = v

        total = 0
        for k in self.types:
            total += self.types[k]
        for k in self.types:
            self.types[k] /= total

    def sample(self) -> str:
        r: float = uniform(0, 1)
        c: float = 0.
        k: str = ""
        for k in self.types:
            c += self.types[k]
            if r < c:
                return k
        return k

CHICKEN_TYPES = ChickenType({
    # "chicken": 1.0,
    "young-chicken": 0.25,
    "adult-chicken": 0.75
})


class Chicken:
    # # cid, zrot, xpos, ypos, sx, sy
    def __init__(self, id: str, type: str, center: POINT2D, rot: float, body_factor: POINT2D):
        assert isinstance(id, str)
        assert isinstance(type, str)

        self.id = id
        self.type = type
        self.center: COORDS = center
        self.body_factor: POINT2D = body_factor
        self.rot:float = rot
    # end
# end


class Camera:
    def __init__(self, id, center: COORDS, size: COORDS):
        self.id = id
        self.center: COORDS = center
        self.size: POINT2D = size

        cx, cy = center
        w, h = size

        self.left = cx - w/2
        self.top = cy - h/2
        self.right = cx + w/2
        self.bottom = cy + h/2
        self.chickens: list[Chicken] = []

    def add_chicken(self, chicken: Chicken):
        assert isinstance(chicken, Chicken)
        cx, cy = chicken.center
        if self.left <= cx <= self.right and self.top <= cy <= self.bottom:
            self.chickens.append(chicken)
    # end
# end


class Hangar:
    def __init__(self, hangar_size: POINT2D, camera_layout: POINT2D, camera_size: POINT2D, chicken_type: ChickenType):
        self.hangar_size: POINT2D = hangar_size
        self.camera_layout: POINT2D = camera_layout
        self.camera_size: POINT2D = camera_size
        self.cameras: dict[COORDS, Camera] = {}
        self.chicken_type = chicken_type

        self.chickens: list[Chicken] = []
    # end

    def compose_cameras(self):
        w, h = self.hangar_size
        nw, nh = self.camera_layout
        cw, ch = self.camera_size
        dw = w/nw
        dh = h/nh

        for i in range(nw):
            for j in range(nh):
                cx = (i+0.5)*dw
                cy = (j+0.5)*dh
                self.cameras[(i,j)] = Camera(id=(i,j), center=(cx, cy), size=(cw, ch))
            # end
        # end
    # end

    def generate_chickens(self, n_chickens: int):
        w, h = self.hangar_size

        for i in range(n_chickens):
            cid = f"c@{i+1}"
            ctype = self.chicken_type.sample()
            cx = uniform(0, w)
            cy = uniform(0, h)
            rot = uniform(0, 360)
            sx = uniform(0.9, 1.1)
            sy = uniform(0.9, 1.1)

            chicken = Chicken(id=cid, type=ctype, center=(cx, cy), rot=rot, body_factor=(sx, sy))
            self.chickens.append(chicken)
        # end
    # end

    def assign_chickens_to_cameras(self):
        for camera in self.cameras.values():
            for chicken in self.chickens:
                camera.add_chicken(chicken)
            pass

    def statistics(self):
        print(f"n chickens: {len(self.chickens)}")
        print(f"n cameras: {len(self.cameras)}")
        n_chickens = 0
        for cij in self.cameras:
            camera = self.cameras[cij]
            c_chickens = len(camera.chickens)
            print(f"... {cij}: {c_chickens}")
            n_chickens += c_chickens
        print(f"camera chickens: {n_chickens}")

    def plot_cameras(self):

        plt.figure(figsize=(10,2))
        ax = plt.gca()
        # fig, ax = plt.subplots()

        ax.set_aspect("equal")

        w, h = self.hangar_size
        ax.add_patch(
            patches.Rectangle((0,0), w, h, linewidth=.0)
        )

        chicks_x = [chicken.center[0] for chicken in self.chickens]
        chicks_y = [chicken.center[1] for chicken in self.chickens]
        plt.scatter(chicks_x, chicks_y, c='white', s=0.005, marker='*')

        for camera in self.cameras.values():
            cx, cy = camera.center
            cw, ch = camera.size
            ax.add_patch(
                patches.Rectangle(
                    (cx-cw/2, cy-ch/2), cw, ch,
                    edgecolor='r',
                    linewidth=0.5,
                    fill=False
                ),
            )


        ax.set_xlim(-1, w+1)
        ax.set_ylim(-1, h+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()

        plt.savefig("hangar.png", dpi=300)
# end


def gen_hangar(hangar_size: COORDS, cameras_layout: COORDS, camera_size: COORDS, chicken_type: ChickenType):
    hangar = Hangar(hangar_size, cameras_layout, camera_size, chicken_type)

    hangar.compose_cameras()
    hangar.generate_chickens(INITIAL_CHICKENS)
    hangar.assign_chickens_to_cameras()
    return hangar


def main():
    hangar = gen_hangar(HANGAR_SIZE, CAMERAS_LAYOUT, CAMERA_SIZE, CHICKEN_TYPES)
    hangar.statistics()
    hangar.plot_cameras()


if __name__ == "__main__":
    main()
