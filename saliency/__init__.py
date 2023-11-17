from .color import *
from .depth import *
from .object import *
from .angle import *
from .basic import *
# from .deepg import *


class SaliencyMap:
    def __init__(self, smap_type):
        self.smap_type = smap_type

    def normalize(self, smap):
        smap = smap.astype(np.float32)
        smap -= smap.min()
        smap /= smap.max()
        return smap

    def get_smap(self, trial):
        if self.smap_type == "intensity":
            smap = smap_intensity(trial)
        elif self.smap_type == "rough":
            smap = smap_rough(trial)
        elif self.smap_type == "fine":
            smap = smap_fine(trial)
        elif self.smap_type == "color":
            smap = smap_color(trial)
        elif self.smap_type == "rg":
            smap = smap_rg_opponency(trial)
        elif self.smap_type == "by":
            smap = smap_by_opponency(trial)
        elif self.smap_type == "orientation":
            smap = smap_orientation(trial)
        elif isinstance(self.smap_type, int):
            smap = smap_angle(trial, self.smap_type)
        elif self.smap_type == "depth":
            smap = smap_depth(trial)
        elif self.smap_type == "deepgaze":
            smap = smap_deepgaze(trial)
        elif self.smap_type == "center":
            smap = smap_center(trial)
        else:
            smap = smap_object(trial, self.smap_type)

        return self.normalize(smap) * 255
