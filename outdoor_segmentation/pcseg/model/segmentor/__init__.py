# raw point
# ...

# range view
from .range.rangenet.model.semantic.rangenet import RangeNet
from .range.salsanext.model.semantic.salsanext import SalsaNext
from .range.fidnet.model.semantic.fidnet import FIDNet
from .range.cenet.model.semantic.cenet import CENet

# bird's eye view
# ...

# voxel
from .voxel.cylinder3d import Cylinder_TS
from .voxel.cylinder3d.cylinder_ts import Cylinder_TS
from .voxel.epcl.epcl_outdoor_seg import EPCLOutdoorSeg
#from .voxel.epcl_hd.epcl_outdoor_seg_exp1 import EPCLOutdoorSegHD

# multi-view fusion
from .fusion.spvcnn.spvcnn import SPVCNN #, MinkUNet
from .fusion.rpvnet.rpvnet import RPVNet



__all__ = {
    # raw point
    # ...

    # range view
    'RangeNet++': RangeNet,
    'SalsaNext': SalsaNext,
    'FIDNet': FIDNet,
    'CENet': CENet,

    # bird's eye view
    # ...

    # voxel
    'Cylinder_TS': Cylinder_TS,
    'EPCLOutdoorSeg': EPCLOutdoorSeg,
    #'EPCLOutdoorSegHD': EPCLOutdoorSegHD,

    # multi-view fusion
    'SPVCNN': SPVCNN,
    'RPVNet': RPVNet,
}


def build_segmentor(model_cfgs, num_class, exp=1, crop=False, lr = 0.01):
    print("EXP", exp)
    print("model", model_cfgs.NAME)
    print("ls", lr)
    if exp==1:
        from .voxel.epcl_hd.epcl_outdoor_seg_exp1 import EPCLOutdoorSegHD
    elif exp==2:
        from .voxel.epcl_hd.epcl_outdoor_seg_exp2 import EPCLOutdoorSegHD
    elif exp==3:
        from .voxel.epcl_hd.epcl_outdoor_seg_exp3 import EPCLOutdoorSegHD
    elif exp==4:
        from .voxel.epcl_hd.epcl_outdoor_seg_exp4 import EPCLOutdoorSegHD 
    elif exp==5:
        if crop:
            from .voxel.epcl_hd.epcl_outdoor_seg_exp5_crop import EPCLOutdoorSegHD
        else:
            from .voxel.epcl_hd.epcl_outdoor_seg_exp5 import EPCLOutdoorSegHD
    elif exp==6:
        from .voxel.epcl_hd.epcl_outdoor_seg_exp6 import EPCLOutdoorSegHD  

    model = eval(model_cfgs.NAME)( #model_cfgs.NAME
        model_cfgs=model_cfgs,
        num_class=num_class,
        lr = lr
    )

    return model
