USE_DEM = False
USE_LGB_LLS = False
MODE = "baseline"  # default

def apply_config(mode):
    global USE_DEM, USE_LGB_LLS, MODE
    MODE = mode
    if mode == "baseline":
        USE_DEM = False
        USE_LGB_LLS = False
    elif mode == "dem":
        USE_DEM = True
        USE_LGB_LLS = False
    elif mode == "full":
        USE_DEM = True
        USE_LGB_LLS = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
