SKIP_TO_NTH = 3
SUBDIV_SIZE = 25
SUBDIV_RAD = 2

# Placement defs
MAX_VAL = 250
MIN_DIST =  5
SCALE_FACT = 10.0

# Enableable stuff
DO_IMAGE_LINEARIZATION = False
DARKEN_NOT_WHITE = True
DARKEN_EDGES = True
SMOOTH_KERNEL_RAD = 1
EDGE_KERNEL_RAD = 12
EDGE_DARKENING_RATIO = 1.0

STD_TRUNC_VAL = 13
STD_IGNORE_BLUE = False

BOXSIZE = '100mmSquare'
BOXSIZE = 'SmallTablet'
# BOXSIZE = 'MaxSize'

LASER_X_MIN = 52
LASER_Y_MIN = 43

LASER_HEIGHT = 0.0 # mm
LASER_POWER = 80.0 # %
LASER_SPEED = 80.0 # mm/s

if BOXSIZE == '100mmSquare':
    # MAX_DIMS = [1000, 1000]
    MAX_DIMS = [950, 950]
    MM_PER_PIX = 0.1 # 100mm square
    LASER_HEIGHT = -5.0 # mm


elif BOXSIZE == 'SmallTablet':
    MAX_DIMS = [800, 540]
    MM_PER_PIX = 0.1 # 100mm square
    LASER_HEIGHT = -3.0 # mm

elif BOXSIZE == 'MaxSize':
    LASER_X_MIN = 10
    LASER_Y_MIN = 10
    MAX_DIMS = [1000, 1000]
    MM_PER_PIX = 0.2 # 100mm square


# # Fiber laser params
# LASER_HEIGHT = 0.0 # mm
# LASER_POWER = 100.0 # %
# LASER_SPEED = 600.0 # mm/s

# Line placement defs
MAX_LINE_LEN = 30
CIRCLE_RAD = 0.8

CIRCLE_RAD /= MM_PER_PIX

# Linearity Defs
SOBEL_PRE_SMOOTH_RAD = 4
SOBEL_KERNEL_RAD = 7
SOBEL_NET_SMOOTH_RAD = 10
SOBEL_DO_RGB = False
TAN_LINE_RAD = 6
SOBEL_MULT = 100.0

# Box Nest Defs
OUTLINE_OFFSET = 5 # Offset of outline of box in pix

# MAX_DIMS = [500, 400]
# MAX_DIMS = [100, 150]