from .worldmap import *

# Disabled ArUco ids: 17 and 37.  Don't use them.

def make_walls():

    # side +1 measures x from the left edge rightward
    # side -1 measures x from the right edge leftward

    w1 = WallSpec(length=300, height=190,
                  marker_specs = { 7 : {'side': +1, 'x':  25, 'y': 35},
                                   2 : {'side': +1, 'x':  75, 'y': 35},
                                   3 : {'side': +1, 'x': 222, 'y': 35},
                                   8 : {'side': +1, 'x': 270, 'y': 35},

                                   9 : {'side': -1, 'x': 270, 'y': 35},
                                   4 : {'side': -1, 'x': 222, 'y': 20},
                                   5 : {'side': -1, 'x':  75, 'y': 20},
                                  10 : {'side': -1, 'x':  25, 'y': 20}, },
                  doorways = { 'd1' : {'x': 150, 'width': 75, 'height': 115} }
                  )


make_walls()
