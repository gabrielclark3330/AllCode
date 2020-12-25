from graphics import *
import math
import numpy as np
#equation of circle
# r^2 = (x-h)^2 + (y-k)**2
# y = math.sqrt(1000 - (x-h)**2)+k
# The below code attempts to render a circle point by point.
def main():
    win = GraphWin("C0ol Renderings Here", 750, 750)
    win.setBackground(color_rgb(0,0,0))
    x = 0
    y = 0
    screen = np.zeros((750,750))
    while x <=500:
        while y <=500:
            x = x+1
            h = 250
            k = 250
            r = 300
            y = y+1
            screen[y][x] = 1
    for y in range(len(screen)):
        for x in range(len(screen[x])):
            if screen[y][x] == 1:
                pt = Point(x, y)
                pt.setOutline(color_rgb(250, 0, 250))
                pt.draw(win)
    win.getMouse()
    win.close

main()
