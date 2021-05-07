import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# flood fill algorithm https://gist.github.com/JDWarner/1158a9515c7f1b1c21f1
def floodfill ( data, seed_coords, fill_value ):
    xsize, ysize = data.shape
    orig_value = data[ seed_coords[0], seed_coords[1] ]

    stack = set(((seed_coords[0], seed_coords[1]),))
    if fill_value == orig_value:
        raise ValueError("Filling region with same value "
                         "already present is unsupported. "
                         "Did you already fill this region?")

    while stack:
        x, y = stack.pop()

        if data [x, y] == orig_value:
            data[x, y] = fill_value
            if x > 0:
                stack.add((x - 1, y))
            if x < (xsize - 1):
                stack.add((x + 1, y))
            if y > 0:
                stack.add((x, y - 1))
            if y < (ysize - 1):
                stack.add((x, y + 1))


def circle(indices, xm, ym, r):

    x = r - 1
    y = 0
    dx = 1
    dy = 1
    err = dx - (r * 2)


    while ( x >= y):
        indices [ xm + x , ym + y] = True
        indices [ xm + y , ym + x] = True
        indices [ xm - y , ym + x] = True
        indices [ xm - x , ym + y] = True
        indices [ xm - x , ym - y] = True
        indices [ xm - y , ym - x] = True
        indices [ xm + y , ym - x] = True
        indices [ xm + x , ym - y] = True

        if err <= 0:
            y+=1
            err += dy
            dy += 2

        if err > 0:
            x-=1
            dy += 2
            err += dx - ( 2*r )


NX = 512 
NY = 512

# setup indices
ind = np.full(( NX, NY), False, dtype=bool)

circle( ind, NX//2, NY//2, 128 )

floodfill( ind, [ NX//2, NY//2] , True)

fig, ax = plt.subplots(1)
ax.pcolor( ind )

plt.show()
