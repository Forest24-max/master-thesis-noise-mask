import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.misc import imread
from mpl_toolkits.mplot3d import axes3d, Axes3D

R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])

im = None
ax = None


def generate_gif(filename, video_array):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    if len(video_array.shape) > 3:
        im = ax.imshow(video_array[0], animated=True)
    else:
        im = ax.imshow(video_array[0], cmap='gray', animated=True, vmin=0, vmax=255)

    def img_func(i):
        im.set_array(video_array[i])
        ax.set_xlabel('frame=%s' % i)
        return im, ax

    build_gif(fig, img_func, video_array.shape[0], filename, show_gif=False)


def generate_3d_gif(filename, video_array):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    start = video_array[0]
    xx, yy = np.mgrid[:start.shape[0], :start.shape[1]]

    ax.set_zlim(0, 450)
    ax.view_init(50, 0)
    ax.plot_surface(xx, yy, start, rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0, alpha=0.6)
    ax.contour(xx, yy, start, zdir='z', offset=1, cmap=plt.cm.coolwarm)
    def img_func(i):
        ax.clear()
        #plot_surface
        ax.view_init(50, i % 360)
        im = ax.plot_surface(xx, yy, video_array[i], rstride=1, cstride=1, cmap=plt.cm.gray, linewidth=0, alpha=0.6)
        ax.contour(xx, yy, video_array[i], zdir='z', offset=1, cmap=plt.cm.coolwarm)
        ax.set_xlabel('frame=%s' % i)
        # ax.set_zlim(0, 255)
        return im

    build_gif(fig, img_func, video_array.shape[0], filename, interval=100, show_gif=False)



def build_gif(fig, fetch_img_func, length, filename, interval=150, show_gif=True):

    im_ani = animation.FuncAnimation(fig, fetch_img_func, frames=range(0, length), interval=interval, repeat_delay=20 * interval)

    im_ani.save(filename, writer='imagemagick')

    if show_gif:
        plt.show()

if __name__ == '__main__':
    a = np.random.randint(0, 255, (25, 3, 3))
    generate_3d_gif('test3d.gif', a)