import numpy as np
from model import model
import matplotlib.pyplot as plt

# Read subset of data
all_data = np.load('simple_data.npz')
imgs_color = all_data['imgs']
speedx = np.concatenate((all_data['spds'], all_data['accel']))
speedx = speedx.reshape((-1,2))
steer = all_data['steer']

#make predictions
start = 45000
stop = 65000
model.load_weights('steer_comma_0_0.00057615.h5')
preds = model.predict([speedx[start:stop],imgs_color[start:stop]])
steer_preds = preds.reshape([-1])


# Video of prediction
import matplotlib.animation as animation
from PIL import Image, ImageDraw
figure = plt.figure()
imageplot = plt.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
val_idx = start
def get_point(s,start=0,end=63,height= 16):
    X = int(s*(end-start))
    if X < start:
        X = start
    if X > end:
        X = end
    return (X,height)
def next_frame(i):
    im = Image.fromarray(np.array(imgs_color[val_idx+i],dtype=np.uint8))
    p = get_point(steer_preds[i])
    t = get_point(steer[i+val_idx])
    draw = ImageDraw.Draw(im)
    draw.line((32,63, p,p),
                fill=(255,0,0,128))
    draw.line((32,63, t,t),
                fill=(0,255,0,128))
    imageplot.set_array(im)
    return imageplot,
animate = animation.FuncAnimation(figure, next_frame, frames=range(0,stop-start), interval=25, blit=False)
plt.show()
