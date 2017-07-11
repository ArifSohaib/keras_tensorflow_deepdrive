import tensorflow as tf
import numpy as np
from model import model

# Read subset of data
all_data = np.load('simple_data.npz')
imgs_color = all_data['imgs']
speedx = np.concatenate((all_data['spds'], all_data['accel']))
speedx = speedx.reshape((-1,2))
steer = all_data['steer']

nb_epoch = 100
mini_epoch = 10
num_steps = int(nb_epoch/mini_epoch)
for step in range(num_steps):
    h = model.fit([speedx, imgs_color], {'steer_out':steer},
                    batch_size = 32, nb_epoch=mini_epoch, verbose=1,
                    validation_split=0.1, shuffle=True)
    model.save_weights('steer_comma_{0}_{1:4.5}.h5'.format(step,h.history['val_loss'][-1]),overwrite=True)
