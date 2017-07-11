import keras
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Embedding, Input, merge, ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2, l1
# from keras.utils.np_utils import to_categirical
from keras import backend as K
# import sklearn.metrics as metrics

#Define model
nrows = 64
ncols = 64
wr = 0.00001
dp = 0.

#speed, accel, distance, angle
real_in = Input(shape=(2,),name='real_input')

#frame_in
frame_in = Input(shape=(nrows,ncols,3), name='img_input') #in theano backends it is channel X row X column, in tenrosflow it is row X column X channel

# convolution for image input
conv1 = Convolution2D(6,3,3,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
conv_l1 = conv1(frame_in)
Econv_l1 = ELU()(conv_l1)
pool_l1 = MaxPooling2D(pool_size=(2,2))(Econv_l1)

conv2 = Convolution2D(8,3,3,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
conv_l2 = conv2(pool_l1)
Econv_l2 = ELU()(conv_l2)
pool_l2 = MaxPooling2D(pool_size=(2,2))(Econv_l2)
drop_l2 = Dropout(dp)(pool_l2)

conv3 = Convolution2D(16,3,3,border_mode='same',W_regularizer=l1(wr),init='lecun_uniform')
conv_l3 = conv3(drop_l2)
Econv_l3 = ELU()(conv_l3)
pool_l3 = MaxPooling2D(pool_size=(2,2))(Econv_l3)
drop_l3 = Dropout(dp)(pool_l3)

flat = Flatten()(drop_l3)

M = merge([flat, real_in], mode='concat', concat_axis=1)

D1 = Dense(32, W_regularizer=l1(wr), init='lecun_uniform')(M)
ED1 = ELU()(D1)
DED1 = Dropout(dp)(ED1)

S1 = Dense(64, W_regularizer=l1(wr), init='lecun_uniform')(DED1)
ES1 = ELU()(S1)

# Break_out = Dense(1, activation='linear', name='break_out', init='lecun_uniform')(ES1)
Steer_out = Dense(1, activation='linear', name='steer_out', init='lecun_uniform')(ES1)

model = Model(input=[real_in,frame_in],output=[Steer_out])
adam = Adam(lr=0.001)
model.compile(loss=['mse'],optimizer=adam, metrics=['mse'])
