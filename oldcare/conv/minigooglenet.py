from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, Dense, \
    BatchNormalization, concatenate, AveragePooling2D, Activation
from keras import Input, Model


class MiniGOOGLENet:
    @staticmethod
    def build(width, height, depth, classes):
        def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
            # define a CONV => BN => RELU pattern
            x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
            x = BatchNormalization(axis=chanDim)(x)
            x = Activation("relu")(x)

            # return the block
            return x

        def inception_module(x, numK1x1, numK3x3, chanDim):
            # define two CONV modules, then concatenate across the
            # channel dimension
            conv_1x1 = conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
            conv_3x3 = conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
            x = concatenate([conv_1x1, conv_3x3], axis=chanDim)

            # return the block
            return x

        def downsample_module(x, K, chanDim):
            # define the CONV module and POOL, then concatenate
            # across the channel dimensions
            conv_3x3 = conv_module(x, K, 3, 3, (2, 2), chanDim,
                                   padding="valid")
            pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
            x = concatenate([conv_3x3, pool], axis=chanDim)

            # return the block
            return x

        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # define the model input and first CONV module
        inputs = Input(shape=inputShape)
        x = conv_module(inputs, 96, 3, 3, (1, 1), chanDim)

        # two Inception modules followed by a downsample module
        x = inception_module(x, 32, 32, chanDim)
        x = inception_module(x, 32, 48, chanDim)
        x = downsample_module(x, 80, chanDim)

        # four Inception modules followed by a downsample module
        x = inception_module(x, 112, 48, chanDim)
        x = inception_module(x, 96, 64, chanDim)
        x = inception_module(x, 80, 80, chanDim)
        x = inception_module(x, 48, 96, chanDim)
        x = downsample_module(x, 96, chanDim)

        # two Inception modules followed by global POOL and dropout
        x = inception_module(x, 176, 160, chanDim)
        x = inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(x)
        x = Dropout(0.5)(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        # create the model
        model = Model(inputs, x, name="minigooglenet")

        # return the constructed network architecture
        return model
