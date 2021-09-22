import tensorflow as tf


def batch_norm():
    return tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def relu():
    return tf.keras.layers.ReLU()


def conv1d(filters, kernel_size=3, strides=1):
    return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding='same', use_bias=False,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling())  # initial weights matrix


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):  # constructor
        super().__init__(**kwargs)
        self.num_channels = num_channels

    def build(self, input_shape):
        self.conv = conv1d(filters=self.num_channels)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.listLayers = [self.bn, self.relu, self.conv]
        super().build(input_shape)

    def call(self, x, **kwargs):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x, y], axis=-1)
        return y


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels, **kwargs):  # constructor
        super().__init__(**kwargs)
        self.num_convs = num_convs
        self.num_channels = num_channels

    def build(self, input_shape):
        self.listLayers = []
        for _ in range(self.num_convs):
            self.listLayers.append(ConvBlock(self.num_channels))
        super().build(input_shape)

    def call(self, x, **kwargs):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels

    def build(self, input_shape):
        self.bn = batch_norm()
        self.relu = relu()
        self.conv = conv1d(self.num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool1D(pool_size=2, strides=2)
        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)


class DenseNet(tf.keras.Model):
    def __init__(self, num_outputs=1, num_convs_in_dense_blocks=(4, 4, 4, 4),
                 first_num_channels=64, growth_rate=(32, 32, 32, 32),
                 block_fn1=DenseBlock, block_fn2=TransitionBlock,
                 include_top=True, **kwargs):  # constructor

        super().__init__(**kwargs)

        # Built Convolution layer
        self.conv1 = conv1d(filters=64, kernel_size=7, strides=2)  # 7×7, 64, stride 2
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.maxpool1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same')  # 3×3 max pool, stride 2

        # Built Dense Blocks and Transition layers
        self.blocks = []
        num_channels = first_num_channels
        for stage, _ in enumerate(num_convs_in_dense_blocks):  # stage = [0,1,2,3] and _=[4,4,4,4]
            dnet_block = block_fn1(num_convs_in_dense_blocks[stage], growth_rate[stage])
            self.blocks.append(dnet_block)

            # This is the number of output channels in the previous dense block
            num_channels += num_convs_in_dense_blocks[stage] * growth_rate[stage]

            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if stage != len(num_convs_in_dense_blocks) - 1:
                num_channels //= 2
                tran_block = block_fn2(num_channels)
                self.blocks.append(tran_block)

            # self.blocks.append(dnet_block)

        # include top layer (full connected layer)
        self.include_top = include_top
        if include_top:
            # average pool, 1-d fc, sigmoid
            self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
            out_act = 'sigmoid' if num_outputs == 1 else 'softmax'
            self.classifier = tf.keras.layers.Dense(num_outputs, out_act)

    def call(self, x, include_top=None, **kwargs):
        if include_top is None:
            include_top = self.include_top

        # Built conv1 layer
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.maxpool1(x)

        # Built other layers
        for dnet_block in self.blocks:
            x = dnet_block(x)

        # include top layer (full connected layer)
        if include_top:
            x = self.global_pool(x)
            x = self.classifier(x)
        return x


# # To built Resnet-18/34
# class ResidualBlock(tf.keras.layers.Layer):
#
#     # def __init__(self, **kwargs): packs all of the keyword arguments used in any given call to __init__ into a dict
#     # super().__init__(**kwargs) expands them into keyword arguments again.
#     # https://stackoverflow.com/questions/41929715/what-does-this-to-in-python-super-init-kwargs
#     def __init__(self, filters, kernel_size=3, strides=1, **kwargs):  # constructor
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#
#     def build(self, input_shape):
#         """ Overrides method 'build' in 'tf.keras.layers.Layer'
#
#          Args:
#            input_shape: Instance of `TensorShape`, or list of instances of
#                         `TensorShape` if the layer expects a list of inputs
#                         (one instance per input).
#         """
#         num_chan = input_shape[-1]  # dimension of input
#
#         # No. of channel (filter) between two other ResidualBlock can be different --> need self.strides
#         self.conv1 = conv1d(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides)
#         self.bn1 = batch_norm()
#         self.relu1 = relu()
#
#         # conv1 and conv2 have same no. channel --> default trides=1
#         self.conv2 = conv1d(filters=self.filters, kernel_size=self.kernel_size, strides=1)
#         self.bn2 = batch_norm()
#         self.relu2 = relu()
#
#         # For shortcut connection
#         # Dimensions of x (num_chan) and F (filters) must be equal
#         # The projection shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions) with stride of 2.
#         if num_chan != self.filters or self.strides > 1:
#             self.proj_conv = conv1d(filters=self.filters, kernel_size=1, strides=self.strides)
#             self.proj_bn = batch_norm()
#             self.projection = True
#         else:
#             self.projection = False
#         super().build(input_shape)
#
#     def call(self, x, **kwargs):
#         """ Overrides method 'call' in 'tf.keras.layers.Layer'
#
#             Args:
#                 x: Input tensor, or list/tuple of input tensors.
#                 **kwargs: Additional keyword arguments. Currently unused.
#             Returns:
#                 A tensor or list/tuple of tensors.
#         """
#         shortcut = x
#         if self.projection:
#             shortcut = self.proj_conv(shortcut)
#             shortcut = self.proj_bn(shortcut)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x + shortcut)
#         return x
#
#
# # To built Resnet-50
# class BottleneckBlock(tf.keras.layers.Layer):
#
#     # expansion = 4 to 256 = 64*4 (see Fig. 5 in Deep Residual Learning for Image Recognition)
#     def __init__(self, filters, kernel_size=3, strides=1, expansion=4, **kwargs):  # constructor
#         # def __init__(self, **kwargs): packs all of the keyword arguments used in any given call to __init__ into a dict
#         # super().__init__(**kwargs) expands them into keyword arguments again.
#         # https://stackoverflow.com/questions/41929715/what-does-this-to-in-python-super-init-kwargs
#         super().__init__(**kwargs)
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.strides = strides
#         self.expansion = expansion
#
#     def build(self, input_shape):
#         """ Overrides method 'build' in 'tf.keras.layers.Layer'
#
#             Args:
#                 input_shape: Instance of `TensorShape`, or list of instances of
#                             `TensorShape` if the layer expects a list of inputs
#                             (one instance per input).
#         """
#         num_chan = input_shape[-1]
#         self.conv1 = conv1d(filters=self.filters, kernel_size=1, strides=1)
#         self.bn1 = batch_norm()
#         self.relu1 = relu()
#         self.conv2 = conv1d(filters=self.filters, kernel_size=self.kernel_size, strides=self.strides)
#         self.bn2 = batch_norm()
#         self.relu2 = relu()
#         self.conv3 = conv1d(filters=self.filters * self.expansion, kernel_size=1, strides=1)
#         self.bn3 = batch_norm()
#         self.relu3 = relu()
#
#         # See equation (2) in Deep Residual Learning for Image Recognition
#         if num_chan != self.filters * self.expansion or self.strides > 1:
#             self.proj_conv = conv1d(filters=self.filters * self.expansion, kernel_size=1, strides=self.strides)
#             self.proj_bn = batch_norm()
#             self.projection = True
#         else:
#             self.projection = False
#         super().build(input_shape)
#
#     def call(self, x, **kwargs):
#         """ Overrides method 'call' in 'tf.keras.layers.Layer'
#
#             Args:
#                 x: Input tensor, or list/tuple of input tensors.
#                 **kwargs: Additional keyword arguments. Currently unused.
#             Returns:
#                 A tensor or list/tuple of tensors.
#         """
#         shortcut = x
#         if self.projection:
#             shortcut = self.proj_conv(shortcut)
#             shortcut = self.proj_bn(shortcut)
#
#         # See 50-layer ResNet in Deep Residual Learning for Image Recognition
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu3(x + shortcut)
#         return x
#
#
# # See Table 1 in Deep Residual Learning for Image Recognition
# class ResNet(tf.keras.Model):
#
#     def __init__(self, num_outputs=1, blocks=(2, 2, 2, 2), filters=(64, 128, 256, 512),
#                  kernel_size=(3, 3, 3, 3), block_fn=ResidualBlock, include_top=True, **kwargs):  # constructor
#         """
#             Args:
#                 num_outputs:    no. of output
#                 blocks:         no. of conv1d in each ResidualBlock (default: 4 bocks for Resnet-18)
#                 filters:        no. of channel of each ResidualBlock
#                 kernel_size:    size of kernel in conv of each ResidualBlock
#                 block_fn:       ResidualBlock function
#                 include_top:    insert top layer or not
#         """
#
#         # def __init__(self, **kwargs): packs all of the keyword arguments used in any given call to __init__ into a dict
#         # super().__init__(**kwargs) expands them into keyword arguments again.
#         # https://stackoverflow.com/questions/41929715/what-does-this-to-in-python-super-init-kwargs
#         super().__init__(**kwargs)
#
#         # Built conv1 layer
#         self.conv1 = conv1d(filters=64, kernel_size=7, strides=2)  # 7×7, 64, stride 2
#         self.bn1 = batch_norm()
#         self.relu1 = relu()
#         self.maxpool1 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same')  # 3×3 max pool, stride 2
#
#         # Built other layers
#         self.blocks = []
#         for stage, num_blocks in enumerate(blocks):  # stage = 0, 1, 2, 3 and num_blocks = 2
#             for block in range(num_blocks):  # block = 0, 1
#                 strides = 2 if block == 0 and stage > 0 else 1  # strides=2 when block=0 && stage=1,2,3
#                 # first conv of ResidualBlock 1,2,3 (stage=1,2,3) when
#                 # change no. of output channel
#                 res_block = block_fn(filters=filters[stage], kernel_size=kernel_size[stage], strides=strides)
#                 self.blocks.append(res_block)
#
#         # include top layer (full connected layer)
#         self.include_top = include_top
#         if include_top:
#             # average pool, 1-d fc, sigmoid
#             self.global_pool = tf.keras.layers.GlobalAveragePooling1D()
#             out_act = 'sigmoid' if num_outputs == 1 else 'softmax'
#             self.classifier = tf.keras.layers.Dense(num_outputs, out_act)
#
#     def call(self, x, include_top=None, **kwargs):
#         """ Overrides method 'call' in 'tf.keras.layers.Layer'
#
#             Args:
#                 x: Input tensor, or list/tuple of input tensors.
#                 **kwargs: Additional keyword arguments. Currently unused.
#             Returns:
#                 A tensor or list/tuple of tensors.
#         """
#         if include_top is None:
#             include_top = self.include_top
#
#         # Built conv1 layer
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#
#         # Built other layers
#         for res_block in self.blocks:
#             x = res_block(x)
#
#         # include top layer (full connected layer)
#         if include_top:
#             x = self.global_pool(x)
#             x = self.classifier(x)
#         return x
