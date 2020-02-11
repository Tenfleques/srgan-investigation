#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (Conv2d,SubpixelConv2d)

import numpy as np

import logging as _logging
from logging import DEBUG, ERROR, FATAL, INFO, WARN

from tensorflow.python.training import moving_averages
from tensorflow.python.ops import math_ops


import os
from abc import abstractmethod
from queue import Queue

import tensorflow as tf
from tensorflow.python.framework import ops as tf_ops

from tensorlayer.files import utils


_global_model_name_dict = {}  # TODO: better implementation?
_global_model_name_set = set()
_act_dict = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "leaky_relu": tf.nn.leaky_relu,
    "lrelu": tf.nn.leaky_relu,
    "softplus": tf.nn.softplus,
    "tanh": tf.nn.tanh,
    "sigmoid": tf.nn.sigmoid,
}


class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Parameters
        ----------
        shape : tuple of int.
            The shape of the tensor.
        dtype : Optional dtype of the tensor.
            If not provided will return tensor of `tf.float32`.

        Returns
        -------

        """
        raise NotImplementedError

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.

        Returns
        -------
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Parameters
        ----------
        config : A python dictionary.
            It will typically be the output of `get_config`.

        Returns
        -------
            An Initializer instance.
        """
        if 'dtype' in config:
            config.pop('dtype')
        return cls(**config)


class Zeros(Initializer):
    """Initializer that generates tensors initialized to 0.
    """

    def __call__(self, shape, dtype=tf.float32):
        return tf.zeros(shape, dtype=dtype)


class Ones(Initializer):
    """Initializer that generates tensors initialized to 1.
    """

    def __call__(self, shape, dtype=tf.float32):
        return tf.ones(shape, dtype=dtype)


class Constant(Initializer):
    """Initializer that generates tensors initialized to a constant value.

    Parameters
    ----------
    value : A python scalar or a numpy array.
        The assigned value.

    """

    def __init__(self, value=0):
        self.value = value

    def __call__(self, shape, dtype=None):
        return tf.constant(self.value, shape=shape, dtype=dtype)

    def get_config(self):
        return {"value": self.value}


class RandomUniform(Initializer):
    """Initializer that generates tensors with a uniform distribution.

    Parameters
    ----------
    minval : A python scalar or a scalar tensor.
        Lower bound of the range of random values to generate.
    maxval : A python scalar or a scalar tensor.
        Upper bound of the range of random values to generate.
    seed : A Python integer.
        Used to seed the random generator.

    """

    def __init__(self, minval=-0.05, maxval=0.05, seed=None):
        self.minval = minval
        self.maxval = maxval
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        return tf.random.uniform(shape, self.minval, self.maxval, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"minval": self.minval, "maxval": self.maxval, "seed": self.seed}


class RandomNormal(Initializer):
    """Initializer that generates tensors with a normal distribution.

    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the random values to generate.
    seed : A Python integer.
        Used to seed the random generator.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        return tf.random.normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


class TruncatedNormal(Initializer):
    """Initializer that generates a truncated normal distribution.

    These values are similar to values from a `RandomNormal`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.


    Parameters
    ----------
    mean : A python scalar or a scalar tensor.
        Mean of the random values to generate.
    stddev : A python scalar or a scalar tensor.
        Standard deviation of the andom values to generate.
    seed : A Python integer.
        Used to seed the random generator.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tf.float32):
        return tf.random.truncated_normal(shape, self.mean, self.stddev, dtype=dtype, seed=self.seed)

    def get_config(self):
        return {"mean": self.mean, "stddev": self.stddev, "seed": self.seed}


def deconv2d_bilinear_upsampling_initializer(shape):
    """Returns the initializer that can be passed to DeConv2dLayer for initializing the
    weights in correspondence to channel-wise bilinear up-sampling.
    Used in segmentation approaches such as [FCN](https://arxiv.org/abs/1605.06211)

    Parameters
    ----------
    shape : tuple of int
        The shape of the filters, [height, width, output_channels, in_channels].
        It must match the shape passed to DeConv2dLayer.

    Returns
    -------
    ``tf.constant_initializer``
        A constant initializer with weights set to correspond to per channel bilinear upsampling
        when passed as W_int in DeConv2dLayer

    """
    if shape[0] != shape[1]:
        raise Exception('deconv2d_bilinear_upsampling_initializer only supports symmetrical filter sizes')

    if shape[3] < shape[2]:
        raise Exception(
            'deconv2d_bilinear_upsampling_initializer behaviour is not defined for num_in_channels < num_out_channels '
        )

    filter_size = shape[0]
    num_out_channels = shape[2]
    num_in_channels = shape[3]

    # Create bilinear filter kernel as numpy array
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_out_channels, num_in_channels), dtype=np.float32)
    for i in range(num_out_channels):
        weights[:, :, i, i] = bilinear_kernel

    # assign numpy array to constant_initalizer and pass to get_variable
    return tf.constant_initializer(value=weights)



def flatten_reshape(variable, name='flatten'):
    """Reshapes a high-dimension vector input.

    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row x mask_col x n_mask]

    Parameters
    ----------
    variable : TensorFlow variable or tensor
        The variable or tensor to be flatten.
    name : str
        A unique layer name.

    Returns
    -------
    Tensor
        Flatten Tensor

    """
    dim = 1
    for d in variable.get_shape()[1:].as_list():
        dim *= d
    return tf.reshape(variable, shape=[-1, dim], name=name)

def _to_channel_first_bias(b):
    """Reshape [c] to [c, 1, 1]."""
    channel_size = int(b.shape[0])
    new_shape = (channel_size, 1, 1)
    # new_shape = [-1, 1, 1]  # doesn't work with tensorRT
    return tf.reshape(b, new_shape)


def _bias_scale(x, b, data_format):
    """The multiplication counter part of tf.nn.bias_add."""
    if data_format == 'NHWC':
        return x * b
    elif data_format == 'NCHW':
        return x * _to_channel_first_bias(b)
    else:
        raise ValueError('invalid data_format: %s' % data_format)


def _bias_add(x, b, data_format):
    """Alternative implementation of tf.nn.bias_add which is compatiable with tensorRT."""
    if data_format == 'NHWC':
        return tf.add(x, b)
    elif data_format == 'NCHW':
        return tf.add(x, _to_channel_first_bias(b))
    else:
        raise ValueError('invalid data_format: %s' % data_format)

def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, data_format, name=None):
    """Data Format aware version of tf.nn.batch_normalization."""
    if data_format == 'channels_last':
        mean = tf.reshape(mean, [1] * (len(x.shape) - 1) + [-1])
        variance = tf.reshape(variance, [1] * (len(x.shape) - 1) + [-1])
        offset = tf.reshape(offset, [1] * (len(x.shape) - 1) + [-1])
        scale = tf.reshape(scale, [1] * (len(x.shape) - 1) + [-1])
    elif data_format == 'channels_first':
        mean = tf.reshape(mean, [1] + [-1] + [1] * (len(x.shape) - 2))
        variance = tf.reshape(variance, [1] + [-1] + [1] * (len(x.shape) - 2))
        offset = tf.reshape(offset, [1] + [-1] + [1] * (len(x.shape) - 2))
        scale = tf.reshape(scale, [1] + [-1] + [1] * (len(x.shape) - 2))
    else:
        raise ValueError('invalid data_format: %s' % data_format)

    with tf_ops.name_scope(name, 'batchnorm', [x, mean, variance, scale, offset]):
        inv = math_ops.rsqrt(variance + variance_epsilon)
        if scale is not None:
            inv *= scale

        a = math_ops.cast(inv, x.dtype)
        b = math_ops.cast(offset - mean * inv if offset is not None else -mean * inv, x.dtype)

        # Return a * x + b with customized data_format.
        # Currently TF doesn't have bias_scale, and tensorRT has bug in converting tf.nn.bias_add
        # So we reimplemted them to allow make the model work with tensorRT.
        # See https://github.com/tensorlayer/openpose-plus/issues/75 for more details.
        df = {'channels_first': 'NCHW', 'channels_last': 'NHWC'}
        return _bias_add(_bias_scale(x, a, df[data_format]), b, df[data_format])
def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def _check_tl_layer_tensors(tensors):
    if not isinstance(tensors, list):
        return hasattr(tensors, '_info')
    else:
        for t in tensors:
            if not hasattr(t, '_info'):
                return False
        return True


def _add_list_to_all_layers(list_member):
    temp_all_layers = list()
    for component in list_member:
        if isinstance(component, Layer):
            temp_all_layers.append(component)
            if not component._built:
                raise AttributeError("Layer %s not built yet." % repr(component))
        elif isinstance(component, Model):
            temp_all_layers.append(component)
        elif isinstance(component, list):
            temp_all_layers.extend(_add_list_to_all_layers(component))
    return temp_all_layers


class Layer(object):
    """The basic :class:`Layer` class represents a single layer of a neural network.

    It should be subclassed when implementing new types of layers.

    Parameters
    ----------
    name : str or None
        A unique layer name. If None, a unique name will be automatically assigned.

    Methods
    ---------
    __init__()
        Initializing the Layer.
    __call__()
        (1) Building the Layer if necessary. (2) Forwarding the computation.
    all_weights()
        Return a list of Tensor which are all weights of this Layer.
    trainable_weights()
        Return a list of Tensor which are all trainable weights of this Layer.
    nontrainable_weights()
        Return a list of Tensor which are all nontrainable weights of this Layer.
    build()
        Abstract method. Build the Layer. All trainable weights should be defined in this function.
    forward()
        Abstract method. Forward computation and return computation results.

    """

    def __init__(self, name=None, act=None, *args, **kwargs):
        """
        Initializing the Layer.

        :param name: str or None
        :param name: str or function or None
        """

        # Layer constants
        # for key in kwargs.keys():
        #     setattr(self, key, self._argument_dict_checkup(kwargs[key]))

        # Auto naming if the name is not given
        global _global_layer_name_dict
        if name is None:
            prefix = self.__class__.__name__.lower()

            if _global_layer_name_dict.get(prefix) is not None:
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
            else:
                _global_layer_name_dict[prefix] = 0
                name = prefix
            while True:
                if _global_layer_name_dict.get(name) is None:
                    break
                _global_layer_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_layer_name_dict[prefix])
        else:
            if _global_layer_name_dict.get(name) is not None:
                pass
                # raise ValueError(
                #     'Layer name \'%s\' has already been used by another layer. Please change the layer name.' % name
                # )
            else:
                _global_layer_name_dict[name] = 0

        self.name = name
        if isinstance(act, str):
            self.act = str2act(act)
        else:
            self.act = act

        # Layer building state
        self._built = False

        # Layer nodes state
        self._nodes = []
        self._nodes_fixed = False

        # Layer weight state
        self._all_weights = None
        self._trainable_weights = []
        self._nontrainable_weights = []

        # nested layers
        self._layers = None

        # Layer training state
        self.is_train = True

        # layer config and init_args
        self._config = None
        self.layer_args = self._get_init_args(skip=3)

    @staticmethod
    def _compute_shape(tensors):
        if isinstance(tensors, list):
            shape_mem = [t.get_shape().as_list() for t in tensors]
        else:
            shape_mem = tensors.get_shape().as_list()
        return shape_mem

    @property
    def config(self):
        # if not self._nodes_fixed:
        #     raise RuntimeError("Model can not be saved when nodes are not fixed.")
        if self._config is not None:
            return self._config
        else:
            _config = {}
            _config.update({'class': self.__class__.__name__.split('.')[-1]})
            self.layer_args.update(self.get_args())
            self.layer_args["name"] = self.name
            _config.update({"args": self.layer_args})
            if self.__class__.__name__ in tl.layers.inputs.__all__:
                _config.update({'prev_layer': None})
            else:
                _config.update({'prev_layer': []})
                for node in self._nodes:
                    in_nodes = node.in_nodes
                    if not isinstance(in_nodes, list):
                        prev_name = in_nodes.name
                    else:
                        prev_name = [in_node.name for in_node in in_nodes]
                        if len(prev_name) == 1:
                            prev_name = prev_name[0]
                    _config['prev_layer'].append(prev_name)
            if self._nodes_fixed:
                self._config = _config
            return _config

    @property
    def all_weights(self):
        if self._all_weights is not None and len(self._all_weights) > 0:
            pass
        else:
            self._all_weights = self.trainable_weights + self.nontrainable_weights
        return self._all_weights

    @property
    def trainable_weights(self):
        nested = self._collect_sublayers_attr('trainable_weights')
        return self._trainable_weights + nested

    @property
    def nontrainable_weights(self):
        nested = self._collect_sublayers_attr('nontrainable_weights')
        return self._nontrainable_weights + nested

    @property
    def weights(self):
        raise Exception(
            "no property .weights exists, do you mean .all_weights, .trainable_weights, or .nontrainable_weights ?"
        )

    def _collect_sublayers_attr(self, attr):
        if attr not in ['trainable_weights', 'nontrainable_weights']:
            raise ValueError(
                "Only support to collect some certain attributes of nested layers,"
                "e.g. 'trainable_weights', 'nontrainable_weights', but got {}".format(attr)
            )
        if self._layers is None:
            return []
        nested = []
        for layer in self._layers:
            value = getattr(layer, attr)
            if value is not None:
                nested.extend(value)
        return nested

    def __call__(self, inputs, *args, **kwargs):
        """
        (1) Build the Layer if necessary.
        (2) Forward the computation and return results.
        (3) Add LayerNode if necessary

        :param prev_layer: np.ndarray, Tensor, Layer, list of Layers
        :param kwargs:
        :return: Layer
        """
        if self.__class__.__name__ in tl.layers.inputs.__all__:
            input_tensors = tf.convert_to_tensor(inputs)
        else:
            input_tensors = inputs

        if not self._built:
            if isinstance(self, LayerList):
                self._input_tensors = input_tensors
            inputs_shape = self._compute_shape(input_tensors)
            self.build(inputs_shape)
            self._built = True

        outputs = self.forward(input_tensors, *args, **kwargs)

        if not self._nodes_fixed:
            self._add_node(input_tensors, outputs)

        return outputs

    def _add_node(self, input_tensors, output_tensors):
        """Add a LayerNode for this layer given input_tensors, output_tensors.

        WARINING: This function should not be called from outside, it should only be called
        in layer.__call__ when building static model.

        Parameters
        ----------
        input_tensors : Tensor or a list of tensors
            Input tensors to this layer.
        output_tensors : Tensor or a list of tensors
            Output tensors to this layer.

        """
        inputs_list = tolist(input_tensors)
        outputs_list = tolist(output_tensors)

        if self.__class__.__name__ in tl.layers.inputs.__all__:
            # for InputLayer, there should be no in_nodes
            in_nodes = []
            in_tensor_idxes = [0]
        else:
            in_nodes = [tensor._info[0] for tensor in inputs_list]
            in_tensor_idxes = [tensor._info[1] for tensor in inputs_list]
        node_index = len(self._nodes)

        new_node = LayerNode(self, node_index, in_nodes, inputs_list, outputs_list, in_tensor_idxes)
        self._nodes.append(new_node)
        for idx, tensor in enumerate(outputs_list):
            tensor._info = (new_node, idx)  # FIXME : modify tensor outside layers? how to deal?

    def _release_memory(self):
        """
        WARINING: This function should be called with great caution.

        self.inputs and self.outputs will be set as None but not deleted in order to release memory.
        """
        # FIXME : not understand why saving inputs/outputs shape
        for node in self._nodes:
            node.in_tensors = None
            node.out_tensors = None

    def _set_mode_for_layers(self, is_train):
        """ Set training/evaluation mode for the Layer"""
        self.is_train = is_train

    def _fix_nodes_for_layers(self):
        """ fix LayerNodes to stop growing for this layer"""
        self._nodes_fixed = True

    def _get_weights(self, var_name, shape, init=RandomNormal(), trainable=True):
        """ Get trainable variables. """
        weight = get_variable_with_initializer(scope_name=self.name, var_name=var_name, shape=shape, init=init)
        if trainable is True:
            if self._trainable_weights is None:
                self._trainable_weights = list()
            self._trainable_weights.append(weight)
        else:
            if self._nontrainable_weights is None:
                self._nontrainable_weights = list()
            self._nontrainable_weights.append(weight)
        return weight

    @abstractmethod
    def build(self, inputs_shape):
        """
        An abstract method which should be overwritten in derived classes
        to define all necessary trainable weights of the layer.

        self.built should be set as True after self.build() is called.

        :param inputs_shape: tuple
        """
        raise Exception("The build(self, inputs_shape) method must be implemented by inherited class")

    @abstractmethod
    def forward(self, inputs):
        """
        An abstract method which should be overwritten in derived classes
        to define forward feeding operations of the layer.

        :param inputs: Tensor
        :return: Tensor
        """
        raise Exception("The forward method must be implemented by inherited class")

    @abstractmethod
    def __repr__(self):
        reprstr = "Layer"
        return reprstr

    def __setitem__(self, key, item):
        raise TypeError("The Layer API does not allow to use the method: `__setitem__`")

    def __delitem__(self, key):
        raise TypeError("The Layer API does not allow to use the method: `__delitem__`")

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            value._fix_nodes_for_layers()
            if self._layers is None:
                self._layers = []
            self._layers.append(value)
        super().__setattr__(key, value)

    def __delattr__(self, name):
        value = getattr(self, name, None)
        if isinstance(value, Layer):
            self._layers.remove(value)
        super().__delattr__(name)

#     @protected_method
    def get_args(self):
        init_args = {"layer_type": "normal"}
        return init_args

#     @protected_method
    def _get_init_args(self, skip=3):
        """Get all arguments of current layer for saving the graph."""
        stack = inspect.stack()

        if len(stack) < skip + 1:
            raise ValueError("The length of the inspection stack is shorter than the requested start position.")

        args, _, _, values = inspect.getargvalues(stack[skip][0])

        params = {}

        for arg in args:

            # some args dont need to be saved into the graph. e.g. the input placeholder
            if values[arg] is not None and arg not in ['self', 'prev_layer', 'inputs']:

                val = values[arg]

                if arg == "dtype" and isinstance(val, tf.DType):
                    params[arg] = repr(val)
                    continue

                # change function (e.g. act) into dictionary of module path and function name
                if inspect.isfunction(val):
                    if ("__module__" in dir(val)) and (len(val.__module__) >
                                                       10) and (val.__module__[0:10] == "tensorflow"):
                        params[arg] = val.__name__
                    else:
                        params[arg] = ('is_Func', utils.func2str(val))
                    # if val.__name__ == "<lambda>":
                    #     params[arg] = utils.lambda2str(val)
                    # else:
                    #     params[arg] = {"module_path": val.__module__, "func_name": val.__name__}
                # ignore more args e.g. TL initializer
                elif arg.endswith('init'):
                    continue
                # for other data type, save them directly
                else:
                    params[arg] = val

        return params
    
    
class LayerNode(object):
    """
    The class :class:`LayerNode` class represents a conceptional node for a layer.

    LayerNode is used for building static model and it is actually a light weighted
    wrapper over Layer. Specifically, it is used for building static computational graph
    (see _construct_graph() in tl.models.Model). In static model, each layer relates to
    one or more LayerNode, and the connection relationship between layers is built upon
    LayerNode. In addition, LayerNode eases layer reuse and weights sharing.

    Parameters
    ----------
    layer : tl.layers.Layer
        A tl layer that wants to create a node.
    node_index : int
        Index of this node in layer._nodes.
    in_nodes ：a list of LayerNode
        Father nodes to this node.
    in_tensors : a list of tensors
        Input tensors to this node.
    out_tensors : a list of tensors
        Output tensors to this node.
    in_tensor_idxes : a list of int
        Indexes of each input tensor in its corresponding node's out_tensors.

    Methods
    ---------
    __init__()
        Initializing the LayerNode.
    __call__()
        (1) Forwarding through the layer. (2) Update its input/output tensors.
    """

    def __init__(self, layer, node_index, in_nodes, in_tensors, out_tensors, in_tensor_idxes):
        """

        Parameters
        ----------
        layer
        node_index
        in_nodes
        in_tensors
        out_tensors
        in_tensor_idxes
        """
        self.layer = layer
        self.node_index = node_index
        self.in_nodes = in_nodes
        self.out_nodes = []
        self.in_tensors = in_tensors
        self.out_tensors = out_tensors
        self.name = layer.name + "_node_{}".format(node_index)

        self.in_tensors_idxes = in_tensor_idxes

        self.visited = False

    def __call__(self, inputs, **kwargs):
        """(1) Forwarding through the layer. (2) Update its input/output tensors."""
        outputs = self.layer.forward(inputs, **kwargs)
        self.in_tensors = tolist(inputs)
        self.out_tensors = tolist(outputs)
        return self.out_tensors

class ModelLayer(Layer):
    """
    The class :class:`ModelLayer` converts a :class:`Model` to a :class:`Layer` instance.

    Note that only a :class:`Model` with specified inputs and outputs can be converted to a :class:`ModelLayer`.
    For example, a customized model in dynamic eager mode normally does NOT have specified inputs and outputs so the
    customized model in dynamic eager mode can NOT be converted to a :class:`ModelLayer`.

    Parameters
    ----------
    model: tl.models.Model
        A model.
    name : str or None
        A unique layer name. If None, a unique name will be automatically assigned.

    Methods
    ---------
    __init__()
        Initializing the ModelLayer.
    weights()
        Same as the weights of the given model.
    build()
        Do nothing because the given model has already been built.
    forward()
        Forward the computation. Simply call the forward() of the given model.
    """

    def __init__(self, model, name=None):
        """
        Initializing the ModelLayer given a instance of Model.

        :param model:  tl.models.Model
        """
        super(ModelLayer, self).__init__(name=name)

        self.model = model

        # Layer building state
        self._built = True

        # Layer weight state
        self._all_weights = model.all_weights
        self._trainable_weights = model.trainable_weights
        self._nontrainable_weights = model.nontrainable_weights

        # Layer training state
        self.is_train = True

        logging.info("ModelLayer %s from Model: %s" % (self.name, self.model.name))

    def __repr__(self):
        tmpstr = 'ModelLayer' + '(\n'

        modstr = self.model.__repr__()
        modstr = _addindent(modstr, 2)

        tmpstr += modstr + ')'
        return tmpstr

    def build(self, inputs_shape):
        pass

    def forward(self, inputs):
        return self.model.forward(inputs)

    def _set_mode_for_layers(self, is_train):
        """ Set training/evaluation mode for the ModelLayer."""
        self.is_train = is_train
        return self.model._set_mode_for_layers(is_train)

    def _fix_nodes_for_layers(self):
        """ fix LayerNodes to stop growing for this ModelLayer."""
        self._nodes_fixed = True
        self.model._fix_nodes_for_layers()

    def _release_memory(self):
        """
        WARINING: This function should be called with great caution.

        self.inputs and self.outputs will be set as None but not deleted in order to release memory.
        """

        super(ModelLayer, self)._release_memory()
        self.model.release_memory()

    def get_args(self):
        init_args = {}
        init_args.update({"layer_type": "modellayer"})
        # init_args["model"] = utils.net2static_graph(self.layer_args["model"])
        init_args["model"] = self.layer_args["model"].config
        return init_args
    
class Model(object):
    """The :class:`Model` class represents a neural network.

    It should be subclassed when implementing a dynamic model,
    where 'forward' method must be overwritten.
    Otherwise, please specify 'inputs' tensor(s) and 'outputs' tensor(s)
    to create a static model. In that case, 'inputs' tensors should come
    from tl.layers.Input().

    Parameters
    -----------
    inputs : a Layer or list of Layer
        The input(s) to the model.
    outputs : a Layer or list of Layer
        The output(s) to the model.
    name : None or str
        The name of the model.

    Methods
    ---------
    __init__(self, inputs=None, outputs=None, name=None)
        Initializing the Model.
    inputs()
        Get input tensors to this network (only avaiable for static model).
    outputs()
        Get output tensors to this network (only avaiable for static model).
    __call__(inputs, is_train=None, **kwargs)
        Forward input tensors through this network.
    all_layers()
        Get all layer objects of this network in a list of layers.
    weights()
        Get the weights of this network in a list of tensors.
    train()
        Set this network in training mode. (affect layers e.g. Dropout, BatchNorm).
    eval()
        Set this network in evaluation mode.
    as_layer()
        Set this network as a ModelLayer so that it can be integrated into another Model.
    release_memory()
        Release the memory that was taken up by tensors which are maintained by this network.
    save_weights(self, filepath, format='hdf5')
        Save the weights of this network in a given format.
    load_weights(self, filepath, format=None, in_order=True, skip=False)
        Load weights into this network from a specified file.
    save(self, filepath, save_weights=True)
        Save the network with/without weights.
    load(filepath, save_weights=True)
        Load the network with/without weights.

    Examples
    ---------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from tensorlayer.layers import Input, Dense, Dropout
    >>> from tensorlayer.models import Model

    Define static model

    >>> class CustomModel(Model):
    >>>     def __init__(self):
    >>>         super(CustomModel, self).__init__()
    >>>         self.dense1 = Dense(n_units=800, act=tf.nn.relu, in_channels=784)
    >>>         self.dropout1 = Dropout(keep=0.8)
    >>>         self.dense2 = Dense(n_units=10, in_channels=800)
    >>>     def forward(self, x):
    >>>         z = self.dense1(x)
    >>>         z = self.dropout1(z)
    >>>         z = self.dense2(z)
    >>>         return z
    >>> M_dynamic = CustomModel()

    Define static model

    >>> ni = Input([None, 784])
    >>> nn = Dense(n_units=800, act=tf.nn.relu)(ni)
    >>> nn = Dropout(keep=0.8)(nn)
    >>> nn = Dense(n_units=10, act=tf.nn.relu)(nn)
    >>> M_static = Model(inputs=ni, outputs=nn, name="mlp")

    Get network information

    >>> print(M_static)
    ... Model(
    ...  (_inputlayer): Input(shape=[None, 784], name='_inputlayer')
    ...  (dense): Dense(n_units=800, relu, in_channels='784', name='dense')
    ...  (dropout): Dropout(keep=0.8, name='dropout')
    ...  (dense_1): Dense(n_units=10, relu, in_channels='800', name='dense_1')
    ... )

    Forwarding through this network

    >>> data = np.random.normal(size=[16, 784]).astype(np.float32)
    >>> outputs_d = M_dynamic(data)
    >>> outputs_s = M_static(data)

    Save and load weights

    >>> M_static.save_weights('./model_weights.h5')
    >>> M_static.load_weights('./model_weights.h5')

    Save and load the model

    >>> M_static.save('./model.h5')
    >>> M = Model.load('./model.h5')

    Convert model to layer

    >>> M_layer = M_static.as_layer()

    """

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def __init__(self, inputs=None, outputs=None, name=None):
        """
        Initializing the Model.

        Parameters
        ----------
        inputs : Tensor or list of tensors
            Input tensor(s), which must come from tl.layers.Input()
        outputs : Tensor or list of tensors
            Output tensor(s), which must be the output(s) of some TL layers
        name : str or None
            Name for this network
        """
        # Auto naming if the name is not given
        self._NameNone = False
        global _global_model_name_dict
        global _global_model_name_set
        if name is None:
            self._NameNone = True
            prefix = self.__class__.__name__.lower()
            if _global_model_name_dict.get(prefix) is not None:
                _global_model_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_model_name_dict[prefix])
            else:
                _global_model_name_dict[prefix] = 0
                name = prefix
            while name in _global_model_name_set:
                _global_model_name_dict[prefix] += 1
                name = prefix + '_' + str(_global_model_name_dict[prefix])
            _global_model_name_set.add(name)
        else:
            if name in _global_model_name_set:
                raise ValueError(
                    'Model name \'%s\' has already been used by another model. Please change the model name.' % name
                )
            _global_model_name_set.add(name)
            _global_model_name_dict[name] = 0

        # Model properties
        self.name = name

        # Model state: train or test
        self.is_train = None

        # Model weights
        self._all_weights = None
        self._trainable_weights = None
        self._nontrainable_weights = None

        # Model args of all layers, ordered by all_layers
        self._config = None

        # Model inputs and outputs
        # TODO: note that in dynamic network, inputs and outputs are both None, may cause problem, test needed
        self._inputs = inputs
        self._outputs = outputs

        # Model converted into a Layer
        self._model_layer = None

        # Layer Node status
        self._nodes_fixed = False

        # Model layers
        self._all_layers = None

        if inputs is None and outputs is None:
            pass

        else:
            # check type of inputs and outputs
            check_order = ['inputs', 'outputs']
            for co, check_argu in enumerate([inputs, outputs]):
                if isinstance(check_argu, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(check_argu):
                    pass
                elif isinstance(check_argu, list):
                    if len(check_argu) == 0:
                        raise ValueError(
                            "The argument `%s` is detected as an empty list. " % check_order[co] +
                            "It should be either Tensor or a list of Tensor."
                        )
                    for idx in range(len(check_argu)):
                        if not isinstance(check_argu[idx], tf_ops._TensorLike) or not tf_ops.is_dense_tensor_like(
                                check_argu[idx]):
                            raise TypeError(
                                "The argument `%s` should be either Tensor or a list of Tensor " % (check_order[co]) +
                                "but the %s[%d] is detected as %s" % (check_order[co], idx, type(check_argu[idx]))
                            )
                else:
                    raise TypeError(
                        "The argument `%s` should be either Tensor or a list of Tensor but received %s" %
                        (check_order[co], type(check_argu))
                    )

            if not _check_tl_layer_tensors(inputs):
                raise TypeError(
                    "The argument `inputs` should be either Tensor or a list of Tensor "
                    "that come from TensorLayer's Input layer: tl.layers.Input(shape). "
                )
            if not _check_tl_layer_tensors(outputs):
                raise TypeError(
                    "The argument `outputs` should be either Tensor or a list of Tensor "
                    "that is/are outputs from some TensorLayer's layers, e.g. tl.layers.Dense, tl.layers.Conv2d."
                )

            # build network graph
            self._node_by_depth, self._all_layers = self._construct_graph()

            self._fix_nodes_for_layers()

    def __call__(self, inputs, is_train=None, **kwargs):
        """Forward input tensors through this network by calling.

        Parameters
        ----------
        inputs : Tensor or list of Tensors, numpy.ndarray of list of numpy.ndarray
            Inputs for network forwarding
        is_train : boolean
            Network's mode for this time forwarding. If 'is_train' == True, this network is set as training mode.
            If 'is_train' == False, this network is set as evaluation mode
        kwargs :
            For other keyword-only arguments.

        """

        self._check_mode(is_train)

        # FIXME: this may cause inefficiency, this is used to check if every layer is built
        self.all_layers

        # fix LayerNodes when first calling
        if self._nodes_fixed is False:
            self._fix_nodes_for_layers()

        # set training / inference mode if necessary
        if is_train is not None:
            self._set_mode_for_layers(is_train)

        # if self._input is a list, then it must be a static network
        if isinstance(self._inputs, list):
            if not isinstance(inputs, list):
                raise ValueError("The argument `inputs` should be a list of values but detected as %s." % type(inputs))
            elif len(inputs) != len(self._inputs):
                raise ValueError(
                    "The argument `inputs` should be a list with len=%d but detected as len=%d." %
                    (len(self._inputs), len(inputs))
                )

        # convert inputs to tensor if it is originally not
        # FIXME: not sure convert_to_tensor here or ask user to do it
        if isinstance(inputs, list):
            for idx in range(len(inputs)):
                inputs[idx] = tf.convert_to_tensor(inputs[idx])
        else:
            inputs = tf.convert_to_tensor(inputs)

        return self.forward(inputs, **kwargs)

    @abstractmethod
    def forward(self, *inputs, **kwargs):
        """Network forwarding given input tensors

        Parameters
        ----------
        inputs : Tensor or list of Tensors
            input tensor(s)
        kwargs :
            For other keyword-only arguments.

        Returns
        -------
            output tensor(s) : Tensor or list of Tensor(s)

        """
        # FIXME: currently using self._outputs to judge static network or dynamic network
        if self._outputs is None:
            raise ValueError(
                "Outputs not defined. Please define inputs and outputs when the model is created. Or overwrite forward() function."
            )

        memory = dict()

        # get each layer's output by going through the graph in depth order
        for depth, nodes in enumerate(self._node_by_depth):
            if depth == 0:
                if isinstance(self.inputs, list):
                    assert len(inputs[0]) == len(nodes)
                    for idx, node in enumerate(nodes):
                        memory[node.name] = node(inputs[0][idx])
                else:
                    memory[nodes[0].name] = nodes[0](inputs[0])
            else:
                for node in nodes:
                    in_nodes = node.in_nodes
                    in_tensors_idxes = node.in_tensors_idxes
                    if len(in_nodes) == 1:
                        node_input = memory[in_nodes[0].name][in_tensors_idxes[0]]
                    else:
                        node_input = [memory[inode.name][idx] for inode, idx in zip(in_nodes, in_tensors_idxes)]
                    memory[node.name] = node(node_input)

        if not isinstance(self._outputs, list):
            return memory[self._outputs._info[0].name][self._outputs._info[1]]
        else:
            return [memory[tensor._info[0].name][tensor._info[1]] for tensor in self._outputs]

    @property
    def all_layers(self):
        """Return all layers of this network in a list."""
        if self._all_layers is not None:
            return self._all_layers

        if self._inputs is not None and self._outputs is not None:
            # static model
            return self._all_layers
        else:
            # dynamic model
            self._all_layers = list()
            attr_list = [attr for attr in dir(self) if attr[:2] != "__"]
            attr_list.remove("all_weights")
            attr_list.remove("trainable_weights")
            attr_list.remove("nontrainable_weights")
            attr_list.remove("_all_weights")
            attr_list.remove("_trainable_weights")
            attr_list.remove("_nontrainable_weights")
            attr_list.remove("all_layers")
            attr_list.remove("_all_layers")
            attr_list.remove("n_weights")
            for idx, attr in enumerate(attr_list):
                try:
                    if isinstance(getattr(self, attr), Layer):
                        nowlayer = getattr(self, attr)
                        if not nowlayer._built:
                            raise AttributeError("Layer %s not built yet." % repr(nowlayer))
                        self._all_layers.append(nowlayer)
                    elif isinstance(getattr(self, attr), Model):
                        nowmodel = getattr(self, attr)
                        self._all_layers.append(nowmodel)
                    elif isinstance(getattr(self, attr), list):
                        self._all_layers.extend(_add_list_to_all_layers(getattr(self, attr)))
                # TODO: define customised exception for TL
                except AttributeError as e:
                    raise e
                except Exception:
                    pass

            # check layer name uniqueness
            local_layer_name_dict = set()
            for layer in self._all_layers:
                if layer.name in local_layer_name_dict:
                    raise ValueError(
                        'Layer name \'%s\' has already been used by another layer. Please change the layer name.' %
                        layer.name
                    )
                else:
                    local_layer_name_dict.add(layer.name)
            return self._all_layers

    @property
    def trainable_weights(self):
        """Return trainable weights of this network in a list."""
        if self._trainable_weights is not None and len(self._trainable_weights) > 0:
            # self._trainable_weights already extracted, so do nothing
            pass
        else:
            self._trainable_weights = []
            for layer in self.all_layers:
                if layer.trainable_weights is not None:
                    self._trainable_weights.extend(layer.trainable_weights)

        return self._trainable_weights.copy()

    @property
    def nontrainable_weights(self):
        """Return nontrainable weights of this network in a list."""
        if self._nontrainable_weights is not None and len(self._nontrainable_weights) > 0:
            # self._nontrainable_weights already extracted, so do nothing
            pass
        else:
            self._nontrainable_weights = []
            for layer in self.all_layers:
                if layer.nontrainable_weights is not None:
                    self._nontrainable_weights.extend(layer.nontrainable_weights)

        return self._nontrainable_weights.copy()

    @property
    def all_weights(self):
        """Return all weights of this network in a list."""
        if self._all_weights is not None and len(self._all_weights) > 0:
            # self._all_weights already extracted, so do nothing
            pass
        else:
            self._all_weights = []
            for layer in self.all_layers:
                if layer.all_weights is not None:
                    self._all_weights.extend(layer.all_weights)

        return self._all_weights.copy()

    @property
    def n_weights(self):
        """Return the number of weights (parameters) in this network."""
        n_weights = 0
        for i, w in enumerate(self.all_weights):
            n = 1
            # for s in p.eval().shape:
            for s in w.get_shape():
                try:
                    s = int(s)
                except:
                    s = 1
                if s:
                    n = n * s
            n_weights = n_weights + n
        # print("num of weights (parameters) %d" % n_weights)
        return n_weights

    @property
    def config(self):
        if self._config is not None and len(self._config) > 0:
            return self._config
        else:
            # _config = []
            _config = {}
            if self._NameNone is True:
                _config.update({"name": None})
            else:
                _config.update({"name": self.name})
            version_info = {
                "tensorlayer_version": tl.__version__,
                "backend": "tensorflow",
                "backend_version": tf.__version__,
                "training_device": "gpu",
                "save_date": None,
            }
            _config["version_info"] = version_info
            # if self.outputs is None:
            #     raise RuntimeError(
            #         "Dynamic mode does not support config yet."
            #     )
            model_architecture = []
            for layer in self.all_layers:
                model_architecture.append(layer.config)
            _config["model_architecture"] = model_architecture
            if self.inputs is not None:
                if not isinstance(self.inputs, list):
                    _config.update({"inputs": self.inputs._info[0].name})
                else:
                    config_inputs = []
                    for config_input in self.inputs:
                        config_inputs.append(config_input._info[0].name)
                    _config.update({"inputs": config_inputs})
            if self.outputs is not None:
                if not isinstance(self.outputs, list):
                    _config.update({"outputs": self.outputs._info[0].name})
                else:
                    config_outputs = []
                    for config_output in self.outputs:
                        config_outputs.append(config_output._info[0].name)
                    _config.update({"outputs": config_outputs})
            if self._nodes_fixed or self.outputs is None:
                self._config = _config

            return _config

    def train(self):
        """Set this network in training mode. After calling this method,
        all layers in network are in training mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.models.vgg16()
        >>> net.train()

        """
        if self.is_train !=True:
            self.is_train = True
            self._set_mode_for_layers(True)

    def eval(self):
        """Set this network in evaluation mode. After calling this method,
        all layers in network are in evaluation mode, in particular, BatchNorm, Dropout, etc.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> net = tl.models.vgg16()
        >>> net.eval()
        # do evaluation

        """
        if self.is_train != False:
            self.is_train = False
            self._set_mode_for_layers(False)

    def test(self):
        """Set this network in evaluation mode."""
        self.eval()

    def infer(self):
        """Set this network in evaluation mode."""
        self.eval()

    def as_layer(self):
        """Return this network as a ModelLayer so that it can be integrated into another Model.

        Examples
        --------
        >>> from tensorlayer.layers import Input, Dense, Dropout
        >>> from tensorlayer.models import Model
        >>> ni = Input([None, 784])
        >>> nn = Dense(n_units=800, act=tf.nn.relu)(ni)
        >>> nn = Dropout(keep=0.8)(nn)
        >>> nn = Dense(n_units=10, act=tf.nn.relu)(nn)
        >>> M_hidden = Model(inputs=ni, outputs=nn, name="mlp").as_layer()
        >>> nn = M_hidden(ni)   # use previously constructed model as layer
        >>> nn = Dropout(keep=0.8)(nn)
        >>> nn = Dense(n_units=10, act=tf.nn.relu)(nn)
        >>> M_full = Model(inputs=ni, outputs=nn, name="mlp")

        """
        if self._outputs is None:
            raise AttributeError("Dynamic network cannot be converted to Layer.")

        if self._model_layer is None:
            self._model_layer = ModelLayer(self)

        return self._model_layer

    def _check_mode(self, is_train):
        """Check whether this network is in a given mode.

        Parameters
        ----------
        is_train : boolean
            Network's mode. True means training mode while False means evaluation mode.

        """
        # contradiction test
        if is_train is None and self.is_train is None:
            raise ValueError(
                "Training / inference mode not defined. Argument `is_train` should be set as True / False. Otherwise please use `Model.train()` / `Model.eval()` to switch the mode."
            )
        elif is_train is not None and self.is_train is not None:
            if is_train == self.is_train:
                logging.warning(
                    "Training / inference mode redefined redundantly. Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode."
                )
            else:
                raise AttributeError(
                    "Training / inference mode mismatch. The argument `is_train` is set as %s, " % is_train +
                    "but the mode is currently set as %s. " %
                    ('Training by Model.train()' if self.is_train else 'Inference by Model.eval()') +
                    "Please EITHER use the argument `is_train` OR `Model.train()` / `Model.eval()` to define the mode."
                )

    def _set_mode_for_layers(self, is_train):
        """Set all layers of this network to a given mode.

        Parameters
        ----------
        is_train : boolean
            Network's mode. True means training mode while False means evaluation mode.

        """
        for layer in self.all_layers:
            if isinstance(layer, Model):
                layer.is_train = is_train
            layer._set_mode_for_layers(is_train)

    def _fix_nodes_for_layers(self):
        """Fix each Layer's LayerNode to stop growing, see LayerNode for more."""
        for layer in self.all_layers:
            layer._fix_nodes_for_layers()
        self._nodes_fixed = True

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            if value._built is False:
                raise AttributeError(
                    "The registered layer `{}` should be built in advance. "
                    "Do you forget to pass the keyword argument 'in_channels'? ".format(value.name)
                )
        super().__setattr__(key, value)

    def __repr__(self):
        # tmpstr = self.__class__.__name__ + '(\n'
        tmpstr = self.name + '(\n'
        for idx, layer in enumerate(self.all_layers):
            modstr = layer.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + layer.name + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr

    ## raise Exceptions for old version codes
    def print_all_layers(self):
        raise Exception("please change net.print_all_layers --> print(net)")

    def count_params(self, **kwargs):
        raise Exception("please change count_params --> count_weights")

    def print_params(self, **kwargs):
        raise Exception("please change print_params --> print_weights")

    @property
    def all_params(self):
        raise Exception("please change all_params --> weights")

    @property
    def all_drop(self):
        raise Exception("all_drop is deprecated")

    def get_layer(self, name=None, index=None):
        """Network forwarding given input tensors

        Parameters
        ----------
        name : str or None
            Name of the requested layer. Default None.
        index : int or None
            Index of the requested layer. Default None.

        Returns
        -------
            layer : The requested layer

        Notes
        -----
        Either a layer name or a layer index should be given.

        """
        if index is not None:
            if len(self.all_layers) <= index:
                raise ValueError(
                    'model only has ' + str(len(self.all_layers)) + ' layers, but ' + str(index) +
                    '-th layer is requested.'
                )
            else:
                return self.all_layers[index]
        elif name is not None:
            for layer in self.all_layers:
                if layer.name == name:
                    return layer
            raise ValueError('Model has no layer named ' + name + '.')
        else:
            raise ValueError('Either a layer name or a layer index should be given.')

    def _construct_graph(self):
        """construct computation graph for static model using LayerNode object"""
        all_layers = []
        node_by_depth = []  # [[node0, node1], [node2, node3], ...]

        input_tensors_list = self.inputs if isinstance(self.inputs, list) else [self.inputs]

        queue_node = Queue()

        # BFS to visit all nodes that should be involved in the computation graph
        output_tensors_list = self.outputs if isinstance(self.outputs, list) else [self.outputs]
        output_nodes = [tensor._info[0] for tensor in output_tensors_list]

        visited_node_names = set()
        for out_node in output_nodes:
            if out_node.visited:
                continue
            queue_node.put(out_node)

            while not queue_node.empty():
                cur_node = queue_node.get()
                in_nodes = cur_node.in_nodes

                for node in in_nodes:
                    node.out_nodes.append(cur_node)
                    if not node.visited:
                        queue_node.put(node)
                        node.visited = True
                        if node.name not in visited_node_names:
                            visited_node_names.add(node.name)
                        # else have multiple layers with the same name
                        else:
                            raise ValueError(
                                'Layer name \'%s\' has already been used by another layer. Please change the layer name.'
                                % node.layer.name
                            )

        # construct the computation graph in top-sort order
        cur_depth = [tensor._info[0] for tensor in input_tensors_list]
        next_depth = []
        indegrees = {}

        visited_layer_names = []
        while not len(cur_depth) == 0:
            node_by_depth.append(cur_depth)
            for node in cur_depth:
                if node.layer.name not in visited_layer_names:
                    all_layers.append(node.layer)
                    visited_layer_names.append(node.layer.name)
                for out_node in node.out_nodes:
                    if out_node.name not in indegrees.keys():
                        indegrees[out_node.name] = len(out_node.in_nodes)
                    indegrees[out_node.name] -= 1
                    if indegrees[out_node.name] == 0:
                        next_depth.append(out_node)

            cur_depth = next_depth
            next_depth = []

        return node_by_depth, all_layers

    def release_memory(self):
        '''
        WARNING: This function should be called with great caution.

        Release objects that MAY NOT be necessary such as layer.outputs (if in a tf.GradientTape() scope).
        For each layer in the model, layer.inputs and layer.outputs will be set as None but not deleted.

        A void function.

        Examples
        --------
        >>> import tensorlayer as tl
        >>> vgg = tl.models.vgg16()
        ... # training preparation
        ... # ...
        ... # back propagation
        >>> with tf.GradientTape() as tape:
        >>>     _logits = vgg(x_batch)
        >>>     ## compute loss and update model
        >>>     _loss = tl.cost.cross_entropy(_logits, y_batch, name='train_loss')
        >>>     ## release unnecessary objects (layer.inputs, layer.outputs)
        >>>     ## this function should be called with great caution
        >>>     ## within the scope of tf.GradientTape(), using this function should be fine
        >>>     vgg.release_memory()

        '''
        for layer in self.all_layers:
            layer._release_memory()

    def save(self, filepath, save_weights=True, customized_data=None):
        """
        Save model into a given file.
        This function save can save both the architecture of neural networks and weights (optional).
        WARNING: If the model contains Lambda / ElementwiseLambda layer, please check the documentation of Lambda / ElementwiseLambda layer and find out the cases that have / have not been supported by Model.save().

        Parameters
        ----------
        filepath : str
            Filename into which the model will be saved.
        save_weights : bool
            Whether to save model weights.
        customized_data : dict
            The user customized meta data.

        Examples
        --------
        >>> net = tl.models.vgg16()
        >>> net.save('./model.h5', save_weights=True)
        >>> new_net = Model.load('./model.h5', load_weights=True)

        """
        # TODO: support saving LambdaLayer that includes parametric self defined function with outside variables
        if self.outputs is None:
            raise RuntimeError(
                "Model save() not support dynamic mode yet.\nHint: you can use Model save_weights() to save the weights in dynamic mode."
            )
        utils.save_hdf5_graph(
            network=self, filepath=filepath, save_weights=save_weights, customized_data=customized_data
        )

    @staticmethod
    def load(filepath, load_weights=True):
        """
        Load model from a given file, which should be previously saved by Model.save().
        This function load can load both the architecture of neural networks and weights (optional, and needs to be saved in Model.save()).
        When a model is loaded by this function load, there is no need to reimplement or declare the architecture of the model explicitly in code.
        WARNING: If the model contains Lambda / ElementwiseLambda layer, please check the documentation of Lambda / ElementwiseLambda layer and find out the cases that have / have not been supported by Model.load().

        Parameters
        ----------
        filepath : str
            Filename from which the model will be loaded.
        load_weights : bool
            Whether to load model weights.

        Examples
        --------
        >>> net = tl.models.vgg16()
        >>> net.save('./model.h5', save_weights=True)
        >>> new_net = Model.load('./model.h5', load_weights=True)
        """
        # TODO: support loading LambdaLayer that includes parametric self defined function with outside variables
        M = utils.load_hdf5_graph(filepath=filepath, load_weights=load_weights)
        return M

    def save_weights(self, filepath, format=None):
        """Input filepath, save model weights into a file of given format.
            Use self.load_weights() to restore.

        Parameters
        ----------
        filepath : str
            Filename to which the model weights will be saved.
        format : str or None
            Saved file format.
            Value should be None, 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            1) If this is set to None, then the postfix of filepath will be used to decide saved format.
            If the postfix is not in ['h5', 'hdf5', 'npz', 'ckpt'], then file will be saved in hdf5 format by default.
            2) 'hdf5' will save model weights name in a list and each layer has its weights stored in a group of
            the hdf5 file.
            3) 'npz' will save model weights sequentially into a npz file.
            4) 'npz_dict' will save model weights along with its name as a dict into a npz file.
            5) 'ckpt' will save model weights into a tensorflow ckpt file.

            Default None.

        Examples
        --------
        1) Save model weights in hdf5 format by default.
        >>> net = tl.models.vgg16()
        >>> net.save_weights('./model.h5')
        ...
        >>> net.load_weights('./model.h5')

        2) Save model weights in npz/npz_dict format
        >>> net = tl.models.vgg16()
        >>> net.save_weights('./model.npz')
        >>> net.save_weights('./model.npz', format='npz_dict')

        """
        if self.all_weights is None or len(self.all_weights) == 0:
            logging.warning("Model contains no weights or layers haven't been built, nothing will be saved")
            return

        if format is None:
            postfix = filepath.split('.')[-1]
            if postfix in ['h5', 'hdf5', 'npz', 'ckpt']:
                format = postfix
            else:
                format = 'hdf5'

        if format == 'hdf5' or format == 'h5':
            utils.save_weights_to_hdf5(filepath, self)
        elif format == 'npz':
            utils.save_npz(self.all_weights, filepath)
        elif format == 'npz_dict':
            utils.save_npz_dict(self.all_weights, filepath)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "Save format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'."
                "Other format is not supported now."
            )

    def load_weights(self, filepath, format=None, in_order=True, skip=False):
        """Load model weights from a given file, which should be previously saved by self.save_weights().

        Parameters
        ----------
        filepath : str
            Filename from which the model weights will be loaded.
        format : str or None
            If not specified (None), the postfix of the filepath will be used to decide its format. If specified,
            value should be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. Other format is not supported now.
            In addition, it should be the same format when you saved the file using self.save_weights().
            Default is None.
        in_order : bool
            Allow loading weights into model in a sequential way or by name. Only useful when 'format' is 'hdf5'.
            If 'in_order' is True, weights from the file will be loaded into model in a sequential way.
            If 'in_order' is False, weights from the file will be loaded into model by matching the name
            with the weights of the model, particularly useful when trying to restore model in eager(graph) mode from
            a weights file which is saved in graph(eager) mode.
            Default is True.
        skip : bool
            Allow skipping weights whose name is mismatched between the file and model. Only useful when 'format' is
            'hdf5' or 'npz_dict'. If 'skip' is True, 'in_order' argument will be ignored and those loaded weights
            whose name is not found in model weights (self.all_weights) will be skipped. If 'skip' is False, error will
            occur when mismatch is found.
            Default is False.

        Examples
        --------
        1) load model from a hdf5 file.
        >>> net = tl.models.vgg16()
        >>> net.load_weights('./model_graph.h5', in_order=False, skip=True) # load weights by name, skipping mismatch
        >>> net.load_weights('./model_eager.h5') # load sequentially

        2) load model from a npz file
        >>> net.load_weights('./model.npz')

        2) load model from a npz file, which is saved as npz_dict previously
        >>> net.load_weights('./model.npz', format='npz_dict')

        Notes
        -------
        1) 'in_order' is only useful when 'format' is 'hdf5'. If you are trying to load a weights file which is
           saved in a different mode, it is recommended to set 'in_order' be True.
        2) 'skip' is useful when 'format' is 'hdf5' or 'npz_dict'. If 'skip' is True,
           'in_order' argument will be ignored.

        """
        if not os.path.exists(filepath):
            raise FileNotFoundError("file {} doesn't exist.".format(filepath))

        if format is None:
            format = filepath.split('.')[-1]

        if format == 'hdf5' or format == 'h5':
            if skip ==True or in_order == False:
                # load by weights name
                utils.load_hdf5_to_weights(filepath, self, skip)
            else:
                # load in order
                utils.load_hdf5_to_weights_in_order(filepath, self)
        elif format == 'npz':
            utils.load_and_assign_npz(filepath, self)
        elif format == 'npz_dict':
            utils.load_and_assign_npz_dict(filepath, self, skip)
        elif format == 'ckpt':
            # TODO: enable this when tf save ckpt is enabled
            raise NotImplementedError("ckpt load/save is not supported now.")
        else:
            raise ValueError(
                "File format must be 'hdf5', 'npz', 'npz_dict' or 'ckpt'. "
                "Other format is not supported now."
            )

    # TODO: not supported now
    # def save_ckpt(self, sess=None, mode_name='model.ckpt', save_dir='checkpoint', global_step=None, printable=False):
    #     # TODO: Documentation pending
    #     """"""
    #     if not os.path.exists(save_dir):
    #         raise FileNotFoundError("Save directory {} doesn't exist.".format(save_dir))
    #     utils.save_ckpt(sess, mode_name, save_dir, self.weights, global_step, printable)
    #
    # def load_ckpt(self, sess=None, mode_name='model.ckpt', save_dir='checkpoint', is_latest=True, printable=False):
    #     # TODO: Documentation pending
    #     """"""
    #     utils.load_ckpt(sess, mode_name, save_dir, self.weights, is_latest, printable)



class Dense(Layer):
    """The :class:`Dense` class is a fully connected layer.

    Parameters
    ----------
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer.
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    in_channels: int
        The number of channels of the previous layer.
        If None, it will be automatically detected when the layer is forwarded for the first time.
    name : None or str
        A unique layer name. If None, a unique name will be automatically generated.

    Examples
    --------
    With TensorLayer

    >>> net = tl.layers.Input([100, 50], name='input')
    >>> dense = tl.layers.Dense(n_units=800, act=tf.nn.relu, in_channels=50, name='dense_1')
    >>> print(dense)
    Dense(n_units=800, relu, in_channels='50', name='dense_1')
    >>> tensor = tl.layers.Dense(n_units=800, act=tf.nn.relu, name='dense_2')(net)
    >>> print(tensor)
    tf.Tensor([...], shape=(100, 800), dtype=float32)

    Notes
    -----
    If the layer input has more than two axes, it needs to be flatten by using :class:`Flatten`.

    """

    def __init__(
        self,
        n_units,
        act=None,
        W_init=TruncatedNormal(stddev=0.05),
        b_init=Constant(value=0.0),
        in_channels=None,
        name=None,  # 'dense',
    ):

        super(Dense, self).__init__(name, act=act)

        self.n_units = n_units
        self.W_init = W_init
        self.b_init = b_init
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build(self.in_channels)
            self._built = True

        logging.info(
            "Dense  %s: %d %s" %
            (self.name, self.n_units, self.act.__name__ if self.act is not None else 'No Activation')
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(n_units={n_units}, ' + actstr)
        if self.in_channels is not None:
            s += ', in_channels=\'{in_channels}\''
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        if self.in_channels is None and len(inputs_shape) != 2:
            raise AssertionError("The input dimension must be rank 2, please reshape or flatten it")
        if self.in_channels:
            shape = [self.in_channels, self.n_units]
        else:
            self.in_channels = inputs_shape[1]
            shape = [inputs_shape[1], self.n_units]
        self.W = self._get_weights("weights", shape=tuple(shape), init=self.W_init)
        if self.b_init:
            self.b = self._get_weights("biases", shape=(self.n_units, ), init=self.b_init)

    # @tf.function
    def forward(self, inputs):
        z = tf.matmul(inputs, self.W)
        if self.b_init:
            z = tf.add(z, self.b)
        if self.act:
            z = self.act(z)
        return z

    
class Flatten(Layer):
    """A layer that reshapes high-dimension input into a vector.

    Then we often apply Dense, RNN, Concat and etc on the top of a flatten layer.
    [batch_size, mask_row, mask_col, n_mask] ---> [batch_size, mask_row * mask_col * n_mask]

    Parameters
    ----------
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> x = tl.layers.Input([8, 4, 3], name='input')
    >>> y = tl.layers.Flatten(name='flatten')(x)
    [8, 12]

    """

    def __init__(self, name=None):  #'flatten'):
        super(Flatten, self).__init__(name)

        self.build()
        self._built = True

        logging.info("Flatten %s:" % (self.name))

    def __repr__(self):
        s = '{classname}('
        s += 'name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape=None):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = flatten_reshape(inputs, name=self.name)
        return outputs
    
class Elementwise(Layer):
    """A layer that combines multiple :class:`Layer` that have the same output shapes
    according to an element-wise operation.
    If the element-wise operation is complicated, please consider to use :class:`ElementwiseLambda`.

    Parameters
    ----------
    combine_fn : a TensorFlow element-wise combine function
        e.g. AND is ``tf.minimum`` ;  OR is ``tf.maximum`` ; ADD is ``tf.add`` ; MUL is ``tf.multiply`` and so on.
        See `TensorFlow Math API <https://www.tensorflow.org/versions/master/api_docs/python/math_ops.html#math>`__ .
        If the combine function is more complicated, please consider to use :class:`ElementwiseLambda`.
    act : activation function
        The activation function of this layer.
    name : None or str
        A unique layer name.

    Examples
    --------
    >>> class CustomModel(tl.models.Model):
    >>>     def __init__(self):
    >>>         super(CustomModel, self).__init__(name="custom")
    >>>         self.dense1 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu1_1')
    >>>         self.dense2 = tl.layers.Dense(in_channels=20, n_units=10, act=tf.nn.relu, name='relu2_1')
    >>>         self.element = tl.layers.Elementwise(combine_fn=tf.minimum, name='minimum', act=tf.identity)

    >>>     def forward(self, inputs):
    >>>         d1 = self.dense1(inputs)
    >>>         d2 = self.dense2(inputs)
    >>>         outputs = self.element([d1, d2])
    >>>         return outputs
    """

    def __init__(
        self,
        combine_fn=tf.minimum,
        act=None,
        name=None,  #'elementwise',
    ):

        super(Elementwise, self).__init__(name, act=act)
        self.combine_fn = combine_fn

        self.build(None)
        self._built = True

        logging.info(
            "Elementwise %s: fn: %s act: %s" %
            (self.name, combine_fn.__name__, ('No Activation' if self.act is None else self.act.__name__))
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(combine_fn={combine_fn}, ' + actstr)
        if self.name is not None:
            s += ', name=\'{name}\''
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def build(self, inputs_shape):
        pass

    # @tf.function
    def forward(self, inputs):
        outputs = inputs[0]
        for input in inputs[1:]:
            outputs = self.combine_fn(outputs, input, name=self.name)
        if self.act:
            outputs = self.act(outputs)
        return outputs
    
class BatchNorm(Layer):

    def __init__(
        self,
        decay=0.9,
        epsilon=0.00001,
        act=None,
        is_train=False,
        beta_init=Zeros(),
        gamma_init=RandomNormal(mean=1.0, stddev=0.002),
        moving_mean_init=Zeros(),
        moving_var_init=Zeros(),
        num_features=None,
        data_format='channels_last',
        name=None,
    ):
        super(BatchNorm, self).__init__(name=name, act=act)
        self.decay = decay
        self.epsilon = epsilon
        self.data_format = data_format
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.moving_mean_init = moving_mean_init
        self.moving_var_init = moving_var_init
        self.num_features = num_features

        self.channel_axis = -1 if data_format == 'channels_last' else 1
        self.axes = None

        if num_features is not None:
            self.build(None)
            self._built = True

        if self.decay < 0.0 or 1.0 < self.decay:
            raise ValueError("decay should be between 0 to 1")

        logging.info(
            "BatchNorm %s: decay: %f epsilon: %f act: %s is_train: %s" %
            (self.name, decay, epsilon, self.act.__name__ if self.act is not None else 'No Activation', is_train)
        )

    def __repr__(self):
        actstr = self.act.__name__ if self.act is not None else 'No Activation'
        s = ('{classname}(num_features={num_features}, decay={decay}' ', epsilon={epsilon}')
        s += (', ' + actstr)
        if self.name is not None:
            s += ', name="{name}"'
        s += ')'
        return s.format(classname=self.__class__.__name__, **self.__dict__)

    def _get_param_shape(self, inputs_shape):
        if self.data_format == 'channels_last':
            axis = -1
        elif self.data_format == 'channels_first':
            axis = 1
        else:
            raise ValueError('data_format should be either %s or %s' % ('channels_last', 'channels_first'))

        channels = inputs_shape[axis]
        params_shape = [channels]

        return params_shape

    def _check_input_shape(self, inputs):
        if inputs.ndim <= 1:
            raise ValueError('expected input at least 2D, but got {}D input'.format(inputs.ndim))

    def build(self, inputs_shape):
        params_shape = [self.num_features] if self.num_features is not None else self._get_param_shape(inputs_shape)

        self.beta, self.gamma = None, None
        if self.beta_init:
            self.beta = self._get_weights("beta", shape=params_shape, init=self.beta_init)

        if self.gamma_init:
            self.gamma = self._get_weights("gamma", shape=params_shape, init=self.gamma_init)

        self.moving_mean = self._get_weights(
            "moving_mean", shape=params_shape, init=self.moving_mean_init, trainable=False
        )
        self.moving_var = self._get_weights(
            "moving_var", shape=params_shape, init=self.moving_var_init, trainable=False
        )

    def forward(self, inputs):
        self._check_input_shape(inputs)

        if self.axes is None:
            self.axes = [i for i in range(len(inputs.shape)) if i != self.channel_axis]

        mean, var = tf.nn.moments(inputs, self.axes, keepdims=False)
        if self.is_train:
            # update moving_mean and moving_var
            self.moving_mean = moving_averages.assign_moving_average(
                self.moving_mean, mean, self.decay, zero_debias=False
            )
            self.moving_var = moving_averages.assign_moving_average(self.moving_var, var, self.decay, zero_debias=False)
            outputs = batch_normalization(inputs, mean, var, self.beta, self.gamma, self.epsilon, self.data_format)
        else:
            outputs = batch_normalization(
                inputs, self.moving_mean, self.moving_var, self.beta, self.gamma, self.epsilon, self.data_format
            )
        if self.act:
            outputs = self.act(outputs)
        return outputs



class BatchNorm1d(BatchNorm):
    """The :class:`BatchNorm1d` applies Batch Normalization over 2D/3D input (a mini-batch of 1D
    inputs (optional) with additional channel dimension), of shape (N, C) or (N, L, C) or (N, C, L).
    See more details in :class:`BatchNorm`.

    Examples
    ---------
    With TensorLayer

    >>> # in static model, no need to specify num_features
    >>> net = tl.layers.Input([None, 50, 32], name='input')
    >>> net = tl.layers.BatchNorm1d()(net)
    >>> # in dynamic model, build by specifying num_features
    >>> conv = tl.layers.Conv1d(32, 5, 1, in_channels=3)
    >>> bn = tl.layers.BatchNorm1d(num_features=32)

    """

    def _check_input_shape(self, inputs):
        if inputs.ndim != 2 and inputs.ndim != 3:
            raise ValueError('expected input to be 2D or 3D, but got {}D input'.format(inputs.ndim))

class _InputLayer(Layer):
    """
    The :class:`Input` class is the starting layer of a neural network.

    Parameters
    ----------
    shape : tuple (int)
        Including batch size.
    dtype: dtype
        The type of input values. By default, tf.float32.
    name : None or str
        A unique layer name.

    """

    def __init__(self, shape, dtype=tf.float32, name=None):  #'input'):
        # super(InputLayer, self).__init__(prev_layer=inputs, name=name)
        super(_InputLayer, self).__init__(name)

        if isinstance(dtype, str):
            try:
                dtype = eval(dtype)
            except Exception as e:
                raise RuntimeError("%s is not a valid dtype for InputLayer." % (dtype))
        if not isinstance(dtype, tf.DType):
            raise RuntimeError("%s is not a valid dtype for InputLayer." % (dtype))

        logging.info("Input  %s: %s" % (self.name, str(shape)))
        self.shape = shape  # shape is needed in __repr__

        shape_without_none = [_ if _ is not None else 1 for _ in shape]
        # self.outputs = self.forward(tl.initializers.random_normal()(shape_without_none))
        outputs = self.forward(tl.initializers.ones()(shape_without_none, dtype=dtype))

        self._built = True

        self._add_node(outputs, outputs)

    def __repr__(self):
        s = 'Input(shape=%s' % str(self.shape)
        if self.name is not None:
            s += (', name=\'%s\'' % self.name)
        s += ')'
        return s

    def __call__(self, inputs, *args, **kwargs):
        return super(_InputLayer, self).__call__(inputs)

    def build(self, inputs_shape):
        pass

    def forward(self, inputs):
        return inputs


def Input(shape, dtype=tf.float32, name=None):
    """
    The :class:`Input` class is the starting layer of a neural network.

    Parameters
    ----------
    shape : tuple (int)
        Including batch size.
    name : None or str
        A unique layer name.

    """
    input_layer = _InputLayer(shape, dtype=dtype, name=name)
    outputs = input_layer._nodes[0].out_tensors[0]
    return outputs



def get_G(input_shape, name="generator"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = Input(input_shape)
    n = Conv2d(64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init)(nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
        nn = BatchNorm2d(act=tf.nn.relu, gamma_init=g_init)(nn)
        nn = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
        nn = BatchNorm2d(gamma_init=g_init)(nn)
        nn = Elementwise(tf.add)([n, nn])
        n = nn

    n = Conv2d(64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=g_init)(n)
    n = Elementwise(tf.add)([n, temp])
    # B residual blacks end

    n = Conv2d(256, (3, 3), (1, 1), padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    n = Conv2d(256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)(n)
    n = SubpixelConv2d(scale=2, n_out_channels=None, act=tf.nn.relu)(n)

    nn = Conv2d(3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init)(n)
    G = Model(inputs=nin, outputs=nn, name=name)
    return G

def get_D(input_shape, name="discriminator"):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    nin = Input(input_shape)
    n = Conv2d(df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init)(nin)

    n = Conv2d(df_dim * 2, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 4, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 32, (4, 4), (2, 2), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 16, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    nn = BatchNorm2d(gamma_init=gamma_init)(n)

    n = Conv2d(df_dim * 2, (1, 1), (1, 1), padding='SAME', W_init=w_init, b_init=None)(nn)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 2, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(act=lrelu, gamma_init=gamma_init)(n)
    n = Conv2d(df_dim * 8, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=None)(n)
    n = BatchNorm2d(gamma_init=gamma_init)(n)
    n = Elementwise(combine_fn=tf.add, act=lrelu)([n, nn])

    n = Flatten()(n)
    no = Dense(n_units=1, W_init=w_init)(n)
    D = Model(inputs=nin, outputs=no, name=name)
    return D
