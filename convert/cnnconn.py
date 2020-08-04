# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 10:11:46 2016

@author: darwin
"""

import numpy as np

def conv2d_connections(input_shape, filters, pads, strides, delay):
    """
    Extract connections to a convolutional layer.
    
    Parameters
    ----------
    input_shape : sequence of int
        Shape of the input tensor, i.e., the height, width and number of
        channels of the input tensor.
    filters : numpy.ndarray of shape (H,W,C,M)
        Convolution filters, with height H, width W, number of channels C and
        number of output feature maps M.
    pads : sequence of int
        Padding sizes on each side of every input channel, i.e.:
        0: Padding size on both the top and the bottom.
        1: Padding size on both the left and the right.
    strides : sequence of int
        Strides in the spatial directions, i.e., the stride in the vertical
        direction and the stride in the horizontal direction.
    delay : real number
        Common delay for all connections to the convolutional layer.
    
    Returns
    -------
    connections : numpy.ndarray of shape (N,4)
        N connections between the neurons corresponding to the input tensor and
        the neurons corresponding to the output feature maps. Each connection
        consists of:
        0: Index of the source neuron within its population.
        1: Index of the destination neuron within its population.
        2: Connection weight.
        3: Connection delay.
    
    Notes
    -----
    The activation function of the original convolutional layer should be ReLU.
    
    Bias of the original convolutional layer should be fixed to zero.
    
    Both the input tensor and the output feature maps are assumed to be HxWxC
    tensors, with H, W and C being the height, width and number of channels,
    respectively. The position (y,x,c) within the tensor is mapped to the neuron
    with index yWC+xC+c.
    The convolution filters are assumed to be a H'xW'xCxM tensor, with H', W'
    and M being the height, width and number of output feature maps,
    respectively.
    
    The initial spatial position of filters is determined by aligning the top
    left corner of the filters to the top left corner of the padded input
    channels. During a convolution or pooling, filters are entirely inside the
    boundaries of the padded input tensor.
    
    During convolution, a series of spatial positions of convolution filters are
    generated. For each of the spatial positions, the convolution filters are
    assumed to be centered at the spatial position of the input tensor.
    """
    connections = []
    # Find all spatial positions for centers of convolution filters.
    # TODO: Your code
    # Consider using _spatial_positions.
    num_vertical_positions,num_horizontal_positions,spatial_positions = \
                _spatial_positions((input_shape[0],input_shape[1]),\
                                   (filters.shape[0],filters.shape[1]),\
                                   pads,strides)
    
    # For each neuron, find pre-synaptic neurons and spatial region within the
    # convolution filters that involve in computation of the neuron.
    # TODO: Your code
    # Consider using _ReceptiveField.
    
    post_synaptic_neuron_index = 0
    filter_height = filters.shape[0]
    filter_width = filters.shape[1]
    filter_channel = filters.shape[2]
    ReceptiveField = _ReceptiveField(input_shape,(filter_height,filter_width,filter_channel))
    receptive_field = ReceptiveField.connections(spatial_positions)
    for pre_synaptic_neuron_indexes,filter_region in receptive_field:
        for m in range(filters.shape[3]):
            weight = filters[filter_region[0]:filter_region[0]+filter_region[1],\
                         filter_region[2]:filter_region[2]+filter_region[3],\
                         :,m]
            weight_reshaped = weight.reshape(-1)
            for n in range(len(pre_synaptic_neuron_indexes)):
                connection = (pre_synaptic_neuron_indexes[n],\
                              post_synaptic_neuron_index,\
                              weight_reshaped[n],delay)
                connections.append(connection)
            post_synaptic_neuron_index += 1
    return connections                       
                        
    # Construct connections.
    # TODO: Your code
    # You may find numpy.cumsum useful.
    
    
    # Determine the number of connections to the convolutional layer.
    # TODO: Your code
    
    # Determine the start and end index of each connection group.
    # TODO: Your code
    
    # Build connections.
    # TODO: Your code
    # You may use numpy.empty to allocate the matrix for storing connection
    # tuples and then fill in the matrix.
    # You may find numpy.tile and numpy.repeat useful.
    

def avg_pool_connections( \
    input_shape, \
    filters_spatial_shape, \
    pads, \
    strides, \
    delay, \
    weight_scaling_factor = 1.0):
    """
    Extract connections to an average pooling layer.
    
    Parameters
    ----------
    input_shape : sequence of int
        Shape of the input tensor, i.e., the height, width and number of
        channels of the input tensor.
    filters_spatial_shape : sequence of int
        Shape of filters in spatial directions, i.e., height and width of
        filters.
    pads : sequence of int
        Padding sizes on each side of every input channel, i.e.:
        0: Padding size on both the top and the bottom.
        1: Padding size on both the left and the right.
    strides : sequence of int
        Strides in the spatial directions, i.e., the stride in the vertical
        direction and the stride in the horizontal direction.
    delay : real number
        Common delay for all connections to the average pooling layer.
    weight_scaling_factor : real number, optional
        Common scaling factor to be applied to weights of connections to the
        average pooling layer.
    
    Returns
    -------
    connections : numpy.ndarray of shape (N,4)
        N connections between the neurons corresponding to the input tensor and
        the neurons corresponding to the output feature maps. Each connection
        consists of:
        0: Index of the source neuron within its population.
        1: Index of the destination neuron within its population.
        2: Connection weight.
        3: Connection delay.
    
    Notes
    -----
    Both the input tensor and the output feature maps are assumed to be HxWxC
    tensors, with H, W and C being the height, width and number of channels,
    respectively. The position (y,x,c) within the tensor is mapped to the neuron
    with index yWC+xC+c.
    
    The initial spatial position of filters is determined by aligning the top
    left corner of the filters to the top left corner of the padded input
    channels. During a convolution or pooling, filters are entirely inside the
    boundaries of the padded input tensor.
    
    During pooling, a series of spatial positions of filters are generated. For
    each of the spatial positions, the filters are assumed to be centered at the
    spatial position of the input tensor.
    
    This function multiplies weights of connections to the average pooling layer
    by *weight_scaling_factor* before returning the connections.
    """
    connections = []
    # Find all spatial positions for centers of filters.
    # TODO: Your code
    # Consider using _spatial_positions.
    num_vertical_positions,num_horizontal_positions,positions = \
                _spatial_positions([input_shape[0],input_shape[1]],\
                                   filters_spatial_shape,\
                                   pads,strides)
    # For each neuron, find pre-synaptic neurons and spatial region within the
    # filters that involve in computation of the neuron.
    # TODO: Your code
    # Consider using _ReceptiveField.
    filter_height = filters_spatial_shape[0]
    filter_width = filters_spatial_shape[1]
    filter_channel = input_shape[2]
    ReceptiveField = _ReceptiveField(input_shape,\
                                    (filter_height,filter_width,filter_channel))
    receptive_field = ReceptiveField.connections(positions)
    post_synaptic_neuron_index = 0
    for pre_synaptic_neuron_index,filter_region in receptive_field:
        pre_synaptic_neuron_index_reshape = \
                        pre_synaptic_neuron_index.reshape(-1,input_shape[2]).T
        for m in range(input_shape[2]):
            for index in pre_synaptic_neuron_index_reshape[m]:
                connection = (index,post_synaptic_neuron_index,\
                                weight_scaling_factor,delay)
                connections.append(connection)
            post_synaptic_neuron_index += 1
    return connections
                
            
    # Construct connections.
    # TODO: Your code
    # You may find numpy.cumsum useful.
    
    # Determine the number of connections to the pooling layer.
    # TODO: Your code
    
    # Determine the start and end index of each connection group.
    # TODO: Your code
    
    # Build connections.
    # TODO: Your code
    # You may use numpy.empty to allocate the matrix for storing connection
    # tuples and then fill in the matrix.
    # You may find numpy.tile and numpy.repeat useful.
    

def fc_connections(weights, delay):
    """
    Extract connections to a fully-connected layer.
    
    Parameters
    ----------
    weights : numpy.ndarray of shape (M,N)
        Weight matrix, with M and N being the dimensionality of each input
        vector and the number of units of the fully-connected-layer,
        respectively.
    delay : real number
        Common delay for all connections to the fully-connected layer.
    
    Returns
    -------
    connections : numpy.ndarray of shape (N,4)
        N connections between the neurons corresponding to the input tensor and
        the neurons corresponding to the units of the fully-connected layer.
        Each connection consists of:
        0: Index of the source neuron within its population.
        1: Index of the destination neuron within its population.
        2: Connection weight.
        3: Connection delay.
    """
    # TODO: Your code
    #weights_ = np.array(weights)
    """
    connections = []
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            connection = (i,j,weights[i][j],delay)
            connections.append(connection)
    """
    num_input = weights.shape[0]
    num_output = weights.shape[1]
    num_connections = num_input * num_output
    input_range = np.arange(num_input)
    output_range = np.arange(num_output)
    output_indices, input_indices = np.meshgrid(output_range, input_range)
    connections = np.empty((num_connections, 4))
    connections[:, 0] = np.reshape(input_indices, (-1,))
    connections[:, 1] = np.reshape(output_indices, (-1,))
    connections[:, 2] = np.reshape(weights, (-1,))
    connections[:, 3] = delay
    return connections

def _spatial_positions(input_shape, filter_shape, pads, strides):
    """
    Calculate spatial positions to which the centers of filters are aligned
    during convolution or pooling.
    
    Parameters
    ----------
    input_shape : sequence of int
        Shape of the input tensor in the spatial directions, i.e., height and
        width of the input tensor.
    filter_shape : sequence of int
        Shape of the filters in the spatial directions, i.e., height and width
        of the filters.
    pads : sequence of int
        Padding sizes on each side of every input channel, i.e.:
        0: Padding size on both the top and the bottom.
        1: Padding size on both the left and the right.
    strides : sequence of int
        Strides in the spatial directions, i.e., the stride in the vertical
        direction and the stride in the horizontal direction.
    
    Returns
    -------
    num_vertical_positions : int
        Number of possible vertical positions.
    num_horizontal_positions : int
        Number of possible horizontal positions.
    positions : numpy.ndarray of shape (N,2)
        The spatial positions. Each row corresponds to one spatial position,
        which consists of:
        0: Position in the vertical direction.
        1: Position in the horizontal direction.
    
    Notes
    -----
    The initial spatial position of filters is determined by aligning the top
    left corner of the filters to the top left corner of the padded input
    channels.
    
    During a convolution or pooling, filters are entirely inside the boundaries
    of the padded input tensor.
    """
    # Determine the initial position and the largest position in each direction,
    # and calculate all possible positions within the range.
    # TODO: Your code
    # Consider writing common code to deal with either the horizontal or the
    # vertical directions, and then call the code on both directions.
    # You may find numpy.arange useful.
    input_shape_ = np.reshape(input_shape, (-1,)).astype(np.int_)
    filter_shape_ = np.reshape(filter_shape, (-1,)).astype(np.int_)
    pads_ = np.reshape(pads, (-1,)).astype(np.int_)
    strides_ = np.reshape(strides, (-1,)).astype(np.int_)
    def num_positions(input_size,filter_size,padding,stride):
        start_position = -padding + filter_size // 2
        stop_position = input_size + padding - filter_size + filter_size // 2
        positions = np.arange(start_position,stop_position+1,stride)

        return positions
    
    #vertical

    vertical_positions = \
                    num_positions(input_shape_[0],filter_shape_[0],pads_[0],strides_[0])
    num_vertical_positions = len(vertical_positions)
    
    #horizontal
    horizontal_positions = \
                    num_positions(input_shape_[1],filter_shape_[1],pads_[1],strides_[1])
    num_horizontal_positions = len(horizontal_positions)
                
    # Merge possible positions on both spatial directions.
    # TODO: Your code
    # You may find numpy.meshgrid, numpy.reshape and numpy.hstack useful.
    [H,V] = np.meshgrid(horizontal_positions,vertical_positions)
    _V = V.reshape(-1,1)
    _H = H.reshape(-1,1)
    positions = np.hstack((_V,_H))
    
    return num_vertical_positions,num_horizontal_positions,positions
    

class _ReceptiveField:
    """
    Receptive field of neurons.
    
    Parameters
    ----------
    input_shape : sequence of int
        Shape of the input tensor, i.e., its height, width and number of
        channels.
    filter_shape : sequence of int
        Shape of convolution filters, i.e., its height, width and number of
        channels.
    """
    def __init__(self, input_shape, filter_shape):
        self._input_h = int(input_shape[0])
        self._input_w = int(input_shape[1])
        self._c = int(input_shape[2])
        if self._input_h <= 0 or self._input_w <= 0 or self._c <= 0:
            raise ValueError( \
                'The height, width and number of channels of the input tensor ' \
                'must be positive integers.')
        self._filter_h = int(filter_shape[0])
        self._filter_w = int(filter_shape[1])
        if self._filter_h <= 0 or self._filter_w <= 0:
            raise ValueError( \
                'The height, width and number of channels of convolution filters ' \
                'must be positive integers.')
        if int(filter_shape[2]) != self._c:
            raise ValueError( \
                'Convolution filters must have the same number of channels as ' \
                'the input tensor.')
    
    def connections(self, spatial_positions):
        """
        For each of the given spatial positions on output feature maps, find
        pre-synaptic neurons that constitute the receptive field of the neurons
        at the spatial position, along with the weights of the connections from
        the pre-synaptic neurons.
        
        Parameters
        ----------
        spatial_positions : numpy.ndarray of integers of shape (N,2)
            N spatial positions on input channels. Each row corresponds to one
            of the spatial positions, which consists of positions in the
            vertical and horizontal directions, respectively.
        
        Returns
        -------
        receptive_fields : sequence of 2-tuples
            Each of the N 2-tuples corresponds to one of the spatial positions.
            The 2-tuples are arranged in the same order as the given spatial
            positions.
            Each of the 2-tuples consists of:
            0: A sequence of the indices of the pre-synaptic neurons, which are
            stored in a numpy.ndarray object of shape (-1,).
            1: The region of convolution filters that serves as weights of the
               connections, which is stored in a sequence of 4 integers:
               0: Top of the region.
               1: Height of the region.
               2: Left of the region.
               3: Width of the region.
               Note that weights of the connections from the pre-synaptic
               neurons are to be extracted from the same region of all filters.
        
        Notes
        -----
        For each of the given spatial positions, the filters are assumed to be
        centered at the spatial position of the input tensor.
        
        Both the input tensor and the convolution filters are assumed to be in
        the HWC format. For each returned 2-tuple, the region of each of the
        convolution filters can be extracted as a order 3 tensor. If the
        extracted tensor is reshaped into a vector, each element in the vector
        is the weight of the connection from the pre-synaptic neuron, whose
        index is stored at the same position of the returned sequence of neuron
        indices.
        """
        spatial_positions_ = \
            np.array(spatial_positions, ndmin = 2).astype(np.int_)
        if len(spatial_positions_.shape) != 2 or \
            spatial_positions_.shape[1] != 2:
            raise ValueError( \
                'The spatial positions must be a numpy.ndarray object of shape ' \
                '(N,2).')
        # For each spatial position ...
        # Determine the spatial range in the input tensor and the spatial range
        # in the filters, which are involved in the computation of the output
        # features.
        # TODO: Your code. Call self._spatial_ranges to obtain the bounding
        # boxes.
        bounding_boxes = self._spatial_ranges(spatial_positions)
    
        # Convert the spatial range of the input tensor to neuron indices, and
        # convert the spatial range of the convolution filters to positions.
        # TODO: Your code.
        # A _PositionToNeuronIndex is needed to to convert spatial positions
        # of input feature maps to neuron indices into the input population.
        # You may find numpy.arange, numpy.tile, numpy.repeat, numpy.meshgrid,
        # numpy.reshape, and numpy.hstack useful.
        position_to_neuron_index = _PositionToNeuronIndex(\
                            (self._input_h,self._input_w,self._c))
        def connection_region(bounding_box,channel):
            input_top = bounding_box[0]
            input_left = bounding_box[1]
            filter_top = bounding_box[2]
            filter_left = bounding_box[3]
            height = bounding_box[4]
            width = bounding_box[5]
            y = np.arange(input_top,input_top + height)
            x = np.arange(input_left,input_left + width)
            c = np.arange(channel)
            [_x,_y,_c] = np.meshgrid(x,y,c)
            Y = _y.reshape(-1,1)
            X = _x.reshape(-1,1)
            C = _c.reshape(-1,1)
            positions = np.hstack((Y,X,C))
            neuron_index = position_to_neuron_index.neuron_indices(positions)
            filter_region = [filter_top,height,filter_left,width]
            receptive_field = (neuron_index,filter_region)
            return receptive_field
        channel = np.repeat(self._c,bounding_boxes.shape[0])
        receptive_fields = map(connection_region,bounding_boxes,channel)
        return receptive_fields
        
        
        
    
    def _spatial_ranges(self, spatial_positions):
        """
        For each of the given spatial positions, find the range of spatial
        positions in the input tensor and the range of spatial positions in the
        filters, which are involved in computation of the output features.
        
        Parameters
        ----------
        spatial_positions : numpy.ndarray of integers of shape (N,2)
            N spatial positions. Each row corresponds to one of the spatial
            positions, which consists of positions in the vertical and
            horizontal directions.
        
        Returns
        -------
        bounding_boxes: numpy.ndarray of shape (N,6)
            Each row corresponds to a spatial position, with elements as listed
            below:
            0: Top of the bounding box in the input tensor.
            1: Left of the bounding box in the input tensor.
            2: Top of the bounding box in the filters.
            3: Left of the bounding box in the filters.
            4: Height of the two bounding boxes.
            5: Width of the two bounding boxes.
        
        Notes
        -----
        For each of the given spatial positions, the filters are assumed to be
        centered at the spatial position of the input tensor.
        """
        def dim_range( \
            input_size, \
            filter_size, \
            positions, \
            input_lower_bounds, \
            filter_lower_bounds, \
            overlap_sizes):
            """
            For each center position, determine the range in the input and the
            range in the filter. Within the ranges, the elements participate
            in the calculation of the inner product.
            
            Parameters
            ----------
            input_size : int
                Size of the input along the spatial dimension.
            filter_size : int
                Size of the filter along the spatial dimension.
            positions : 1-D numpy.ndarray of integers
                Positions along the spatial dimension within the input, which
                the center of the filter is aligned with.
            input_lower_bounds : 1-D numpy.ndarray of integers
                input_lower_bounds[i] is the lower bound of positions, i.e.,
                left or top, within the input that is covered by the filter
                centered at positions[i].
            filter_lower_bounds : 1-D numpy.ndarray of integers
                filter_lower_bounds[i] is the lower bound of positions, i.e.,
                left or top, within the filter centered at positions[i].
            overlap_sizes : 1-D numpy.ndarray of integers
                overlap_sizes[i] is the size, i.e., width or height, of the
                overlapping region of the input and the filter centered at
                positions[i].
            
            Notes
            -----
            This function works with a single spatial dimension.
            Note that *input_lower_bounds*, *filter_lower_bounds*, and
            *overlap_sizes* are outputs of this function.
            """
            # If the entire filter is within the bounds of the input ...
            # TODO: Your code
            input_lower_bounds[:] = positions - filter_size//2
            input_upper_bounds = input_lower_bounds + filter_size -1
            filter_lower_bounds[:] = 0
            overlap_sizes[:] = filter_size            
            # If part of the filter is beyond the lower bound of the input ...
            # TODO: Your code
            beyond_lower = (input_lower_bounds < 0)
            beyond_lower_size = -input_lower_bounds[beyond_lower]
            input_lower_bounds[beyond_lower] = 0
            filter_lower_bounds[beyond_lower] += beyond_lower_size
            overlap_sizes[beyond_lower] -= beyond_lower_size
            
            # If part of the filter is beyond the upper bound of the input ...
            # TODO: Your code
            beyond_upper = (input_upper_bounds >= input_size)
            beyond_upper_size = \
                input_upper_bounds[beyond_upper] - input_size + 1
            overlap_sizes[beyond_upper] -= beyond_upper_size
            
            
        bounding_boxes = \
            np.empty((spatial_positions.shape[0], 6), dtype = np.int_)
        # Vertical
        dim_range( \
            self._input_h, \
            self._filter_h, \
            spatial_positions[:, 0], \
            bounding_boxes[:, 0], \
            bounding_boxes[:, 2], \
            bounding_boxes[:, 4])
        # Horizontal
        dim_range( \
            self._input_w, \
            self._filter_w, \
            spatial_positions[:, 1], \
            bounding_boxes[:, 1], \
            bounding_boxes[:, 3], \
            bounding_boxes[:, 5])
        return bounding_boxes

class _PositionToNeuronIndex:
    """
    Conversion from positions within a tensor to neuron indices within a neuron
    population.
    
    Parameters
    ----------
    shape : sequence of int
        Height, width and number of channels of the tensor.
    """
    def __init__(self, shape):
        self._h = int(shape[0])
        self._w = int(shape[1])
        self._c = int(shape[2])
        if self._h <= 0 or self._w <= 0 or self._c <= 0:
            raise ValueError( \
                'The height, width, and number of channels must be positive ' \
                'integers.')
    
    def neuron_indices(self, positions):
        """
        Convert the positions within the tensor to neuron indices within the
        neuron population.
        
        Parameters
        ----------
        positions : numpy.ndarray of shape (N,3)
            N positions in the tensor. Each row corresponds to a position, which
            consists of indices in the vertical, horizontal and channel
            directions.
        
        Returns
        -------
        i : numpy.ndarray of shape (N,)
            Neuron indices.
        """
        # TODO: Your code
        positions_ = np.array(positions, ndmin = 2).astype(np.int_)
        W = self._w
        C = self._c
        i = W * C *positions_[:,0] + C * positions_[:,1] + positions_[:,2]
        return i
        

if __name__ == "__main__":
    pass
        
        
        
        
        
        
        
        
