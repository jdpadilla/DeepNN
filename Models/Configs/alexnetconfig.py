import types

layerargs = types.SimpleNamespace()
convargs = types.SimpleNamespace()
normargs = types.SimpleNamespace()
poolargs = types.SimpleNamespace()
fcargs = types.SimpleNamespace()


def alexnetargs1(iweight, ibias):

    layerargs = dellayerargs()

    convargs.filter_width = 11
    convargs.filter_height = 11
    convargs.num_filters = 96
    convargs.stride_y = 4
    convargs.stride_x = 4
    convargs.padding = 'VALID'
    convargs.actv = 'relu'
    convargs.groups = 1
    convargs.iweight = iweight
    convargs.ibias = ibias

    layerargs.convargs = convargs

    poolargs.filter_height = 3
    poolargs.filter_width = 3
    poolargs.stride_y = 2
    poolargs.stride_x = 2
    poolargs.padding = 'VALID'

    layerargs.poolargs = poolargs

    normargs.radius = 2
    normargs.alpha = 2e-05
    normargs.beta = 0.75
    normargs.bias = 1.0

    layerargs.normargs = normargs

    return layerargs


def alexnetargs2(iweight, ibias):

    layerargs = dellayerargs()

    convargs.filter_width = 5
    convargs.filter_height = 5
    convargs.num_filters = 256
    convargs.stride_y = 1
    convargs.stride_x = 1
    convargs.padding = 'SAME'
    convargs.actv = 'relu'
    convargs.groups = 2
    convargs.iweight = iweight
    convargs.ibias = ibias

    layerargs.convargs = convargs

    poolargs.filter_height = 3
    poolargs.filter_width = 3
    poolargs.stride_y = 2
    poolargs.stride_x = 2
    poolargs.padding = 'VALID'

    layerargs.poolargs = poolargs

    normargs.radius = 2
    normargs.alpha = 2e-05
    normargs.beta = 0.75
    normargs.bias = 1.0

    layerargs.normargs = normargs

    return layerargs


def alexnetargs3(iweight, ibias):

    layerargs = dellayerargs()

    convargs.filter_width = 3
    convargs.filter_height = 3
    convargs.num_filters = 384
    convargs.stride_y = 1
    convargs.stride_x = 1
    convargs.padding = 'SAME'
    convargs.actv = 'relu'
    convargs.groups = 1
    convargs.iweight = iweight
    convargs.ibias = ibias

    layerargs.convargs = convargs

    return layerargs


def alexnetargs4(iweight, ibias):

    layerargs = dellayerargs()

    convargs.filter_width = 3
    convargs.filter_height = 3
    convargs.num_filters = 384
    convargs.stride_y = 1
    convargs.stride_x = 1
    convargs.padding = 'SAME'
    convargs.actv = 'relu'
    convargs.groups = 2
    convargs.iweight = iweight
    convargs.ibias = ibias

    layerargs.convargs = convargs

    return layerargs


def alexnetargs5(iweight, ibias):

    layerargs = dellayerargs()

    convargs.filter_width = 3
    convargs.filter_height = 3
    convargs.num_filters = 256
    convargs.stride_y = 1
    convargs.stride_x = 1
    convargs.padding = 'SAME'
    convargs.actv = 'relu'
    convargs.groups = 2
    convargs.iweight = iweight
    convargs.ibias = ibias

    layerargs.convargs = convargs

    poolargs.filter_height = 3
    poolargs.filter_width = 3
    poolargs.stride_y = 2
    poolargs.stride_x = 2
    poolargs.padding = 'VALID'

    layerargs.poolargs = poolargs

    return layerargs


def alexnetargs6(iweight, ibias):

    layerargs = dellayerargs()

    fcargs.num_in = 6*6*256
    fcargs.num_out = 4096
    fcargs.actv = 'relu'
    fcargs.iweight = iweight
    fcargs.ibias = ibias

    layerargs.fcargs = fcargs

    return layerargs


def alexnetargs7(iweight, ibias):

    layerargs = dellayerargs()

    fcargs.num_in = 4096
    fcargs.num_out = 4096
    fcargs.actv = 'relu'
    fcargs.iweight = iweight
    fcargs.ibias = ibias

    layerargs.fcargs = fcargs

    return layerargs


def alexnetargs8(nclass, iweight, ibias):

    layerargs = dellayerargs()

    fcargs.num_in = 4096
    fcargs.num_out = nclass
    fcargs.actv = ''
    fcargs.iweight = iweight
    fcargs.ibias = ibias

    layerargs.fcargs = fcargs

    return layerargs


def dellayerargs():

    if hasattr(layerargs, 'convargs'):
        del layerargs.convargs
    if hasattr(layerargs, 'poolargs'):
        del layerargs.poolargs
    if hasattr(layerargs, 'normargs'):
        del layerargs.normargs
    if hasattr(layerargs, 'fcargs'):
        del layerargs.fcargs

    return layerargs
