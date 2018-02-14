from DeepNN2_0.Models.buildalexnet import AlexNet

def build_this(x, args):
    if args.model == 'alexnet':
        return AlexNet(x, args)

    elif args.model == 'googlenet':
        pass
        # return GoogLeNet(args)

    elif args.model == 'resnet':
        pass
        # return ResNet(args)

    elif args.model == 'squeezenet':
        pass
        # return SqueezeNet(args)

    elif args.model == 'vgg':
        pass
        # return VGG(args)

    elif args.model == 'dcgan':
        pass
        # return DCGAN(args)

    elif args.model == 'custom':
        pass
        # model = Custom(args)
    else:
        return AlexNet(x, args)