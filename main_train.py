from utils import trainer

if __name__ == '__main__':
    trainer("osnet_ibn_x1_0_fusion", max_epoch=30)
    # trainer("densenet169", max_epoch=30)
    # trainer('osnet_ibn_x1_0', max_epoch=30)
    # trainer("osnet_ain_x1_0", max_epoch=30)
    # trainer("osnet_ain_x0_25", max_epoch=30)
    # trainer("resnet50_fusion")