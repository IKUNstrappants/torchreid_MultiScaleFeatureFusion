import torchreid

def get_datamanager(dataset="market1501"):
    return torchreid.data.ImageDataManager(
        root="reid-data",
        sources=dataset,
        targets=dataset,
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
        workers=0
    )
def get_model(name:str, datamanager:torchreid.data.ImageDataManager):
    model = torchreid.models.build_model(
        name=name,
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True
    ).cuda()
    return model

def get_optimizer(model, lr):
    return torchreid.optim.build_optimizer(
        model=model,
        optim="adam",
        lr=lr
    )

def get_scheduler(optimizer):
    return torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler="single_step",
        stepsize=20
    )

def get_engine(model_name:str, lr=0.0003):
    datamanager = get_datamanager()
    model = get_model(name=model_name, datamanager=datamanager)
    optimizer = get_optimizer(model=model, lr=lr)
    scheduler = get_scheduler(optimizer=optimizer)
    return torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

def trainer(model_name, save_dir = None, max_epoch = 30, lr=0.0003):
    engine = get_engine(model_name=model_name, lr=lr)
    if save_dir is None:
        save_dir = f"log/{model_name}-{max_epoch}epoch-lr={lr}"
    engine.run(
        save_dir=save_dir,
        max_epoch=max_epoch,
        eval_freq=10,
        print_freq=100,
        test_only=False
    )

if __name__ == '__main__':
    get_engine(model_name="resnet50").run(
        save_dir="log/resnet50",
        max_epoch=30,
        eval_freq=10,
        print_freq=100,
        test_only=False
    )