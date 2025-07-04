from timm import scheduler
from timm import optim

from optim.scheduler import PolynomialLR, PolyLR


def create_scheduler(opt_args, optimizer):
    print(opt_args)
    if opt_args.sched == "polynomial":
        lr_scheduler = PolynomialLR(
            optimizer,
            opt_args.poly_step_size,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_lr,
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler


def create_optimizer(opt_args, model):
    return optim.create_optimizer(opt_args, model)
