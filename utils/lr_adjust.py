import math


def linear_warmup_func(start_lr, step, end_iter, start_iter=0):
    if step < start_iter or step > end_iter:
        raise ValueError(f'step {step} is not in teh range [{start_iter}, {end_iter}]', flush=True)

    cur_lr = (step - start_iter)* 1.0 * start_lr / (end_iter - start_iter)
    return cur_lr

def linear_func(start_lr, step, end_iter, end_lr=0, start_iter=0):
    if step < start_iter or step > end_iter:
        raise ValueError(f'step {step} is not in teh range [{start_iter}, {end_iter}]', flush=True)

    cur_lr = start_lr + (step - start_iter)* 1.0 * (end_lr -  start_lr) / (end_iter - start_iter)
    return cur_lr

def cosine_decay_func(start_lr, step, end_iter, end_lr=0, start_iter=0):
    if step < start_iter or step > end_iter:
        raise ValueError(f'step {step} is not in teh range [{start_iter}, {end_iter}]', flush=True)
    cur_lr = end_lr + 0.5 * (start_lr - end_lr) * (1 + math.cos(math.pi * (step - start_iter) / float(end_iter - start_iter)))
    return cur_lr

def constant_decay_func(start_lr, step, end_iter, end_lr=0, start_iter=0):
    if step < start_iter or step > end_iter:
        raise ValueError(f'step {step} is not in teh range [{start_iter}, {end_iter}]', flush=True)
    return start_lr
"""
[
{"lr": 1e-4, "end_iter": 0,   "method": "constant"},
{"lr": 1e-4, "end_iter": 100, "method": "wamrup"},
{"lr": 2e-4, "end_iter": 200, "method": "wamrup"},

]
"""

class LRAdjuster(object):
    def __init__(self, 
                 lr_list,
                 optimizer=None,
                 gradient_accumulation_steps=1,
                 ):
        self.lr_list = lr_list
        self.optimizer = optimizer if optimizer else None
        self.prev_lr = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
    
    def get_last_lr(self):
        return [self.prev_lr]

    def step(self, step):
        start_lr = 0
        start_iter = 0
        next_lr = self.prev_lr
        for i,row in enumerate(self.lr_list):
            end_lr = float(row['lr'])
            end_iter = int(row['end_iter'])
            if step > end_iter:
                start_lr = float(row['lr'])
                start_iter = int(row['end_iter'])
                continue
            
            if 0 == i:
                next_lr = float(row['lr'])
            elif row['method'] == 'warmup':
                if i > 0:
                    raise ValueError(f'warmup shoulde not be used after stage 1', flush=True)
                next_lr = linear_warmup_func(start_lr, step, end_iter, start_iter)
            elif row['method'] == 'linear':
                next_lr = linear_func(start_lr, step, end_iter, end_lr, start_iter)
            elif row['method'] == 'cosine':
                next_lr = cosine_decay_func(start_lr, step, end_iter, end_lr=end_lr, start_iter=start_iter)
            elif row['method'] == 'constant':
                next_lr = end_lr
            break
        

        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = next_lr

        self.prev_lr = next_lr
        return next_lr
    

def draw_curve(x, y):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    assert len(x) == len(y), 'length of x and y must be equal!'
    plt.plot(x, y, color='r', label="lr by step", linewidth=1)
    plt.xlabel('steps')
    plt.ylabel('lr')
    plt.title('learning rate by step')
    plt.legend(loc='best')
    plt.savefig('tmp/lr_by_step.png')

def draw_lr(lr_res):
    x, y = zip(*lr_res)
    draw_curve(x, y)


if __name__ == '__main__':
    lr_list = [
        {"lr": 1e-8, "end_iter": 0,   "method": "constant"},
        {"lr": 1e-4, "end_iter": 100, "method": "linear"},
        {"lr": 1e-4, "end_iter": 200, "method": "constant"},
        {"lr": 2e-5, "end_iter": 300, "method": "cosine"},
        {"lr": 1e-7, "end_iter": 400, "method": "linear"},
    ]

    myLR = LRAdjuster(lr_list)

    lr_res = [(step, myLR.step(step)) for step in range(500)]

    draw_lr(lr_res=lr_res)
