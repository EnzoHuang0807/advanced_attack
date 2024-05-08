import torch
from tqdm import tqdm
import torch.nn as nn


def uap_spgd(model, loader, nb_epoch, eps, beta=9, step_decay=0.1, loss_function=None,
            batch_size = None, Momentum=0, uap_init=None, center_crop=32):
    '''
    INPUT
    model       model
    loader      dataloader
    nb_epoch    number of optimization epochs
    eps         maximum perturbation value (L-infinity) norm
    beta        clamping value
    step_decay  single step size
    loss_fn     custom loss function (default is CrossEntropyLoss)
    uap_init    custom perturbation to start from (default is random vector with pixel values {-eps, eps})
    center_crop image size
    Momentum    momentum item (default is false)
    
    log output
    batch_size  batch size 
    loader_eval evaluation dataloader
    ''' 
    model.eval()
    if uap_init is None:
        batch_delta = torch.zeros(batch_size,3,center_crop,center_crop)  # initialize as zero vector
    else:
        batch_delta = uap_init.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    delta = batch_delta[0]
    
    # loss function
    if loss_function:
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        beta = torch.cuda.FloatTensor([beta])

        def clamped_loss(output, target):
            loss = torch.mean(torch.min(loss_fn(output, target), beta))
            return loss


    batch_delta.requires_grad_()
    v = 0
    for epoch in tqdm(range(nb_epoch)):

        # perturbation step size with decay
        eps_step = eps * step_decay

        for i, data in enumerate(loader):
            x_val = data[0]
            with torch.no_grad():
                outputs_ori = model(x_val.cuda())
                _, target_label = torch.max(outputs_ori, 1)
            if i > 0 or epoch > 0:
                batch_delta.grad.data.zero_()

            batch_delta.data = delta.unsqueeze(0).repeat([x_val.shape[0], 1, 1, 1])

            perturbed = torch.clamp((x_val + batch_delta).cuda(), 0, 1)
            outputs = model(perturbed)
            # loss function value
            if loss_function:
                loss = clamped_loss(outputs, target_label.cuda())
            else:
                loss = -torch.mean(outputs.gather(1, (target_label.cuda()).unsqueeze(1)).squeeze(1))
            loss.backward()
            # batch update
            #momentum
            if Momentum:
                batch_delta_grad = batch_delta.grad.data.mean(dim=0)
                if torch.norm(batch_delta_grad,p=1) == 0:
                    batch_delta_grad = batch_delta_grad
                else:
                    batch_delta_grad = batch_delta_grad / torch.norm(batch_delta_grad, p=1)
                v = 0.9*v + batch_delta_grad
                grad_sign = v.sign()
            else:
                grad_sign = batch_delta.grad.data.mean(dim=0).sign()

            delta = delta + grad_sign * eps_step
            delta = torch.clamp(delta, -eps, eps)
            batch_delta.grad.data.zero_()

    return delta.data