
def middle_model_fp(model, concat_bottom):
    model.train()
    # middle_output = model(concat_bottom).cpu()
    middle_output = model(concat_bottom)
    return middle_output


def middle_model_bp(middle_grad, optimizer, middle_output, concat_bottom):
    concat_bottom.retain_grad()
    optimizer.zero_grad()
    # middle_output.backward(middle_grad.cpu())
    middle_output.backward(middle_grad)
    optimizer.step()
    bottom_grads = concat_bottom.grad

    return bottom_grads
