def print_model_parameters(model):
    total_params = 0
    print("{:40} {:>20}".format("Layer Name", "Param Count"))
    print("=" * 60)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print("{:40} {:20,}".format(name, param_count))

    print("=" * 60)
    print("{:40} {:20,}".format("Total Parameters", total_params))