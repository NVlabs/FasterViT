# Get learning rates as each training step
def draw_lr(args, optimizer, lr_scheduler, out_fname):
    learning_rates = []
    learning_rates_backbone = []
    learning_rates_linear_proj = []

    for i in range(args.epochs):
        optimizer.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
        learning_rates_backbone.append(optimizer.param_groups[1]["lr"])
        learning_rates_linear_proj.append(optimizer.param_groups[2]["lr"])
        lr_scheduler.step()
    
    draw(learning_rates, learning_rates_backbone, learning_rates_linear_proj, out_fname, EPOCHS=args.epochs, max_lr=max(max(learning_rates), max(learning_rates_backbone), max(learning_rates_linear_proj)))

def draw(overall_lr, backbone_lr, linear_proj_lr, out_fname, EPOCHS=None, max_lr=None, STEPS_IN_EPOCH=1):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        
        # add graph: learning_rates
        ax.plot(range(EPOCHS * STEPS_IN_EPOCH),
                overall_lr,
                marker='o',
                color='black',
                label='Overall Learning Rate')
        # add graph : learning_rates_backbone
        ax.plot(range(EPOCHS * STEPS_IN_EPOCH),
                backbone_lr,
                marker='s',
                color='blue',
                label='Backbone Learning Rate')

        # add graph: learning_rates_linear_proj
        ax.plot(range(EPOCHS * STEPS_IN_EPOCH),
                linear_proj_lr,
                marker='^',
                color='red',
                label='Linear Projection Learning Rate')
        ax.set_xlim([0, 30])
        ax.set_ylim([0, max_lr + 0.0001])
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(STEPS_IN_EPOCH))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.legend()  # add legend
        plt.savefig(out_fname)
