import json
import matplotlib.pyplot as plt

# Load your JSON data
with open('allName.json', 'r') as f:
    data = json.load(f)

# Define color scheme and styling
colors = {
    'RL1e-6': '#1f77b4',  # Blue
    'RL2e-7': '#2ca02c',  # Green
    'RL5e-7': '#d62728'   # Red
}

# Assign unique line styles to each category
line_styles = {
    'noneName': 'dashed',
    'markName': ':',
    'withName': '-'
}

# Assign unique line widths to each category for better distinction
line_widths = {
    'noneName': 3.0,
    'markName': 2.0,
    'withName': 1.5
}

def create_plots(metric):
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{metric.capitalize()} Comparison', fontsize=16)
    
    # Define plot configurations
    configs = [
        ('Conversation', 'training'),
        ('Conversation', 'validation'),
        ('Question', 'training'),
        ('Question', 'validation')
    ]
    
    for idx, (data_type, phase) in enumerate(configs):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]
        
        # Plot each configuration
        for name_group in ['noneName', 'markName', 'withName']:
            group_data = data[data_type][name_group][0]
            for rl_param in group_data:
                values = group_data[rl_param][phase][metric]
                epochs = range(1, len(values) + 1)
                
                ax.plot(epochs, values,
                        color=colors[rl_param],
                        linestyle=line_styles[name_group],
                        linewidth=line_widths[name_group],
                        label=f'{rl_param} ({name_group})')

        ax.set_title(f'{data_type} - {phase.capitalize()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'all_{metric}_comparison.png', bbox_inches='tight')
    plt.close()

# Generate both plots
create_plots('loss')
create_plots('accuracy')
