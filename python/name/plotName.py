import json
import matplotlib.pyplot as plt

# List of file prefixes for the three JSON files
file_prefixes = ["none", "mark", "with"]

# Create a figure with 3 rows and 2 columns, sharing the x-axis within each column
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 18), sharex='col')
fig.suptitle("Validation Metrics: loss (lef) and accuracy (right) for 200 epochs training", fontsize=16, y=0.98)

# Lists to accumulate all loss and accuracy values for common y-limits
all_loss_values = []
all_acc_values = []

# Loop over each JSON file corresponding to a row in the subplot grid
for i, prefix in enumerate(file_prefixes):
    # Load JSON data from file
    with open(f"{prefix}Name.json", "r") as f:
        data = json.load(f)
    
    # Extract validation data for Conversation and Question
    conv_data = data["Conversation"][0]
    quest_data = data["Question"][0]
    
    # Get the RL keys (assumed to be the same for Conversation and Question)
    rl_keys = list(conv_data.keys())
    
    # Define a colormap that gives a gradient from light to deep blue.
    colormap = plt.cm.get_cmap('Blues', len(rl_keys) + 1)
    
    # Get the subplots for the current JSON file (row)
    ax_loss = axes[i, 0]
    ax_acc = axes[i, 1]
    
    # Set the y-label for the left subplot only, with vertical text (rotated 90°)
    ax_loss.set_ylabel(f"{prefix} name", rotation=90, labelpad=10, fontsize=12)
    
    # Plot each RL key’s data with gradually deepening colors
    for idx, rl in enumerate(rl_keys):
        # Extract validation metrics for current RL key
        conv_loss = conv_data[rl]["validation"]["loss"]
        conv_acc = conv_data[rl]["validation"]["accuracy"]
        quest_loss = quest_data[rl]["validation"]["loss"]
        quest_acc = quest_data[rl]["validation"]["accuracy"]
        
        # Accumulate values to later set common y-axis limits
        all_loss_values.extend(conv_loss)
        all_loss_values.extend(quest_loss)
        all_acc_values.extend(conv_acc)
        all_acc_values.extend(quest_acc)
        
        # Create epoch indices based on the number of data points
        epochs_conv = list(range(len(conv_loss)))
        epochs_quest = list(range(len(quest_loss)))
        
        # Use the colormap to assign a color that deepens as idx increases
        color = colormap(idx + 1)
        
        # Plot Loss: dotted line for Conversation, solid line for Question
        ax_loss.plot(epochs_conv, conv_loss, linestyle=':', color=color, label=f"{rl} Conversation")
        ax_loss.plot(epochs_quest, quest_loss, linestyle='-', color=color, label=f"{rl} Question")
        
        # Plot Accuracy similarly on the right subplot
        ax_acc.plot(epochs_conv, conv_acc, linestyle=':', color=color, label=f"{rl} Conversation")
        ax_acc.plot(epochs_quest, quest_acc, linestyle='-', color=color, label=f"{rl} Question")
    
    # Only set the x-axis label on the bottom subplots to avoid duplication.
    if i == len(file_prefixes) - 1:
        ax_loss.set_xlabel("Epochs")
        ax_acc.set_xlabel("Epochs")

# Create one global legend and place it just below the overall title
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(handles)//2, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("combined_validation.png", dpi=300, bbox_inches='tight')
plt.show()
