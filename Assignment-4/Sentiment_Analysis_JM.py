#!/usr/bin/env python
# coding: utf-8

# ## CS6140_Machine_learning_Assignment_4 - Neural Networks
# 
# ### By Jaamie Maarsh
# 
# ### Assisted by Prof. Shanu 

# ### Installing necessary Libraries

# In[28]:


pip install numpy tensorflow


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc


# ### Data loading

# In[30]:


# Importing the IMDb dataset:
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences    
     


# ### Creating the Neural Network Model:
# 
# #### Input layer

# In[31]:


(vocab_size, max_length) = (20000, 200)
# dataset from tensorflow already contains preprocessed data 
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to ensure all reviews are of equal length
x_train = pad_sequences(x_train, maxlen=max_length, padding='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post')

# Display sample data for 
print("Sample review (tokenized):", x_train[1])
print("Sample label:", y_train[1])


# ##### Insights
# 
# The above code prepares the IMDB movie review dataset for training by loading and preprocessing it to create uniform input sequences:
# 
# Data Loading and Vocabulary Limit:
# 
# The dataset is preprocessed to include only the 20,000 most common words, keeping the vocabulary manageable and focused on relevant words.
# Padding for Uniform Length:
# 
# All reviews are padded or truncated to 200 words. This makes each review the same length, which is essential for model training.
# Tokenized Sample Review:
# 
# A sample review is shown as a sequence of integers, where each number represents a word. Shorter reviews are padded with zeros to reach the 200-word limit.
# 
# Overall, the preprocessing prepares the dataset for embedding layers or neural network models by standardizing review lengths and focusing on frequently used words.

# ### Neural Network Architecture

# In[32]:


embedding_size = 32

# Defining the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# ###### Insights and working
# 
# The above defines a simple neural network model with an embedding layer to process the IMDB dataset:
# 
# Embedding Layer:
# 
# Converts each word in the review into a 32-dimensional vector, enabling the model to understand word relationships. The input dimension is set to 20,000 (vocab size), meaning it learns embeddings for 20,000 unique words.
# Flattening Layer:
# 
# Flattens the 2D embeddings into a 1D vector, making the data compatible with the dense layers that follow.
# Dense Layers:
# 
# First dense layer has 128 units with ReLU activation for feature extraction.
# Final layer has 1 unit with sigmoid activation to output probabilities for binary classification (positive or negative sentiment).
# Model Compilation:
# 
# Uses the Adam optimizer and binary cross-entropy loss, suitable for binary classification tasks.
# The model summary warning indicates that the input_length parameter in the embedding layer is deprecated.

# ### Model Training

# In[33]:


# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ##### Insights
# 
# 
# Training Process:
# 
# The model trained over 5 epochs with a batch size of 32, achieving near-perfect training accuracy by the final epoch. However, as the training progressed, the model began to overfit, with validation loss increasing after the first epoch.
# Evaluation:
# 
# On the test set, the model achieved an accuracy of 84.72%, indicating it performs well on unseen data but shows signs of overfitting (high training accuracy vs. lower validation accuracy). This suggests the model may have learned patterns specific to the training data that do not generalize as well to new data.
# Potential Improvements:
# 
# Regularization techniques (e.g., dropout) or reducing model complexity could help improve generalization and prevent overfitting.
# Overall, while the model achieves good accuracy, adjusting it to handle overfitting could further enhance performance on the test set.

# In[34]:


# Get model predictions for the test set
y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype("int32")  # Apply threshold to get binary predictions

# Calculate F1 Score and ROC-AUC
f1_new = f1_score(y_test, y_pred)
roc_auc_new = roc_auc_score(y_test, y_pred_probs)

print(f"F1 Score: {f1_new:.2f}")
print(f"ROC-AUC: {roc_auc_new:.2f}")


# In[37]:


# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc_value = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random chance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()


# ###### Insights
# 
# F1 Score (0.85): The F1 score indicates a good balance between precision and recall, showing the model is effectively identifying positive and negative classes.
# 
# ROC-AUC (0.93): The high ROC-AUC score of 0.93 means the model has strong discriminatory power, effectively distinguishing between the two classes across various decision thresholds.
# 
# The metrics suggest the model is performing well with a high ability to correctly classify and separate positive and negative cases, though fine-tuning could still enhance results further.

# ### Hyperparametric tuning

# In[40]:


# Define hyperparameter ranges
embedding_sizes = [16, 64]
hidden_units_options = [64, 128, 256]

# Dictionaries to store results
test_accuracies = {}
history_dict = {}
f1_scores = {}
roc_aucs = {}

# Train, evaluate, and calculate additional metrics
for embedding_size in embedding_sizes:
    for hidden_units in hidden_units_options:
        print(f"\nTraining model with embedding_size={embedding_size} and hidden_units={hidden_units}")
        
        # Define the model
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length),
            Flatten(),
            Dense(hidden_units, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test), verbose=1)
        history_dict[(embedding_size, hidden_units)] = history.history
        
        # Evaluate the model and store test accuracy
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        test_accuracies[(embedding_size, hidden_units)] = test_accuracy
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        
        # Get predictions for F1 and ROC-AUC
        y_pred = (model.predict(x_test) > 0.5).astype("int32")
        
        # Calculate F1 Score and ROC-AUC
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict(x_test))
        
        # Store F1 and ROC-AUC scores
        f1_scores[(embedding_size, hidden_units)] = f1
        roc_aucs[(embedding_size, hidden_units)] = roc_auc
        print(f"F1 Score: {f1:.2f}, ROC-AUC: {roc_auc:.2f}")


# In[41]:


# Plotting test accuracies for each combination
plt.figure(figsize=(10, 6))
labels = [f"Emb {emb}, Units {units}" for emb in embedding_sizes for units in hidden_units_options]
values = [test_accuracies[(emb, units)] * 100 for emb in embedding_sizes for units in hidden_units_options]

plt.bar(labels, values, color='skyblue')
plt.xlabel("Embedding Size and Hidden Units")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy for Different Embedding Sizes and Hidden Units")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot training and validation accuracy and loss for each combination
fig, axes = plt.subplots(len(embedding_sizes), len(hidden_units_options), figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Training and Validation Accuracy & Loss for Different Hyperparameters')

for i, embedding_size in enumerate(embedding_sizes):
    for j, hidden_units in enumerate(hidden_units_options):
        ax = axes[i, j]
        history = history_dict[(embedding_size, hidden_units)]
        
        # Plot accuracy
        ax.plot(history['accuracy'], label='Train Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        
        # Plot loss on secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(history['loss'], label='Train Loss', linestyle='--', color='tab:red')
        ax2.plot(history['val_loss'], label='Validation Loss', linestyle='--', color='tab:orange')
        
        # Title and labels
        ax.set_title(f'Emb Size: {embedding_size}, Units: {hidden_units}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax2.set_ylabel('Loss')
        
        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# ###### Insights
# 
# All different combinations are compared and below are its results
# 
# Embedding Size and Hidden Units: Models trained with larger embeddings (64 vs. 16) and more hidden units generally achieved better performance, especially on the test accuracy, F1, and ROC-AUC scores.
# 
# Best Performance:
# 
# The configuration with embedding_size=64 and hidden_units=128 performed the best, achieving Test Accuracy of 84.98%, F1 Score of 0.85, and ROC-AUC of 0.93. This setup balanced accuracy with generalization effectively.
# Overfitting Concerns:
# 
# Across configurations, models showed high training accuracy (reaching nearly 100%), but this did not translate fully to validation accuracy, which dropped in later epochs. This indicates a degree of overfitting, suggesting the need for additional regularization, increased dropout, or early stopping.
# Training Time:
# 
# Models with higher hidden units and embedding sizes required more time per epoch. The most complex configuration (embedding_size=64, hidden_units=256) had slightly lower validation accuracy (83.84%) despite longer training times, indicating diminishing returns with excessive complexity.
# Summary: The model with embedding_size=64 and hidden_units=128 offers the best balance between performance and complexity, achieving high accuracy and robustness with minimal overfitting signs.

# In[42]:


# Set up the seaborn style for the plots
sns.set(style="whitegrid")

# Plot ROC curve for each embedding size and hidden units combination
plt.figure(figsize=(12, 8))

for i, (embedding_size, hidden_units) in enumerate(f1_scores.keys(), 1):
    # Get model predictions as probabilities
    y_pred_probs = model.predict(x_test)
    
    # Compute ROC curve and AUC score
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    roc_auc_value = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'Embedding {embedding_size}, Hidden Units {hidden_units} (AUC = {roc_auc_value:.2f})')

# Plot the diagonal line for reference
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# Formatting the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Embedding Sizes and Hidden Units')
plt.legend(loc='lower right')
plt.show()


# ##### Conclusion
# 
# the model evaluation summary are as follows:
# 
# Consistent ROC-AUC Scores: All configurations reached a high ROC-AUC of approximately 0.92, suggesting that each model effectively distinguishes between positive and negative sentiment, regardless of embedding size or hidden units. The slight visual overlap among curves in the ROC plot supports that performance differences between configurations are minimal in terms of AUC.
# 
# Best Configuration Identified: While all configurations performed similarly in terms of AUC, the embedding_size=64 and hidden_units=128 setup provided the best balance of accuracy and generalization based on test accuracy and F1 score.
# 
# Overfitting and Model Complexity: Increasing hidden units beyond 128 did not yield better validation accuracy or AUC, suggesting that larger models may suffer from diminishing returns and risk overfitting. Regularization or simpler architectures might be beneficial to control overfitting.
# 
# Potential for Fine-Tuning: The model’s overfitting signs (high training accuracy, lower validation accuracy) indicate that fine-tuning through dropout or early stopping could enhance generalization. This is essential if the model needs to perform consistently across different datasets.
# 
# ###### Conclusion
# The embedding_size=64 and hidden_units=128 configuration is recommended as it combines solid performance metrics (accuracy, F1, ROC-AUC) with efficient training times, indicating it is a robust choice for sentiment classification on this dataset. Further adjustments to control overfitting could improve the model’s applicability to unseen data.
