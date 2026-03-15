

train_eval = history.evaluate(train_gen, steps=5)
print(f"Val Eval Accuracy: {train_eval[1]}")

val_eval = history.evaluate(val_gen, steps=5)
print(f"Val Eval Accuracy: {val_eval[1]}")

def plot_history(history):
  # make a figure with two subplots: accurary and loss
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

  # first plot: accurary
  ax1.plot(history.history['accuracy'], label='Training Accuracy', color='green')
  ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
  ax1.set_title("Model Accuracy")
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Accuracy")
  ax1.legend()

  # second plot: loss
  ax2.plot(history.history['loss'], label='Training Loss', color='green')
  ax2.plot(history.history['val_loss'], label='Validation Loss', color='red')
  ax2.set_title("Model Loss")
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel('Loss')
  ax2.legend()

  plt.show()

plot_history(history)