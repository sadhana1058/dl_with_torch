import matplotlib.pyplot as plt

def plot_train_test_accurracy_loss(train_losses,train_accuracy, test_losses,  test_accuracy):
  fig, axs = plt.subplots(2,2,figsize=(16,10))
  axs[0, 0].plot(train_losses, label='Training Loss') 
  axs[0, 0].set_title("Training Loss") 
  axs[0, 0].set_xlabel("Number of epochs")
  axs[0, 0].set_ylabel("Training loss") 
  axs[0, 0].legend()

  axs[1, 0].plot(train_accuracy, label='Training Accuracy')
  axs[1, 0].set_title("Training Accuracy")
  axs[1, 0].set_xlabel("Number of epochs")
  axs[1, 0].set_ylabel("Training accuracy")  
  axs[1, 0].legend()

  axs[0, 1].plot(test_losses, label='Test Loss')
  axs[0, 1].set_title("Test Loss")
  axs[0, 1].set_xlabel("Number of epochs")
  axs[0, 1].set_ylabel("Test loss")  
  axs[0, 1].legend()

  axs[1, 1].plot(test_accuracy, label='Test Accuracy')
  axs[1, 1].set_title("Test Accuracy")
  axs[1, 1].set_xlabel("Number of epochs")
  axs[1, 1].set_ylabel("Test accuracy")  
  axs[1, 1].legend()		

  fig.tight_layout(pad=3.0)
	
	