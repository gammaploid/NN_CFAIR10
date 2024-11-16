class ModelBuilder(object):
    def __init__(self, model, loss_fn, optimizer, print_loss_freq=10):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.train_loader = None
        self.val_loader = None
        # self.writer = None

        self.train_losses = []
        self.val_losses = []

        self.train_accuracies = []
        self.val_accuracies = []

        self.total_epochs = 0

        # sets frequency of printing the losses
        self.print_loss_freq = print_loss_freq

        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        # This method allows the user to specify a different device
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Couldn't send it to {device}, sending it to {self.device} instead.")
            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Calculate accuracy
            _, predicted = torch.max(yhat.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)

            return loss.item(), accuracy

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)

            # Calculate accuracy
            _, predicted = torch.max(yhat.data, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)

            return loss.item(), accuracy

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []

        mini_batch_accuracies = []

        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss, mini_batch_accuracy = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
            mini_batch_accuracies.append(mini_batch_accuracy)

        loss = np.mean(mini_batch_losses)
        accuracy = np.mean(mini_batch_accuracies)
        return loss, accuracy

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, seed=42):
        # To ensure reproducibility of the training process
        self.set_seed(seed)

        for epoch in range(n_epochs):
            # Keeps track of the numbers of epochs
            self.total_epochs += 1

            # inner loop
            # Performs training using mini-batches
            loss, accuracy = self._mini_batch(validation=False)
            self.train_losses.append(loss)
            self.train_accuracies.append(accuracy)

            # VALIDATION - no gradients in validation!
            with torch.no_grad():
                val_loss, val_accuracy = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)

            # print out the losses every few epochs
            if (epoch + 1) % self.print_loss_freq == 0:
                print(
                    f"Epoch {epoch+1}, "
                    f"Training loss: {loss:.4f}, accuracy: {accuracy:.4f}, "
                    f"Validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}"
                )

    def predict(self, x):
        # Set it to evaluation mode for predictions
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        # Send input to device and uses model for prediction
        y_hat_tensor = self.model(x_tensor.to(self.device))
        # Set it back to train mode
        self.model.train()
        # Detaches it, brings it to CPU and back to Numpy
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(self.train_losses, label="Training Loss", c="b")
        ax.plot(self.val_losses, label="Validation Loss", c="r")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        fig.legend()
        fig.tight_layout()

    def plot_accuracies(self):
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(self.train_accuracies, label="Training Accuracy", c="b")
        ax.plot(self.val_accuracies, label="Validation Accuracy", c="r")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        fig.legend()
        fig.tight_layout()
        plt.show()

    def save_checkpoint(self, filename):
        # Builds dictionary with all elements for resuming training
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        # Loads dictionary
        checkpoint = torch.load(filename)

        # Restore state for model and optimizer
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["epoch"]
        self.train_losses = checkpoint["train_loss"]
        self.val_losses = checkpoint["val_loss"]

        self.model.train()  # always use TRAIN for resuming training