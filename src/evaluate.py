import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_learning_curves(history, title_prefix="Model"):
    """Affiche les courbes d'accuracy et de loss (train vs validation)."""
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.title(f"{title_prefix} — Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Train loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.title(f"{title_prefix} — Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_ds, class_names):
    """Affiche la matrice de confusion et le rapport de classification."""
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


def show_misclassified(model, val_ds, class_names, n=9):
    """Affiche les images mal classifiées avec la vraie classe et la prédiction."""
    misclassified_imgs = []
    misclassified_true = []
    misclassified_pred = []

    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        for i in range(len(labels)):
            true = labels.numpy()[i]
            pred = pred_labels[i]
            if true != pred:
                misclassified_imgs.append(images[i].numpy())
                misclassified_true.append(true)
                misclassified_pred.append(pred)
        if len(misclassified_imgs) >= n:
            break

    n_show = min(n, len(misclassified_imgs))
    if n_show == 0:
        print("Aucune erreur de classification.")
        return

    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).flatten()

    for i in range(n_show):
        img = misclassified_imgs[i]
        # normalisation pour l'affichage si nécessaire
        if img.max() > 1.0:
            img = img / 255.0
        axes[i].imshow(img)
        axes[i].set_title(
            f"Vrai : {class_names[misclassified_true[i]]}\n"
            f"Prédit : {class_names[misclassified_pred[i]]}",
            fontsize=9,
        )
        axes[i].axis("off")

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Analyse qualitative des erreurs", fontsize=12)
    plt.tight_layout()
    plt.show()
