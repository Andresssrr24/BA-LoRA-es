''' Extract features and labels from last hidden layer for tsne_visualization '''
import numpy as np
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_NAME = "TASS_DATASET_POLARITY"
VAL_DATASET = f'/content/{DATASET_NAME}/tass_train_dev/dev.tsv'

print("Extracting features and labels...")

model.eval()

val_loader = DataLoader(val_dataset, batch_size=32) # Validation set dataloader
features_l = []
labels_l = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        labels = batch['label']

        # Get hidden states
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1] # (batch, seq_len, hidden_dim)
        
        sentence_embeddings = last_hidden.mean(dim=1) # (batch, hidden_dim)

        features_l.append(sentence_embeddings.cpu())
        labels_l.append(labels.cpu())
    
    # Concatenate extracted features and labels
    features_concat = torch.cat(features_l, dim=0).numpy()
    labels_concat = torch.cat(labels_l, dim=0).numpy()

    np.save("features_step_final.npy", features_concat)
    np.save("labels_step_final.npy", labels_concat)

    print("Features and labels saved")
