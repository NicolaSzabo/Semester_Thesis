import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(spatial_dims = 3, in_channels = 1, out_channels = num_class).to(device)


### Evaluate the model on the test dataset
model.load_state_dict(torch.load(f"/home/hodl24team4/efficientnet_pytorch/hip_fracture_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = model(test_images).argmax(dim = 1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

print(classification_report(y_true, y_pred, target_names = class_names, digits = 4)) 
