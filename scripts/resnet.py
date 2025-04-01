import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import cv2
import numpy as np

class ResNetSegmentation(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetSegmentation, self).__init__()
        
        # Pretrained ResNet as Encoder
        base_model = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)  # Final layer
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = ResNetSegmentation(num_classes=1)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Preprocessing
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def segment_ladybug(image, boxes):
    """
    Segment the ladybugs using the bounding boxes from YOLO.
    Args:
        image: Input image (numpy array).
        boxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...]
    """
    results = []
    for (x_min, y_min, x_max, y_max) in boxes:
        # Crop image using YOLO box
        cropped = image[y_min:y_max, x_min:x_max]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        input_tensor = transform(cropped).unsqueeze(0).to(device)

        # Forward pass through the model
        with torch.no_grad():
            output = model(input_tensor)
            output = torch.sigmoid(output).cpu().numpy()[0, 0]  # [0, 0] for single-channel mask
        
        # Resize back to original cropped size
        output_resized = cv2.resize(output, (x_max - x_min, y_max - y_min))

        # Reposition mask to original image size
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = output_resized
        results.append(mask)

    # Combine all masks
    final_mask = np.max(results, axis=0)
    return final_mask

# Example usage
# YOLO example boxes: [(x_min, y_min, x_max, y_max), ...]
# image = cv2.imread('ladybug.jpg')
# boxes = [(30, 40, 200, 220), (150, 100, 300, 280)]
# mask = segment_ladybug(image, boxes)

# cv2.imshow('Segmented Ladybug', mask)
# cv2.waitKey(0)
