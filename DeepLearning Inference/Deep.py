import torchvision.models as models
from torchvision import transforms
from PIL import Image
# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Set the model to evaluation mode
model.eval();

# Move the model to GPU
model.cuda()
# Load an image
image = Image.open("path_to_your_image.jpg")

# Define the preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocess the image
input_tensor = preprocess(image)

# Create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# Move the input to GPU
input_batch = input_batch.cuda()
# Perform inference
with torch.no_grad():
    output = model(input_batch)

# Interpret the output (get the predicted class)
_, predicted_class = torch.max(output[0], 0)

print(f"Predicted class: {predicted_class.item()}")
