
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

# Placeholder for a pre-trained style transfer model
# In a real application, this would load a complex model like VGG or a dedicated NST network
class DummyStyleTransferModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate a simple transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.inv_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
            transforms.ToPILImage(),
        ])

    def forward(self, content_image: Image.Image, style_image: Image.Image = None):
        # In a real model, style_image would be used to extract style features
        # For this dummy, we just process the content image
        tensor_image = self.transform(content_image).unsqueeze(0)
        # Simulate some processing, e.g., applying a fixed filter or just returning it
        # For realism, let's just return the processed content image for now
        # A real NST would involve feature extraction, style transfer, and image reconstruction
        return self.inv_transform(tensor_image.squeeze(0).cpu())

app = FastAPI(
    title="Neural Style Transfer API",
    description="API for applying artistic styles to images using deep learning.",
    version="1.0.0",
)

# Load the dummy model (or a real one)
style_model = DummyStyleTransferModel()

@app.get("/", summary="Root endpoint")
async def read_root():
    return {"message": "Neural Style Transfer API is online! Visit /docs for API documentation.", "status": "healthy"}

@app.post("/stylize/", summary="Apply neural style transfer to an image")
async def stylize_image(
    content_image: UploadFile = File(..., description="Content image to stylize"),
    style_image: UploadFile = File(None, description="Optional style image to apply. If not provided, a default style might be used.")
):
    try:
        # Read content image
        content_bytes = await content_image.read()
        content_img = Image.open(io.BytesIO(content_bytes)).convert("RGB")

        # Read style image if provided
        style_img = None
        if style_image:
            style_bytes = await style_image.read()
            style_img = Image.open(io.BytesIO(style_bytes)).convert("RGB")

        # Perform style transfer (using the dummy model)
        # In a real scenario, style_model.forward would take both images
        # and perform the actual style transfer. Here, we just pass content_img.
        stylized_img_pil = style_model.forward(content_img, style_img)

        # Save stylized image to a byte buffer
        img_byte_arr = io.BytesIO()
        stylized_img_pil.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during style transfer: {str(e)}")

@app.post("/upload-style-model/", summary="Upload a custom style model (for advanced users)")
async def upload_style_model(model_file: UploadFile = File(..., description="PyTorch model file (.pth) for style transfer")):
    # This endpoint would handle saving and loading custom models
    # For now, it's a placeholder to show API extensibility
    return {"message": f"Model {model_file.filename} uploaded successfully. Integration pending.", "status": "received"}

@app.get("/test/")
def test_endpoint():
    return {"message": "Test endpoint working!"}

