# Adapted from code in this thread:
# https://stackoverflow.com/questions/32342935/using-opencv-with-tkinter

from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk
import argparse
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import torch.nn as nn

class CrackDetectionFrame(ttk.Frame):
    def __init__(self, parent, model):
        """ Initialize frame using OpenCV + Tkinter. """
        # OpenCV captures an image periodically from video on webcam

        super().__init__(parent)
        self.pack()

        self.vs = cv2.VideoCapture(0) # from computer webcam (default)
        self.model = model
        self.model.eval()

        self.current_image = None 
        self.pil_font = ImageFont.truetype("fonts/DejaVuSans.ttf", 30) #set font and size
        
        # self.destructor function gets fired when the window is closed
        parent.protocol('WM_DELETE_WINDOW', self.destructor)

        # displays image
        self.panel = ttk.Label(self)  
        self.panel.pack(padx=10, pady=10)

        # shows the video on screen
        self.video_loop()

    def video_loop(self):
        """ Capture video frame, process it, and display prediction. """
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            self.current_image = ImageOps.crop(self.current_image, (100,0,100,0)) #make sure webcam is big enough
            self.current_image = ImageOps.mirror(self.current_image) 

            # Preprocess image for model
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel if necessary
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean/std for grayscale
            ])
            img_tensor = transform(self.current_image).unsqueeze(0)  # Add batch dimension
            
            # Predict using the model
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, pred_idx = torch.max(outputs, 1)  

            # Map class indices to labels
            class_labels = {0: "No Crack", 1: "Crack"}
            pred_label = class_labels.get(pred_idx.item(), "Unknown")
            pred_str = f"Prediction: {pred_label}"


            # add text prediction on image
            draw = ImageDraw.Draw(self.current_image)
            draw.text((7, 7), pred_str, font=self.pil_font, fill='aqua')
            
            # Convert image to tkinter format
            imgtk = ImageTk.PhotoImage(image=self.current_image) 
            self.panel.imgtk = imgtk  
            self.panel.config(image=imgtk)

        self.after(30, self.video_loop) 

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] Closing...")
        self.master.destroy()
        self.vs.release()  
        cv2.destroyAllWindows()  

if __name__ == '__main__':
    
    model = models.mobilenet_v2(pretrained=False)  # pretrained=False since we're using our trained model
    
    # make sure input is the correct for the model
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

    # Modify the final classifier layer to have 2 output classes (binary classification)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    # Load the weights from the saved model
    model_path = 'crack_detection_model_mobilenet.pth'
    
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        print(f"[INFO] Model loaded from {model_path}")
    except Exception as e:
        print(f"[ERROR] Could not load model from {model_path}")
        print(f"        {e}")
        exit()


    # Start the GUI
    print("[INFO] Starting...")
    gui = tk.Tk() 
    gui.title("Crack Detection Prediction")  
    #gui.geometry("800x600")
    CrackDetectionFrame(gui, model)
    gui.mainloop()
     
