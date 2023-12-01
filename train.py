from ultralytics import YOLO
import onnx
model = YOLO('yolov8n.yaml')
model = YOLO('yolov8n.pt')
path = '/content/datasets/waste-detection-9/data.yaml'
results = model.train(data=path, epochs=50)
results = model.val()
success = model.export(format='onnx')

