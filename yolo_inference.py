from ultralytics import YOLO  

model = YOLO('models/best.pt')  


results = model.predict('uploads/121364_0.mp4', save=True, project='runs', name='detect_output', format='mp4')
print(results[0])
print("***********")
for box in results[0].boxes:
    print(box)
    
    
    
    
    
    
    
    