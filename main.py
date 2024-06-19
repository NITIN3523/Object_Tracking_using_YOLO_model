from ultralytics import YOLO
import cv2


## load yolov8 model ##
model = YOLO('yolov8n.pt')

## load video ##
videopath = 'Video/video3.mp4'
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter('output_3.avi', fourcc, 10.0, (700,700))  


## read frame ##
cap = cv2.VideoCapture(videopath)

while True:

    ret,frame = cap.read()
    frame = cv2.resize(frame, (700,700))

    if ret:

        ## detect objects ##
        ## track objects ##
        results = model.track(frame,persist=True)
        # print('reee',results[0].plot())

        ## plot results ## 
        frame_ = results[0].plot()      
        
        out.write(frame_)
        


        ## visualize ## 
        cv2.imshow('re',frame_)

        if cv2.waitKey(10) & 0xFF == ord('s'):
            break
    else:
        break



cap.release()
out.release()