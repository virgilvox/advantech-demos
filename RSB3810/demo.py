import cv2
import numpy as np
import sys
import tflite_runtime.interpreter as tflite

with open('labels.txt', 'r') as f:
    labels = [line.strip() for line in f if line.strip()]

if labels and labels[0].startswith('???'):
    labels[0] = 'background'

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

video_device = sys.argv[1] if len(sys.argv) > 1 else '/dev/video0'
cap = cv2.VideoCapture(video_device)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    input_data = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.uint8(input_data)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    for i in range(len(scores)):
        if scores[i] > 0.5:
            cid = int(classes[i]) + 1
            label = labels[cid] if cid < len(labels) else f"Unknown ({cid})"
            print(f"Detected ID {cid}: {label} at {scores[i]*100:.1f}% confidence")

    if hasattr(cv2, 'imshow'):
        for i in range(len(scores)):
            if scores[i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, top, right, bottom) = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]),
                                              int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
                cid = int(classes[i]) + 1
                label = labels[cid] if cid < len(labels) else f"Unknown ({cid})"
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {scores[i]*100:.1f}%", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('RSB‑3810 Live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()