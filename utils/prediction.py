import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('models/ship-model.h5')

def detect_ships(image, boxes, confidence_threshold=0.8):
    (H, W) = image.shape[:2]
    rois, locs = [], []

    for (x, y, w, h) in boxes:
        if w / float(W) < 0.01 or h / float(H) < 0.01:
            continue
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rois.append(roi)
        locs.append((x, y, x+w, y+h))

    if not rois:
        return image

    input_rois = np.array(rois, dtype=np.float32) / 255.0
    probs = model.predict(input_rois)

    for j, prob in enumerate(probs):
        label = np.argmax(prob)
        confidence = prob[label]
        if label == 1 and confidence >= confidence_threshold:
            (startX, startY, endX, endY) = locs[j]
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, f'{confidence:.2f}', (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image
