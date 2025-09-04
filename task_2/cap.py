import cv2

cap = cv2.VideoCapture(0)  # 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Resize for model input
    frame_resized = cv2.resize(frame, (224, 224))

    # Step 2: Normalize (0-1)
    frame_normalized = frame_resized / 255.0

    # Step 3: (Optional) Convert to tensor for PyTorch / NumPy array for other libs
    # model_input = torch.tensor(frame_normalized).permute(2, 0, 1).unsqueeze(0)

    # Display live video
    cv2.imshow('Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
