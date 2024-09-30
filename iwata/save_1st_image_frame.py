import cv2


video_path = '../assets/video/hiro_demo_2024-09-27-11-33-18.mp4'

cap = cv2.VideoCapture(video_path)

if cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('first_frame.jpg', frame)
    else:
        print('failed')
else:
    print('Could not open')

cap.release()
