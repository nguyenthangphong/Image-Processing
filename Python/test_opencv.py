import cv2
import time

if __name__ == "__main__":
    videoCapture = cv2.VideoCapture("./Video/video1.mp4")

    if not videoCapture.isOpened():
        print("Khong the mo video !")
        exit()

    ret, frame = videoCapture.read()

    if not ret:
        print("Khong the doc khung hinh dau tien !")
        videoCapture.release()
        exit()

    bbox = cv2.selectROI("Tracking", frame, False, False)

    if bbox == (0, 0, 0, 0):
        print("Khong chon ROI!")
        videoCapture.release()
        cv2.destroyAllWindows()
        exit()

    tracker = cv2.TrackerMIL.create()

    tracker.init(frame, bbox)

    fps = 0
    frameCount = 0
    lastTime = time.time()

    while True:
        ret, frame = videoCapture.read()

        if not ret:
            print("Video ket thuc hoac loi khung hinh !")
            break

        ok, bbox = tracker.update(frame)

        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            cv2.putText(frame, "Tracking failure detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        frameCount += 1
        elapsedTime = time.time() - lastTime

        if elapsedTime > 0:
            fps = frameCount / elapsedTime

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    videoCapture.release()
    cv2.destroyAllWindows()
