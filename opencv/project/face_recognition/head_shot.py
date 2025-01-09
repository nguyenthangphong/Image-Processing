import cv2
import os
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python head_shots.py <name> <path>")
        sys.exit(1)

    name = sys.argv[1]
    path = sys.argv[2]

    save_path = os.path.join(path, name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Error: Could not open the camera.")
        sys.exit(1)

    cv2.namedWindow("Press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Press space to take a photo", 640, 480)

    count_image = 0

    while True:
        ret, frame = capture.read()
        if not ret or frame is None:
            print("Error: Could not load the image.")
            break
        cv2.imshow("Press space to take a photo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Escape hit, closing...")
            break
        elif key == 13:
            image_name = os.path.join(save_path, f"image_{count_image}.jpg")
            if not cv2.imwrite(image_name, frame):
                print(f"Error: Failed to save image {image_name}")
            else:
                print(f"{image_name} written!")
            count_image += 1

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
