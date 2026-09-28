import serial
import numpy as np
import cv2
import threading
import time
from collections import deque

# Configuration
WINDOW_WIDTH, WINDOW_HEIGHT = 960, 720 # 384, 384
CAM_WIDTH, CAM_HEIGHT = 320, 240 # 128, 128
PIXEL_COUNT = CAM_WIDTH * CAM_HEIGHT
FRAME_RATE = 30
SERIAL_PORT = "/dev/cu.usbmodem2101"
SYNC_HEADER = b'\xFF\xD8\xFF\xD9'

# Thread-safe buffer and queue
data_buffer = bytearray()
buffer_lock = threading.Lock()
frame_queue = deque(maxlen=2)

# Frame state
receiving_frame = False
frame_index = 0
sync_index = 0
frame_buffer = bytearray(PIXEL_COUNT)

# Display window
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)

def serial_reader(ser):
    """Continuously read serial data into buffer."""
    MAX_BUFFER = PIXEL_COUNT * 5
    while ser.is_open:
        try:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                with buffer_lock:
                    if len(data_buffer) > MAX_BUFFER:
                        data_buffer.clear()
                        ser.reset_input_buffer()
                    data_buffer.extend(chunk)
            else:
                time.sleep(0.0001)
        except Exception as e:
            print("Serial read error:", e)
            break

def process_buffer():
    """Parse buffer and extract complete frames."""
    global receiving_frame, frame_index, sync_index
    
    with buffer_lock:
        if not data_buffer:
            return 0
        buf = bytearray(data_buffer)
        data_buffer.clear()

    frames = 0
    for b in buf:
        if not receiving_frame:
            # Sync detection
            if b == SYNC_HEADER[sync_index]:
                sync_index += 1
                if sync_index == len(SYNC_HEADER):
                    receiving_frame = True
                    frame_index = 0
                    sync_index = 0
            else:
                sync_index = 0
        else:
            frame_buffer[frame_index] = b
            frame_index += 1
            if frame_index == PIXEL_COUNT:
                frame_queue.append(np.frombuffer(frame_buffer, dtype=np.uint8).reshape(CAM_HEIGHT, CAM_WIDTH))
                receiving_frame = False
                frames += 1
    return frames

def main():
    fps = 0
    frame_times = deque(maxlen=30)
    last_frame = None

    try:
        ser = serial.Serial(SERIAL_PORT, PIXEL_COUNT * 8 * FRAME_RATE, timeout=0)
        ser.reset_input_buffer()

        threading.Thread(target=serial_reader, args=(ser,), daemon=True).start()
        print("Connected. Waiting for frames... Press 'q' to quit")

        while True:
            process_buffer()

            if frame_queue:
                last_frame = frame_queue[-1]
                frame_times.append(time.time())
                if len(frame_times) > 1:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])

            if last_frame is not None:
                img = cv2.resize(last_frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)
                # cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow('Camera Feed', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except serial.SerialException as e:
        print("Serial error:", e)
    except KeyboardInterrupt:
        pass
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
