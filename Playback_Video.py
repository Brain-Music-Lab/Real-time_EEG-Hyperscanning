import multiprocessing as mp
import cv2
from mne_lsl.player import PlayerLSL
import mne


def player_process(raw, status):
    player = PlayerLSL(raw, chunk_size=200, name="Simple2_Combined")
    player.start()
    status.value = 1  # signal that LSL streaming has started
    while status.value:
        pass
    player.stop()

def video_process(video_path, status, start_frame=0, scale=3.0):
    # wait until LSL player has started
    while status.value != 1:
        pass

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second [web:15][web:18]

    # jump to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # seek to specific frame [web:12][web:20]

    # Create resizable window
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

    while status.value and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for display
        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
        
        cv2.imshow("Video", resized_frame)
        # present each frame for ~1/fps seconds
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    status.value = 0  # also stop LSL when video ends

if __name__ == "__main__":
    manager = mp.Manager()
    status = manager.Value("i", 0)

    raw = mne.io.read_raw_fif("raw_simple_combined.fif", preload=False)
    video = "sub-P005_ses-simple_task-Default_run-002_eeg.avi"
    first_frame = 172.0

    while True:
        status.value = 0  # Reset status for new iteration

        p_lsl = mp.Process(target=player_process, args=(raw, status))
        p_vid = mp.Process(target=video_process,
                           args=(video, status, first_frame, 2.0))

        p_lsl.start()
        p_vid.start()

        p_lsl.join()
        p_vid.join()
