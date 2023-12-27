import multiprocessing

def process_function(result_queue):
    # The code to be executed in the separate process
    boxes, classes, scores, centroids_old = [], [], [], []
    trackerType = trackerTypes[4]
    multiTracker = cv2.MultiTracker_create()

    count = 0
    num_frame_to_detect = 5

    while True:
        t1 = cv2.getTickCount()
        frame = videostream.read()

        _, frame = polygon_cal.cut_frame_polygon(frame)

        success, boxes_update = multiTracker.update(frame)

        if count == num_frame_to_detect:
            centroids = polygon_cal.centroid(boxes_update)
            PointsInfor = polygon_cal.check_result(centroids, centroids_old, frame)
            result_queue.put(PointsInfor)
            count = 0

        if count == 0:
            start_time = time.time()
            boxes, classes, scores, centroids_old = detect_ssd(frame)
            multiTracker = cv2.MultiTracker_create()
            for bbox in boxes:
                box_track = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                multiTracker.add(createTrackerByName(trackerType), frame, box_track)

            if len(scores) == 0:
                count = -1

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        count += 1

def main_process():
    # Initialize Queue to pass data between processes
    result_queue = multiprocessing.Queue()

    # Create and start the separate process
    process = multiprocessing.Process(target=process_function, args=(result_queue,))
    process.start()

    while True:
        try:
            # Retrieve data from the Queue
            PointsInfor = result_queue.get_nowait()
            print(f"Information point: {PointsInfor}\n")
        except queue.Empty:
            pass

        # Continue with your main process logic here
        # ...

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    process.join()

if __name__ == "__main__":
    main_process()
