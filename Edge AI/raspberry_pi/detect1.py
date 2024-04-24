import argparse
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

def load_model(model_path, enable_edge_tpu=False):
    interpreter_options = {}
    if enable_edge_tpu:
        interpreter_options['experimental_use_pure_tflite'] = False
        interpreter_options['experimental_optimization_level'] = tf.lite.Optimize.DEFAULT

    interpreter = tf.lite.Interpreter(model_path=model_path, **interpreter_options)
    interpreter.allocate_tensors()

    return interpreter

def run_inference(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num_detections = int(interpreter.get_tensor(output_details[3]['index']))

    return boxes, classes, scores, num_detections

def load_ground_truth_data(ground_truth_path):
    # Load ground truth data from the CSV file
    df = pd.read_csv(ground_truth_path)
    ground_truths = df[['left', 'top', 'right', 'bottom']].values
    return ground_truths

def evaluate_detections(ground_truths, detections):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for detection in detections:
        if any(detection_overlaps_with_ground_truth(detection, ground_truth) for ground_truth in ground_truths):
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(ground_truths) - true_positives

    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    return precision, recall, f1_score

def detection_overlaps_with_ground_truth(detection, ground_truth):
    # Simple bounding box overlap check
    return (
        detection[0] < ground_truth[2] and
        detection[2] > ground_truth[0] and
        detection[1] < ground_truth[3] and
        detection[3] > ground_truth[1]
    )

def preprocess_frame(frame):
    # Check if the frame is valid
    if frame is None or frame.size == 0:
        return None

    # Resize the frame to match the input size expected by the model
    input_tensor_size = (320, 320)  # Adjust this size based on your model requirements

    try:
        resized_frame = cv2.resize(frame, input_tensor_size)
    except cv2.error as e:
        print(f"Error resizing frame: {e}")
        return None

    # Normalize pixel values to be in the range [0, 1] and convert to UINT8
    normalized_frame = tf.image.convert_image_dtype(resized_frame, dtype=tf.uint8)

    # Expand dimensions to create a batch size of 1
    input_tensor = np.expand_dims(normalized_frame.numpy(), axis=0)

    return input_tensor

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--groundTruth',
        help='Path of the ground truth CSV file.',
        required=False,
        default='test_data/table_results.csv')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=1)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from the camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from the camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    interpreter = load_model(args.model, args.enableEdgeTPU)

    # Load ground truth data
    ground_truths = load_ground_truth_data(args.groundTruth)

    # Capture video from camera
    cap = cv2.VideoCapture(args.cameraId)
    cap.set(3, args.frameWidth)
    cap.set(4, args.frameHeight)

    while True:
        ret, frame = cap.read()

        # Preprocess the frame for inference (replace with your preprocessing code)
        input_tensor = preprocess_frame(frame)

        # Run inference
        boxes, _, _, num_detections = run_inference(interpreter, input_tensor)
# Extract relevant information from detection results
        detections = []
        for i in range(num_detections):
            detection = boxes[0][i]
            detections.append(detection)

        # Evaluate detections
        precision, recall, f1_score = evaluate_detections(ground_truths, detections)

        # Display the results (you can customize this part based on your needs)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        # Display the resulting frame
        cv2.imshow('Object Detection', frame)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main()

