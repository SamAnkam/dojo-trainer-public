# dojo-trainer-public
following the principles of Microsoft's Kinect motion capture and Google's Mediapipe library to assist karate students (karateka) as they practice their forms at home. this allows a student to input a sequence of moves, or a kata, that's specific to their dojo, and the webcam on their device will follow them and showcase similarity scores.

how it's made:
we've used Google MediaPipe's library as a starting point to create reference files of the skeleton as it moves through each pre-recorded video for a specific pose (these videos will need to be handed over by the dojo master, to tailor the trainer to their liking). this code is not included in this public repository*. once a student runs the trainer, their body landmarks will be augmented (to reduce/remove yaw, pitch, and varying body types), in real time, to ensure normalization with the reference database before checking for similarity using a DTW algorithm.

*email the owner (Sammy Ankam, samhithankam@gmail.com) for access to the entire codebase
