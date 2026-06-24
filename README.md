# TinyML_Dog_Emotion_Classification_System
Developed an edge AI system that classifies dog emotions from barking audio using TensorFlow Lite and MFCC-based feature extraction. Optimized the model for low-memory deployment on smart pet collars, enabling real-time emotion recognition through on-device inference.

remove_silence.py - removes unnecessary silence for barking audios
trimming_2.py - For splitting into 2 second audio
split_train_test.py - split the multiple 2 second audio clips into test and train data
model.py - creates the tensorflow model
classify.py - uses the model to classify the barking sounds into happy,sad,normal and angry
optimize.py - reduces the size of the model from ~80kb to ~10kb (implemented due to memory constraints in hardware)



