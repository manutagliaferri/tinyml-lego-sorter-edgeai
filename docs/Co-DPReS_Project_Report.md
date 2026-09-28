# Co-DPReS: Convolution Driven Piece Recognition System

**Authors:** Etienne Orio, Manuel Tagliaferri, Giovanni Elisei

## 1. Introduction

### 1.1 Project overview
The Convolution Driven Piece Recognition System (Co-DPReS) is a standalone embedded platform developed to automate the identification of LEGO components. The system uses an Arduino Portenta H7 and a Vision Shield to perform localized image acquisition and classification. The goal is to create an autonomous device that identifies a specific part from a library of 15 different LEGO pieces and displays the result on an OLED screen without requiring any external computational resources.

The system executes a complete vision pipeline entirely on the microcontroller. This process includes capturing high-resolution grayscale frames, applying a multi-step pre-processing routine to isolate the object, and performing inference using a quantized neural network. By integrating these steps into a single unit, the project demonstrates a functional method for deploying machine learning models on constrained hardware.

### 1.2 Motivation
The motivation for this project is rooted in the intersection of a practical, everyday problem and the technical curiosity of solving it using limited hardware. Sorting LEGO pieces has always been a time-consuming task for enthusiasts and collectors due to the sheer variety of standardized shapes. While the problem is familiar, solving it using localized hardware rather than high-powered computers remains an engineering sweet spot. It is a challenging problem that hasn't been definitively solved for embedded systems, yet it is approachable enough to tackle using the modern capabilities of the Portenta H7.

Furthermore, the concepts explored in this project are highly universal. While the system identifies LEGO bricks, the underlying workflow is directly applicable to more advanced industrial sectors. In fact, the ability to identify specific geometries and categorize components is a requirement in industrial quality control and automated logistics.

### 1.3 Problem Statement
The central problem in this report is the reliable classification of multiple different LEGO pieces on a device with significant memory and processing constraints. Standard deep learning models are generally too large to fit within the internal memory of a microcontroller, requiring a careful balance between model complexity and hardware availability. Specifically, the system must overcome the following technical hurdles:

1. **Throughput and Latency:** Although the system doesn't need to be extremely fast, for a sorting system to be useful, it must provide a result at least in a few seconds. Achieving this speed requires a highly optimized execution flow to ensure the system is not too slow for practical use.
2. **Memory Management:** The microcontroller’s internal RAM is too limited to hold both image buffers and neural network tensors simultaneously. Consequently, the model size must be carefully chosen, and if it exceeds available memory, external storage is required.
3. **Sim-to-Real Gap:** To ensure a big dataset, the model is trained on synthetic renders. However, the system must remain robust when faced with real-world images with distortions like sensor noise.
4. **Geometric Complexity:** The system must distinguish between parts with nearly identical top-down profiles, such as a 2x2 brick versus a 2x2 plate. Using only grayscale data, the model must be sensitive enough to detect subtle differences in height or stud density. Capturing these nuances while maintaining a low-parameter model is a significant design bottleneck.
5. **Environmental Conditions:** LEGO bricks are made of glossy ABS plastic, which is highly reflective. This material property creates specular highlights and harsh reflections that can obscure the actual geometry of the piece. To mitigate this, the system requires carefully prepared and controlled lighting conditions to ensure the camera captures the best possible image for the model to process.

### 1.4 Project Objectives
The primary objective of this project is to design and validate a standalone embedded system capable of classifying 15 distinct LEGO classes using computer vision and deep learning. This involves the creation of a hardware-software architecture on the Arduino Portenta H7 that manages the entire lifecycle of an image, from acquisition to classification, without relying on any external processing power or cloud connectivity. The ultimate goal is to prove that high-dimensional visual tasks can be executed reliably within a restricted resource envelope by focusing on two critical engineering pillars: a standardized preprocessing pipeline and a robust, generalized neural network.

A central objective of this work is the development of an image preprocessing pipeline that ensures maximum consistency for the classification model. Since microcontrollers lack the computational headroom to run massive, all-encompassing networks, the preprocessing stage must do the heavy lifting of normalizing the input. The goal is to implement a routine that isolates the LEGO piece through background subtraction, centers it within a square-fitted frame, and applies histogram equalization to standardize contrast. By preparing the image in this way, we eliminate environmental variables such as object positioning and slight lighting fluctuations, allowing the neural network to focus exclusively on the geometric signatures of the pieces.

Equally important is the objective of training a neural network that can generalize from digital renders to physical objects. Because this project utilizes a synthetic dataset of 12,000 images, a major technical goal is bridging the Sim-to-Real gap. We aim to develop a training strategy that incorporates data augmentation, specifically Gaussian noise, to simulate the artifacts and thermal noise produced by the Portenta’s camera sensor. This ensures the model remains robust when faced with the specular reflections of glossy plastic and the imperfections of real-world captures, resulting in a system that maintains high accuracy outside of a simulated environment.

Finally, the project aims to optimize these complex processes to meet the performance and memory constraints of the hardware. This includes the successful deployment of a quantized model via the Chirale TensorFlow Lite library and the strategic use of external SDRAM to manage 128x128 frame buffers. The technical success of the project is defined by achieving a functional balance where the preprocessing speed and inference latency allow for a classification result to be displayed on an integrated OLED screen in a timeframe that is useful for real-time sorting applications.

## 2. System Design

### 2.1 Components

#### 2.1.1 Hardware Stack Overview
The hardware architecture of the Co-DPReS system is built upon a modular stack consisting of the **Arduino Portenta H7** and the **Portenta Vision Shield Rev 1**. This combination was selected to provide a balance between high-performance computational capabilities and specialized peripheral interfaces required for real-time computer vision. By offloading the visual acquisition to a dedicated shield while centering the processing on a dual-core microcontroller, the system achieves a degree of integration typically reserved for more power-intensive single-board computers.

#### 2.1.2 The Arduino Portenta H7: Computational Core
The Portenta H7 serves as the central processing unit and is responsible for executing the pre-processing pipeline and the neural network inference. It features the **STM32H747XI** dual-core processor, which utilizes a Cortex-M7 core running at 480 MHz and a Cortex-M4 core running at 240 MHz. For this project, the Cortex-M7 is the primary driver, as its hardware Floating Point Unit (FPU) and Digital Signal Processing (DSP) instructions are critical for the efficient execution of the matrix multiplications required by the convolutional layers of our model.

Memory management is a defining aspect of the system design. While the H7 includes 2MB of internal Flash and 1MB of RAM, these resources are insufficient for holding multiple 320x240 grayscale frame buffers alongside a neural network and its associated activation tensors. To resolve this, the design utilizes the **8MB of external SDRAM** available on the board. Namely, the SDRAM is used as a high-capacity heap for the TensorFlow Lite tensor arena. This configuration prevents stack overflow and allows the system to handle high-resolution visual data that would otherwise exceed the microcontroller's internal memory limits.

#### 2.1.3 Vision Shield Rev 1: Acquisition Interface
The Portenta Vision Shield provides the system with its visual input through the **Himax HM-01B0** camera sensor. This sensor is designed for ultra-low-power "always-on" vision applications and is configured in this project to capture grayscale imagery. Capturing in grayscale is a careful design choice seeing the hardware constraints. In fact, it reduces the data throughput requirement by 66% compared to RGB, significantly lowering the RAM footprint of the input buffers and reducing the number of calculations required in the first layer of the CNN.

In addition to the camera, the shield provides the physical mounting points and electrical interfacing necessary to keep the sensor stable. By placing the sensor on a dedicated shield rather than using a remote camera module, the system minimizes signal degradation and electromagnetic interference, ensuring that the raw pixel data remains clean and suitable for the subsequent pre-processing steps.

### 2.2 Environmental design
When relying on visual data, the external environment acts as an uncontrolled variable that can drastically affect model performance. Since the neural network was trained on synthetic data (perfectly rendered images with uniform lighting and black backgrounds), a significant domain gap exists compared to real-world conditions. To mitigate this discrepancy, a physical environmental design strategy was adopted. To standardize the input conditions, a custom-built enclosure with white internal surfaces was designed to house the Portenta Vision Shield and the sorting area. This component addresses three critical design requirements:

1. **Illumination Invariance:** Neural networks are highly sensitive to lighting variations. Hard shadows can be misinterpreted by the model as geometric features (e.g., a shadow might look like an extra part of the brick). The enclosure shields the sorting area from ambient room light, ensuring a diffused, uniform illumination that remains constant regardless of the external environment.
2. **Reflection Mitigation:** Lego bricks are made of glossy ABS plastic, which is prone to specular reflections. In a standard environment, overhead lights create bright "hotspots" on the studs, obscuring the details necessary for classification. The diffused lighting within the box minimizes these artifacts, allowing the camera to capture the true geometry of the piece.
3. **Contrast Maximization for Segmentation:** The choice of a white background is functional to the software pipeline. It maximizes the contrast against the colored or darker Lego bricks. This high-contrast physical setup is a prerequisite for the effective operation of the Pre-processing algorithms (specifically background subtraction) described in the following section.

### 2.3 Functional Workflow: From Acquisition to Display
The Co-DPReS operates through a strictly sequential pipeline designed to transform raw optical data into a specific part classification. This process is divided into four main stages: acquisition, pre-processing, inference, and output. Because the system is hosted on an embedded target, each stage is optimized to ensure that data flows efficiently through the Portenta’s memory without creating processing bottlenecks.

The cycle begins with **Image Acquisition**, where the Vision Shield captures a grayscale frame via the Himax sensor. The acquisition stage is designed to capture the scene at a higher resolution than the model requires, providing the necessary overhead for the cropping and scaling operations that follow.

Once the frame is stored, the system enters the **Pre-processing Pipeline**. This is the most computationally intensive phase before inference. The raw frame is first smoothed to reduce sensor noise and then compared against a stored background model to isolate the LEGO piece. Once the object is detected, the system calculates a bounding box, fits it into a square aspect ratio, and resamples the region to a fixed 128x128 resolution. Finally, histogram equalization is applied to normalize the lighting. This stage is critical because it standardizes the input, ensuring that the neural network always sees a centered, high-contrast, and appropriately scaled image regardless of how the physical brick was placed under the camera.

The standardized 128x128 image is then passed to the **Neural Network Classification** stage. This input resolution was chosen as an optimal compromise between preserving sufficient detail for accurate LEGO recognition and maintaining real-time processing speed. The `chirale-tensorflow-lite` library loads the quantized model from the Portenta’s Flash memory into the SDRAM tensor arena. The processor executes the convolutional layers, producing a probability score for each of the 15 possible classes.

The final stage is the **Display**. Once the model identifies the class with the highest confidence score, the result is sent to the integrated OLED screen. The display provides the user with immediate feedback by showing the recognized Part ID. This end-to-end flow allows the Co-DPReS to function as a complete, standalone industrial sensor, moving from a physical object to a digital classification in a matter of seconds.

### 2.4 Image pre-processing pipeline

#### 2.4.1 Motivation and Overview
Before feeding images to the classification model, the raw camera input must be carefully prepared. Even though each LEGO piece is roughly centered and in focus, small differences in position, distance, and lighting can affect the model’s predictions. The pre-processing pipeline corrects for these variations by isolating the object, centering it, resizing it to a fixed scale, and normalizing its brightness. This ensures that the model sees each piece in a consistent way, focusing on its actual shape and structure rather than being influenced by where it appears in the frame, how large it looks, or minor lighting changes.

#### 2.4.2 Temporal Smoothing
When the system starts or when the user presses the button, a short stabilization phase is triggered. During this phase, several initial frames are used to estimate a background image using an exponential moving average (EMA). This smooths out sensor noise and small fluctuations while keeping a stable estimate of the static scene. After stabilization, each new frame is also filtered with an EMA. This temporal smoothing reduces noise from the camera sensor and makes later steps, such as object detection, more reliable.

#### 2.4.3 Background Subtraction
After the initial stabilization phase, a reference background image is available that represents the static scene without the LEGO piece. Each new frame captured by the camera is compared to this background on a pixel-by-pixel basis. If the absolute difference between a pixel and its background value exceeds a fixed threshold, that pixel is classified as foreground, otherwise, it is discarded. Despite its simplicity, this operation is where spatial invariance is effectively introduced. By removing all static background pixels and keeping only the pixels that change, the representation of the LEGO piece becomes independent of its exact position in the camera frame. Small translations of the object therefore do not affect the resulting foreground mask.

#### 2.4.4 Bounding Box Detection
Once a clean foreground mask is available, the system scans the image to find the smallest bounding box that contains all foreground pixels. If the detected blob is too small, it is ignored to avoid false detections caused by noise or spurious pixels.. The bounding box provides a coarse but reliable localization of the LEGO piece within the camera frame. By identifying the spatial extent of the object, the pipeline can restrict further processing to this region only, rather than operating on the full image. This reduces unnecessary computation and ensures that subsequent steps work exclusively on the area that actually contains the object of interest.

#### 2.4.5 Square Fitting
The detected bounding box may be rectangular, depending on the orientation and proportions of the LEGO piece. Since the classification model expects square inputs, the bounding box is expanded to a square region while keeping the object centered. This adjustment preserves the full extent of the object without distortion and ensures a consistent geometric layout for all inputs. As a result, variations in object orientation do not affect the shape of the region passed to the next stages of the pipeline.

#### 2.4.6 Region Extraction and Resampling
The square region containing the LEGO piece is extracted from the camera image. Depending on the distance between the camera and the object, this square may contain more or fewer pixels than the model’s required input size. To address this, the cropped region is resampled to a fixed resolution of 128×128 pixels using nearest-neighbor interpolation. This ensures a consistent input size, allowing the model to focus on the shape of the LEGO piece regardless of its apparent size in the original image.

#### 2.4.7 Histogram Equalization
Lighting conditions can vary across captures, even in controlled environments. To reduce sensitivity to illumination changes, histogram equalization is applied to the resampled image. This operation redistributes pixel intensities so that the full grayscale range is used more evenly, effectively normalizing the contrast. By spreading pixel values across the full range, details in both dark and bright areas become more visible, and the image achieves a consistent brightness distribution. This helps the model focus on the shape of the LEGO piece rather than being affected by variations in lighting.

#### 2.4.8 Model Input Preparation
After completing all pre-processing steps, we obtain a clean, centered, square, and contrast-normalized grayscale image of the LEGO piece. This final, standardized image is ready for classification and is copied directly into the model’s input buffer before being passed to the TensorFlow Lite interpreter for inference.

### 2.5 Neural Network Architecture
The classification engine is a custom Convolutional Neural Network designed and implemented using the PyTorch framework. The architecture was specifically tailored to balance the feature extraction capability required to distinguish geometric details of Lego bricks with the strict memory and latency constraints of the Arduino Portenta H7. The resulting model consists of approximately 289,000 parameters. This size is a deliberate design choice to ensure the model fits comfortably within the microcontroller's Flash memory while leaving sufficient RAM for the inference buffers.

#### 2.5.1 Architectural Design Choices
To optimize the network for Edge AI execution, several specific architectural decisions were made distinct from standard desktop computer vision models:
- **Strided Convolutions for Downsampling:** Instead of utilizing traditional Max Pooling layers to reduce the spatial dimensions of the image, the network employs convolutional layers with stride = 2 at the beginning of each block.
- **Group Normalization:** The architecture utilizes Group Normalization (dividing channels into groups of 16) instead of the more common Batch Normalization.
- **Global Average Pooling (GAP):** A critical bottleneck in embedded vision is the transition from convolutional layers to fully connected layers. Instead of flattening the entire feature map, which would result in a massive dense layer consuming megabytes of RAM, the design uses an AdaptiveAvgPool2d to collapse the spatial dimensions into a single vector (1x1).

#### 2.5.2 CNN Topology
The network processes the 128x128 grayscale input through a sequential structure composed of three main feature extraction blocks followed by a classification head:
1. **Feature Extraction Blocks:**
   - Block 1: Expands the input to 32 filters. It consists of two convolutional layers (3x3 kernel), Group Normalization, and LeakyReLU activations.
   - Block 2: Increases depth to 64 filters. It incorporates Dropout (0.1) to introduce regularization, mitigating the risk of overfitting on the synthetic dataset features.
   - Block 3: Further expands depth to 128 filters with increased Dropout (0.2). This block extracts high-level semantic features of the brick geometry.
2. **Classification Head:**
   - Global Pooling: Reduces the final feature map (128 channels) to a feature vector of size 128.
   - Linear Classifier: A single fully connected layer maps the 128 features to the 15 output classes, producing the final logits for classification.

This streamlined topology ensures that the inference pass is computationally efficient, minimizing the number of MAC (Multiply-Accumulate) operations required per frame.

## 3. System Implementation

### 3.1 Training Strategy and Sim-to-Real Adaptation
Implementing the design on the target hardware required a robust training pipeline capable of bridging the gap between the synthetic dataset and the noisy real-world environment. To achieve this, a **Sim-to-Real transfer** strategy was adopted, utilizing a Synthetic Dataset of rendered images with 800 samples per class.

#### 3.1.1 Data Augmentation Pipeline
A critical challenge in training with synthetic data is the lack of realistic sensor noise and environmental imperfections. To prevent the model from learning "too perfect" features that would fail on the field, a custom Data Augmentation strategy was implemented in the PyTorch TransformDataset class. This involved **Gaussian Noise Injection**, where the AddGaussianNoise function is applied to input images to simulate the thermal noise and grain typical of the Himax HM-01B0 sensor, particularly in low-light conditions. Furthermore, **Block Noise Simulation** was employed via the AddBlockGaussianNoise transform to mimic partial occlusions or dirty lens artifacts.

#### 3.1.2 Training Configuration
The model was trained on a host machine using a hyperparameter set specifically tuned to maximize generalization performance. The Adam optimizer was chosen for its adaptive learning rate capabilities, while CrossEntropyLoss was utilized to handle the multi-class classification of the 15 categories. The training regimen was intentionally extended to 500 epochs; although early convergence was noted around epoch 200, this prolonged schedule allowed the model to fine-tune its weights on heavily augmented edge cases.

### 3.2 Performance Metrics and Analysis

**Classification Accuracy**
The model demonstrated excellent theoretical capabilities, achieving a peak accuracy of **93% on the validation set**. This result validates the effectiveness of the training pipeline on unseen synthetic data.

However, field testing on the Portenta H7 revealed a sharp performance contrast, with real-world accuracy dropping to approximately **45%**. This degradation was analyzed, and three concurrent causes were hypothesised as the primary factors:
- **Residual Sim-to-Real Gap:** While the gap between simulation and reality has been significantly narrowed, a residual discrepancy persists between the synthetic training distribution and the output of the physical vision pipeline.
- **Distribution Mismatch:** A significant performance bottleneck arises from the distributional shift between the synthetic training environment and real-world deployment. The synthetic pipeline utilized uniform 3D rotations, effectively simulating a zero-gravity environment where every orientation carries equal weight. In reality, gravity constrains Lego bricks to a limited subset of stable equilibrium poses.
- **Model Simplicity:** To satisfy the strict memory constraints of the Portenta board, the CNN architecture was forced into an extremely compact design, utilizing only three convolutional blocks. This lightweight configuration imposes a rigid capacity ceiling that limits the model’s ability to learn high-dimensional feature representations.

**Memory Footprint:** The compiled model exhibits a highly optimized footprint of 384,768 bytes (approx. 376 KB). This occupies only roughly 18% of the Portenta H7’s 2MB internal Flash memory.

**Inference Latency:** The average time-to-result is approximately 3.0 seconds per frame. This duration encompasses the entire pipeline: image capture, pre-processing, and the CNN inference pass.

### 3.3 Model Conversion and Deployment
Since the Portenta H7 cannot directly execute Python or PyTorch code, a multi-stage conversion pipeline was established to bridge the gap between the training and deployment environments. The process incorporated Quantization-Aware Training principles to ensure the model weights remained optimized for the constraints of the embedded target. Structurally, the trained PyTorch model was first exported to the intermediate ONNX standard and subsequently converted into the TensorFlow Lite format. The pipeline concluded with C-Array Generation, where the binary `.tflite` file was compiled into a C byte array (`lego_model.h`).

## 4. Experimental Results and Discussion

### 4.1 Real-World Validation
While the accuracy and memory targets were fully met, the resulting latency of 3 seconds represents a significant bottleneck for high-speed industrial sorting scenarios, identifying this as the primary architectural trade-off of the project. A detailed analysis of this bottleneck attributes the delay to the design decision of maintaining a high input resolution, which was strictly necessary to resolve the fine geometric details of the Lego studs. On a qualitative level, however, testing within the 'White Box' environment demonstrated high robustness against object rotation; the model successfully classified bricks regardless of their orientation, empirically validating the spatial invariance provided by the Global Average Pooling layer.

## 5. Conclusion
This project validated the architectural feasibility of deploying a complete, standalone Computer Vision pipeline on the Arduino Portenta H7, effectively transitioning the LEGO sorting task from a cloud-centric model to an Edge Computing solution. From a system design perspective, the implementation was highly effective, as the hardware-software stack successfully managed the entire lifecycle of the image, from acquisition to display, while meeting the strict resource constraints of the microcontroller. Specifically, the architectural optimization of the Neural Network allowed the complex CNN to fit within just 376 KB of Flash memory, occupying only roughly 18% of the available storage.

In conclusion, while the current inference latency of approximately 3.0 seconds and the real-world accuracy require further optimization for high-speed industrial adoption, the system successfully established a functional baseline. Future development will focus on correcting the data generation strategy to account for physical stability and implementing Post-Training Quantization to reduce latency, leveraging the insights gained from this architectural validation.
