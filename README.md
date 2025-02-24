# Underwater-Image-Restoration
Abstract

Underwater images often face challenges due to turbidity caused by suspended particles, leading to hazy and distorted visuals. Another problem of the lack of underwater data substantially reduces the efficiency of the trained models. This project aims to propose a novel approach to restore underwater river turbid image using diffusion models. The primary objective is to quantify turbidity noise using a custom laboratory dataset (that simulates progressive turbidity, mimicking the forward noise-adding process of diffusion models, and later, enhancing image clarity by implementing de-noising architecture. The denoising model should be able to use the predicted noise variance for a reverse diffusion process, which iteratively reconstructs the degraded images, restoring their visual quality. Beyond aesthetically pleasing improvements, enhanced images can contribute significantly to underwater research contribution to monitoring marine ecosystems, tracking fish migration patterns, and studying coral reefs. 

Index Terms—Underwater Images, Image Restoration, Diffusion Models 
 
INTRODUCTION 

Underwater imaging plays a vital role in marine research, biodiversity studies, and monitoring environmental changes. However, achieving clear and detailed underwater visuals is often hindered by turbidity—a condition caused by suspended particles in water that scatter and absorb light. This results in hazy, blurry, and distorted images, making it difficult to extract reliable information. The literature review conducted shows that efforts to improve underwater image quality have included both traditional and modern approaches. 
Traditional methods, such as Wiener filtering and the Lucy-Richardson algorithm, focus on de-blurring by reducing noise and restoring lost details. Techniques like Gaussian curvature filtering enhance structural elements while also reducing the runtime of underwater image enhancement, reducing the need to calculate partial derivative operations.  
In recent years, deep learning models have significantly advanced the field of image restoration. U-Net architecture has demonstrated effectiveness in both segmentation and restoration tasks due to their ability to capture spatial hierarchies. The U-Net model takes into consideration the accuracy and the computation cost to enable real-time implementation on underwater visual tasks using an end-to-end auto-encoder network. Adding to these advancements, diffusion models have been implemented as well to address complex distortions in underwater images. These models simulate the process of image degradation through noise addition and reverse it step by step to reconstruct clear and detailed visuals.  
This research focuses on implementing diffusion models for underwater image enhancement with appropriate architecture being used for de-noising. A simulated dataset with progressively increasing turbidity is used to train a model that predicts the variance parameters required for the reverse diffusion process. By restoring clarity and preserving image details, the project aims to improve underwater image quality 



PROPOSED METHODOLOGY  

In this project we aim to propose a novel diffusion model architecture using real life underwater dataset. We propose a turbid image de-noising methodology with three main steps: 
Noise Parameter Prediction Using Real Underwater Data:

The first step of our proposed methodology involves estimating the noise parameters from real underwater images, which capture the turbidity and degradation characteristics of the environment. In the context of the experimental study, a system was designed to replicate the conditions of a natural underwater environment, ensuring a one-to-one correspondence between the environmental conditions and the captured underwater images. This system was used to create a controlled dataset by supplying the necessary materials and establishing the required conditions.  
Variability in underwater turbidity was simulated using different materials, allowing for controlled measurements of turbidity levels. Samples with varying turbidity levels were used to assess the impact of these changes on image quality. The study examined both direct and indirect factors influencing underwater conditions, with a focus on understanding how variations in turbidity affect object detection performance in underwater environments. Figure 1 shows a snippet of some images from the acquired dataset. 
 
Mathematical Formulation 

To quantify the system, let us create a mathematical expression. Let the original underwater image be represented as x0, and the noisy image at step t in the diffusion process as xt. 

Variance estimation for the turbidity noise with respect to the training epochs where αt is a noise scheduling parameter and ϵt ∼ N (0, I) represents Gaussian noise.  
The objective is to estimate the mean µt and variance σt of the added noise at each step t. For a given dataset of real underwater images, the parameters are learned by minimizing the following loss function.

This loss function ensures that the model accurately predicts the noise parameters, which are critical for generating realistic noisy datasets.  

Forward Diffusion Model for Dataset Generation 
Using the estimated noise parameter σt, a synthetic dataset is generated by iteratively applying the forward diffusion process. This creates a series of noisy images  , where T is the total number of diffusion steps.  
Specifically, a series of 1000 progressively noisy images is created so far, with 100 noisy variations derived from each original. This ensures the model is exposed to diverse noise patterns, enhancing its adaptability to real-world underwater conditions. 
	Reverse Diffusion Model for Image De-noising  
The reverse diffusion model has to be employed to reconstruct the degraded images. The model has to be trained iteratively on the synthetic dataset of increasing turbidity levels, enabling it to learn the underlying noise patterns and understand the noise variance required for the reverse process. The reverse diffusion process aims to reconstruct the original image x0 from xt iterative de-noising.  

Code Implementation 
In the code implementation for training the diffusion model, we first initialize learnable beta parameters that control the noise schedule during the forward diffusion process. These beta values are essential for adding noise progressively to the input images at each time step, and they are updated during training to optimize the model's ability to de-noise the images. 
Steps: 
	Initialization of Learnable Beta: We initialize the beta parameters as learnable variables, allowing the model to adjust them during training. The beta values control the amount of noise added to the images at each step in the forward diffusion process. 
	Forward Diffusion Process: For each input image, we simulate the forward diffusion process. In this process, noise is progressively added to the image over time using the learnable beta parameters. At each time step t, the image is updated by adding noise sampled from a Gaussian distribution, controlled by the corresponding beta value. 
	Noise Prediction: The model attempts to predict the noise added to the image during the forward diffusion process. This predicted noise is then compared to the actual noise that was added to the image. 
	Loss Calculation: The loss function computes the difference between the predicted noise and the actual noise using Mean Squared Error (MSE). This loss is minimized during training to improve the model's de-noising performance. 
	Optimization Using Adam Optimizer: The Adam optimizer is employed to minimize the loss. By adjusting the model's weight based on the gradients of the loss, the optimizer helps the model learn to predict the noise accurately and progressively restore the original image as the training advances. 
	Noisy Images: Creation of noisy images for training the model 
	De-noising model: Implementation of a de-noising architecture such as U-net or Deconvolution method like Weiner filters to extract the upgraded and de-noised image. 
The code has been implemented in Python using different libraries like OpenCv, Pytorch, tensorflow etc. 
Code:

RESULTS AND OUTPUTS OBTAINED
Training loss vs. Epochs 

The model was trained for 20 epochs, and the loss was computed and optimized for each epoch such that after 20 epochs the loss reduced to 0.023.

Comparison of actual turbidity noise and predicted noise 
The next step was to compare the actual turbidity noise and predicted noise and based on their mean square error difference, the loss function was optimized.  
            
Comparison of final predicted noise (Top) vs. actual noise (Bottom) in the diffusion model
Generating Training Data 
Using the estimated and quantified noise parameters, we tend to create noisy images for training the denoising model. 
 
 
De-noising model results 

We implemented two different de-noising techniques on the test data set. The test dataset includes pictures from TUDS.

U-NET MODEL 
The result effectively reduces the haziness while preserving the integrity of the color channels and luminance. Notably, the output appears visually natural, maintaining a realistic representation of the scene. Figure 5 showcases examples of the trained U-Net model’s capability in restoring an image, demonstrating accurate noise estimation, and preserving intricate details without introducing perceptual loss. It is important to note that the quality of the input images played a significant role in influencing the effectiveness of the results. 

DECONVOLUTION 
To compare our results with the trained U-Net model we implemented a traditional deblurring image technique known as deconvolution.
Deconvolution in underwater image restoration reverses blurring caused by light scattering and absorption in turbid water. It helps recover clear images by estimating and inverting the degradation process, often modeled as a convolution with a point spread function (PSF).
Figure 6 depicts the outputs the restoration results using a conventional deconvolution method known as Weiner Deconvolution. It basically works by applying a filter in the frequency domain that enhances the sharpness of the image and suppresses the noise.
In a typical blurring process, the image is convolved with a blur function and then corrupted with additive noise. By applying the Fourier transform we can convert this process in the frequency. To account for both blur and noise Weiner filtering is used. Weiner filtering often assumes that the blur function is invariant (spread evenly across the entire image). However, environmental phenomena such as motion blur and turbidity make Weiner convolution less effective. Figure 6 depicts the shortcomings in the performance of traditional restoration techniques, which emphasizes the need for more efficient deep learning methodologies.

FUTURE WORK 
Moving forward, one key area of improvement is exploring additional de-noising techniques to further enhance the quality of the generated outputs. By experimenting with different noise reduction methods, we can refine the model’s ability to reconstruct clearer and more accurate results. Another important step is expanding the diversity of the training data, ensuring that the model is exposed to a wider range of variations, which can help improve its generalization across different scenarios.
Additionally, optimizing the efficiency of the diffusion model will be a priority. Reducing inference time and computational overhead will make the model more practical for real-world applications. This could involve refining the model architecture, implementing more efficient sampling strategies, or leveraging hardware acceleration techniques. By addressing these aspects, we aim to improve both the performance and usability of the system, making it more accessible for broader applications.
 
