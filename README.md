<h2>TensorFlow-FlexUNet-Image-Segmentation-Endscopy-Multi-Organ-Disease (2026/02/03)</h2>
Sarah T.  Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>EndScopy Multi Organ Disease (EDD2020) </b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass), 
and an <a href="https://drive.google.com/file/d/1nd7I_SbOUFQBzihfClDqlq00k2ZU-zsT/view?usp=sharing">Augmented-EDD2020-ImageMask-Dataset.zip</a> 
with colorized masks, which was derived by us from
<br><br>
<a href="https://www.kaggle.com/datasets/orvile/edd2020-endoscopy-detection-and-segmentation">
<b>EDD2020: Endoscopy Detection and Segmentation</b> </a> dataset on the kaggle.com.
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original <b>EDD2020</b> dataset,
we used our offline augmentation tool
<a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a> to generate our Augmented-EDD2020 dataset.
<br><br> 
<hr>
<b>Actual Image Segmentation for EDD2020 Images  </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the augmented dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {BE:cyan, cancer:red,  HGD:green, polyp:magenta,  suspicious:blue}</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10042.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10042.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10042.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10122.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10122.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10182.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://www.kaggle.com/datasets/orvile/edd2020-endoscopy-detection-and-segmentation">
<b>EDD2020: Endoscopy Detection and Segmentation</b> </a> <br>
<b>Multi-Organ Disease Analysis Dataset</b>
<br><br>
For more information, please refer to <a href="https://ieee-dataport.org/competitions/endoscopy-disease-detection-and-segmentation-edd2020">
<b>Endoscopy Disease Detection and Segmentation (EDD2020)</b>
</a><b>IEEE<i>DataPort</i></b><br><br>
The following explanation was taken from <a href="https://www.kaggle.com/datasets/orvile/edd2020-endoscopy-detection-and-segmentation">
EDD2020: Endoscopy Detection and Segmentation</a><br><br>
<b>About Dataset</b><br>
EDD2020 tackles disease detection and segmentation in endoscopy videos from 5 global centers. <br>
With bounding boxes and pixel-level masks for conditions like polyps, cancer, and Barrett’s, it’s a benchmark for real-time monitoring 
and offline analysis—boosting precision in GI healthcare. 
<br><br>
<b>What’s Inside?</b><br>
<ul>
<li>Images: 386 (all labeled)</li>
<li>Classes: 5 (BE, suspicious, HGD, cancer, polyp)</li>
<li>Annotations: Bounding boxes (VOC format) & instance segmentation masks</li>
<li>Format: .jpg, .txt, .tif</li>
<li>Size: 57.36 MB</li>
<li>Source: Multi-organ (colon, esophagus, stomach) from 5 centers</li>
</ul><br>
<b>Citation</b><br>
Ali, Sharib; Braden, Barbara; Lamarque, Dominique; Realdon, Stefano; Bailey, Adam; Cannizzaro, Renato; Ghatwary, Noha; <br>
Rittscher, Jens; Daul, Christian; East, James. (2020). Endoscopy Disease Detection and Segmentation (EDD2020) [Dataset]. <br>
IEEE DataPort. <a href="https://dx.doi.org/10.21227/f8xg-wb80">https://dx.doi.org/10.21227/f8xg-wb80</a>
<br><br>
<b>Contributors</b><br>
<ul>
<li>Sharib Ali (University of Oxford)</li>
<li>Barbara Braden (University of Oxford)</li>
<li>Dominique Lamarque (Université de Versailles)</li>
<li>Stefano Realdon (Instituto Oncologico Veneto)</li>
<li>Adam Bailey (University of Oxford)</li>
<li>Renato Cannizzaro (CRO Aviano)</li>
<li>Noha Ghatwary (University of Lincoln)</li>
<li>Jens Rittscher (University of Oxford)</li>
<li>Christian Daul (University of Lorraine)</li>
<li>James East (University of Oxford)</li>
</ul>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/"><b>Attribution 4.0 International (CC BY 4.0)</b></a>
<br>
<br>
<h3>
2 EDD2020 ImageMask Dataset
</h3>
 If you would like to train this EDD2020 Segmentation model by yourself,
please down load our dataset <a href="https://drive.google.com/file/d/1nd7I_SbOUFQBzihfClDqlq00k2ZU-zsT/view?usp=sharing">
<b>Augmented-EDD2020-ImageMask-Dataset.zip</b> </a> on the google-drive.
expand the downloaded , and put it under ./dataset.<br>
<pre>
./dataset
└─EDD2020
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>EDD2020 Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/EDD2020/EDD2020_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br><br>

<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained EDD2020 TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/EDD2020/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/EDD2020 and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 6
base_filters   = 16
base_kernels  = (9,9)
num_layers    = 8
dropout_rate   = 0.05
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for EDD2020 1+5 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;EDD2020 1+5
;{"BE":(0,255,255), "cancer":(255,0,0), "HGD":(10,128,10), "polyp":(200,0,200), "suspicious":(40,40,255)}
rgb_map = {(0,0,0):0,  (0,255,255):1, (255,0,0):2, (10,128,10):3, (200,0,200):4, (40,40,255):5}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (27,28,29)</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (54,55,56)</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 56 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/train_console_output_at_epoch56.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/EDD2020/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/EDD2020/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/EDD2020</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for EDD2020.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/evaluate_console_output_at_epoch56.png" width="880" height="auto">
<br><br>Image-Segmentation-EDD2020

<a href="./projects/TensorFlowFlexUNet/EDD2020/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this EDD2020/test was low, and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.2778
dice_coef_multiclass,0.9082
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/EDD2020</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for EDD2020.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/EDD2020/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  EDD2020  Images</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {BE:cyan, cancer:red,  HGD:green, polyp:magenta,  suspicious:blue}</b>
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10055.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10055.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10055.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10182.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10182.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10262.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10262.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10262.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/10299.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/10299.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/10299.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/deformed_alpha_1300_sigmoid_7_10002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/deformed_alpha_1300_sigmoid_7_10002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/deformed_alpha_1300_sigmoid_7_10002.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/images/deformed_alpha_1300_sigmoid_7_10006.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test/masks/deformed_alpha_1300_sigmoid_7_10006.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/EDD2020/mini_test_output/deformed_alpha_1300_sigmoid_7_10006.png" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
<h3>
References
</h3>
<b>1. TensorFlow-FlexUNet-Image-Segmentation-Merged-Polyp</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Merged-Polyp">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Merged-Polyp
</a>
<br>
<br>
<b>2. TensorFlow-FlexUNet-Image-Segmentation-Gastrointestinal-Polyp</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Gastrointestinal-Polyp">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Gastrointestinal-Polyp
</a>
<br><br>
<b>3. TensorFlow-FlexUNet-Image-Segmentation-Sessile-Polyp</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Sessile-Polyp">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Sessile-Polyp
</a>
<br><br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-ETIS-LaribPolypDB</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-ETIS-LaribPolypDB">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-ETIS-LaribPolypDB
</a>
<br>
<br>
<b>5. Tensorflow-Image-Segmentation-Augmented-Colon-Polyp</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Augmented-Colon-Polyp">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Augmented-Colon-Polyp
</a>
<br>
<br>
