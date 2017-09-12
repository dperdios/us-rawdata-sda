[Ecole Polytechnique Fédérale de Lausanne (EPFL)]: https://www.epfl.ch/
[Signal Processing Laboratory (LTS5)]: https://lts5www.epfl.ch
[IEEE International Ultrasonics Symposium (IUS 2017)]: http://ewh.ieee.org/conf/ius/2017/
[paper]: https://infoscience.epfl.ch/record/230991
[PICMUS]: https://www.creatis.insa-lyon.fr/EvaluationPlatform/picmus/index.html
[k-Wave]: http://www.k-wave.org
[TensorFlow]: (https://www.tensorflow.org)

# A Deep Learning Approach to Ultrasound Image Recovery

Dimitris Perdios<sup>1</sup>, Adrien Besson<sup>1</sup>, Marcel Arditi<sup>1</sup>, and Jean-Philippe Thiran<sup>1, 2</sup>

<sup>1</sup>[Signal Processing Laboratory (LTS5)], [Ecole Polytechnique Fédérale de Lausanne (EPFL)], Switzerland

<sup>2</sup>Department of Radiology, University Hospital Center (CHUV), Switzerland

[Paper] accepted at the [IEEE International Ultrasonics Symposium (IUS 2017)].

> Based on the success of deep neural networks for image recovery, we propose a new paradigm for the compression and decompression of ultrasound (US) signals which relies on stacked denoising autoencoders.
> The first layer of the network is used to compress the signals and the remaining layers perform the reconstruction.
> We train the network on simulated US signals and evaluate its quality on images of the publicly available PICMUS dataset.
> We demonstrate that such a simple architecture outperforms state-of-the art methods, based on the compressed sensing framework, both in terms of image quality and computational complexity.

### [Paper] updates:
Please note that the accepted preprint version (v1) has been updated.
Each version can be found [here](http://infoscience.epfl.ch/record/230991).
* v1: accepted preprint
* v2: fix typos in the abbreviations used in Table II w.r.t. the text
* v3: fix fig. 2(f) display dB range and corresponding interpretation

## Installation
1. Install Python 3.6 and optionally create a dedicated environment.
1. Clone the repository.
    ```bash
    git clone https://github.com/dperdios/us-rawdata-sda
    cd us-rawdata-sda
    ```
1. Install the Python dependencies from `requirements.txt`.
    * **Note 1:** by default, it will install the GPU version of [TensorFlow].
    If you do not have a compatible GPU, please edit `requirements.txt` as follow:
        * Comment the line starting with `tensorflow-gpu`. 
        * Uncomment the line starting with `tensorflow`.
    * **Note 2:** [TensorFlow] 1.3 has not been tested yet, hence `requirements.txt` will install 1.2.1.
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        ```
1. Download the [PICMUS] dataset, which will be used as the test set.
    ```bash
    python3 setup.py
    ```
    
    This will download, under `datasets/picmus17`, the following acquisitions, provided by the [PICMUS] authors:
    * `dataset_rf_in_vitro_type1_transmission_1_nbPW_1`
    * `dataset_rf_in_vitro_type2_transmission_1_nbPW_1`
    * `dataset_rf_in_vitro_type3_transmission_1_nbPW_1`
    * `dataset_rf_numerical_transmission_1_nbPW_1`
    
    It will also download, under `datasets/picmus16` two *in vivo* acquisitions provided by the [PICMUS] authors (from the [first](https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/home) version of the challenge), namely:
    * `carotid_cross_expe_dataset_rf`
    * `carotid_long_expe_dataset_rf`


## Usage

### Trained networks
To reproduce the results on the [PICMUS] dataset described above, using the trained networks provided under `networks/ius2017`, use the following command:
```bash
python3 ius2017_results.py
```
This will compute the beamformed image for all the trained networks on every [PICMUS] acquisitions.
The corresponding images will be saved as PDF under `results/ius2017`.

**Note 1:** This can take some time since [PICMUS] acquisition for each measurement ratio will be beamformed using a non-optimized delay-and-sum beamformer. 

**Note 2:** The codes used to generate the CS results cannot be provided for the moment. Sorry for the incovenience.

Once the results have been computed, it is possible to lauch the following command for a better visualization:
```bash
python3 ius2017_add_figure.py
```
This will save a PDF image, under `results/ius2017/*_metrics.pdf`, for each [PICMUS] acquisition (see above) displaying the results in terms of PSNR for each method (SDA-CL, SDA-CNL and CS) and measurement ratio.

### Training the networks (not yet available)
We are currently investigating an appropriate way to release and distribute the training set that has been numerically generated using the open-source [k-Wave] toolbox.
The code is however already provided.
Once the training set is available, it will be possible to re-train the networks using the following command:
```bash
python3 ius2017_train.py
```

## License
The code is released under the terms of the [MIT license](LICENSE.txt).

If you are using this code, please cite our [paper].

## Acknowledgements
This work was supported in part by the UltrasoundToGo RTD project (no. 20NA21_145911), funded by [Nano-Tera.ch](http://www.nano-tera.ch).