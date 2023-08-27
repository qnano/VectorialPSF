# Vectorial PSF

Vectorial PSF is a tool designed to facilitate the fitting of aberrations in your optical fluorescent microscope system.


## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
We utilize a vector PSF model to address axial localization errors arising from model mismatches in commonly used Gaussian point spread functions (PSFs). This model incorporates super-critical angle fluorescence and an aplanatic correction factor, providing insights into accurate 3D emitter localization.

## Features

Explore the capabilities of the Vectorial PSF tool and unlock valuable insights into your optical system:

1. **Customize Zernike Aberrations and Investigate PSF Effects**

   Select optical Zernike aberrations and study their impact on the Point Spread Function (PSF) and its influence on the Cramer-Rao Lower Bound (CRLB). 

   [View Slideshow](https://imgur.com/a/0HlrptA)

2. **Accurate Aberration Determination Using Through-Focus Scans**

   Utilize a through-focus scan of fluorescent beads to identify aberrations within your system.

   ![Through-Focus Scan](https://imgur.com/a/WzoDf6G)

3. **Spot Fitting from SMLM Data**

   Load candidate emitters from Single-Molecule Localization Microscopy (SMLM) data and fit spots. Save the results in HDF5 files, ready for further in-depth analysis using Picasso (https://github.com/jungmannlab/picasso).

   ![Spot Fitting](https://imgur.com/a/qTn27H8)



## Getting Started

### Prerequisites

This system has been tested on Ubuntu 22.04. The main requirements are Python >3.8 and PyTorch (tested on PyTorch 2.0.1 and CUDA 11.8). A CUDA compatible driver is recommended for faster calculations, but a CPU version should work as well.

### Installation

Follow these steps to get started:

1. **Clone the Repository:**
   ```shell
   git clone https://github.com/pvanvelde/VectorialPSF.git
   cd VectorialPSF
 
2. **Create and Activate a New Conda Virtual Environment:**
   ```shell
   conda create --name vectorial_psf_env python=3.10
   conda activate vectorial_psf_env

3. **Install Dependencies:**
   ```shell
   pip install -r requirements.txt

4. **Run the Application:**
   ```shell
   python GUI.py
   


