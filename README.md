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

Guide users through setting up and using your project. Provide step-by-step instructions and examples.

### Prerequisites

List any dependencies or prerequisites users need to have installed before using your project.

### Installation

Provide detailed installation instructions. You can include commands for cloning the repository, installing dependencies, etc.

```shell
git clone https://github.com/yourusername/your-project.git
cd your-project
npm install
