# Tool README

## Overview

This tool implements the methodology introduced in the referenced research paper. It supports the entire process of generating machine learning datasets suitable for AI from MP4 files based on structural and form-based units, as well as training and detecting. For more details, please refer to the paper: [URL].
This document provides detailed information on how to use and set up the tool. This includes prerequisites, installation steps, and usage instructions for both the pre-built executable file and the source code.

## Prerequisites

### 1. For Using the Executable File (`.exe`)

- **No additional installations are required.** Simply download and run the provided executable file.

### 2. For Modifying or Compiling the Source Code

- Ensure that **Python 3.8 or higher** is installed on your system.
- Install the necessary dependencies listed in the `requirements.txt` file.
- Ensure that `ffprobe` is installed on your system. (See the installation instructions below.)

## Installation

### 1. Setting up Python Environment

1. [Download Python](https://www.python.org/downloads/) and install
2. Verify the installation by running:
   ```bash
   python --version
   ```

### 2. Installing Dependencies

1. Navigate to the directory containing the tool's source code.
2. Run the following command to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Installing `ffprobe`

`ffprobe` is required for this tool to function correctly. Follow these steps to install it:

1. Download `ffmpeg` from the [official website](https://ffmpeg.org/download.html) suitable for your operating system.
2. Extract the downloaded archive.
3. Add the `bin` folder (inside the extracted directory) to your system's PATH environment variable.
   - **Windows**: Refer to [this guide](https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/) to update PATH.
   
4. Verify the installation by running:
   ```bash
   ffprobe -version
   ```

## Running the Program

### 1. Using the Executable File (`.exe`)

1. **Double-click the `.exe` file.**
2. Follow the on-screen prompts to use the program.

### 2. Running from Source Code

1. **Open a terminal or command prompt.**
2. Navigate to the directory where the source code resides.
3. Run the program with the following command:
   ```bash
   python createtraining.py
   ```

## Usage

### Command-Line Usage (if applicable)

The program supports the following command-line arguments:

```bash
python createtraining.py [OPTIONS]
```

### Interactive Usage

- If no arguments are provided, the program enters an **interactive mode** where you can provide inputs step by step as prompted.

## Additional Notes

- This tool includes `h264bitstream` for processing H.264 bitstreams. (https://github.com/aizvorski/h264bitstream)
- For detailed error logging, refer to the `logs` folder generated during execution.
- If you encounter any issues, ensure all dependencies are installed correctly and that your Python version is compatible.

## Code Structure

Below is the hierarchical structure of the main code and its related components:

```plaintext
createtraining (Main Code)
├── Case Management
│   └── Manages data and settings for each case
├── Training Data Generation
│   ├── extract_sps (Extracts SPS data)
│   └── extractframe (Extracts GOP patterns)
├── Detection
│   └── Detection logic and execution
└── Training Code
    ├── train_GRUprocess (Binary classification)
    └── train_GRUprocess_multi (Multi-class classification)
```

## Contact

For any questions or issues, please contact the developer at [blind].

## ETC
The initial concept and planning of this research were undertaken as part of a collaborative effort with the (blind). 

