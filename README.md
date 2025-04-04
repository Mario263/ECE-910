Automatic Action Sequence Detection in Movies

This repository contains the full pipeline to automatically detect, extract, and classify action sequences in movies using Vision-Language Models (VLMs) like LLaVA and SigLIP.

Project Overview

The goal of this project is to break down full-length movies into meaningful story units such as:
	•	Fight
	•	Pursuit
	•	Rescue
	•	Escape
	•	Heist
	•	Speed
	•	Capture
	•	None of the Above

The project leverages Histogram + Bhattacharyya Distance, Dynamic Thresholding, and Supervised Fine-Tuning (SFT) of Vision-Language Models for high-accuracy scene classification.

Methodology
	1.	Frame Extraction
Extract frames from movies at a controlled FPS using OpenCV and PIL.
	2.	Shot Boundary Detection
Use PySceneDetect to segment movies into meaningful scenes based on content.
	3.	Keyshot Extraction
	•	Use Grayscale Histograms.
	•	Calculate Bhattacharyya Distance between consecutive frames.
	•	Apply Dynamic Thresholding to retain key moments while removing redundancy.
	4.	Annotation
	•	Fine-tune LLaVA-0.5B and LLaVA-7B models on action-specific categories.
	•	Generate descriptions and labels for each extracted keyshot.
	5.	Post-processing
	•	Use SigLIP for semantic verification.
	•	Optimize output into clean CSV files ready for analysis.

•	Dynamic Threshold at 0.3 achieved the best precision-recall balance for Keyshot Extraction.
•	Fine-tuned models showed significant improvement over zero-shot baselines.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/Mario263)
![Issues](https://img.shields.io/github/issues/https://github.com/Mario263/ECE-910/tree/main)
![Stars](https://img.shields.io/github/stars/Mario263/ECE-910)
