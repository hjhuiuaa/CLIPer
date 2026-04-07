# CLIPer
**CLIPer** stands for **Contrastive Learning for Intrinsically disordered Protein linker classification**.
CLIPer is a binary classification project for identifying protein linker regions using contrastive learning.  
The training data is curated from **DisProt**, and the evaluation setting is aligned with the **CAID3** benchmark.  
Unlike related work that focuses on small-molecule binding sites, this project targets **protein linker classification** and uses **ProstT5** as the feature extractor.
## Project Goal
Build a robust binary classifier before the next CAID competition (May) by adapting and improving the training pipeline for linker-focused prediction.
## Key Characteristics
- **Task:** Protein linker binary classification
- **Learning strategy:** Contrastive learning
- **Feature model:** ProstT5
- **Training data source:** DisProt
- **Benchmark reference:** CAID3
## Current Focus
- Refactor and adapt training code to the CLIPer task setup
- Validate data processing and split strategy
- Establish reproducible training and evaluation workflow
## Planned Milestone
- Deliver a competition-ready binary model before May
## Citation / Acknowledgment
This project is inspired by contrastive-learning-based protein prediction studies.  
If you use this repository, please cite the final manuscript or repository release once available.
