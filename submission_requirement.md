# Submission
In order to participate to the CAID challenge you have to complete an online form. Please make sure your software meets the requirements listed below.

## Requirements
- Must include clear instructions on how to install and run the predictor.
- Must be able to run on CPU machines, without GPU.
- Must implement a command line interface (no user interaction, etc.).
- Must work without access to the internet.
- Must work with less than 48 GB of RAM.
- Must work with no more than 24 threads.
- Must last less than 6 hours for a single sequence prediction

## Best practices
Do not include public databases inside the predictor code (e.g., UniProt, HHBlits, etc.). Just provide the name and version of the database you want to use, and we will take care of providing it.
Avoid including standard dependencies inside the predictor code (e.g., the entire conda/venv environment). Instead, include an environment.txt, conda-env.yml, or any best-practice dependency descriptor.
Avoid using hard-coded paths in your code. Instead, create a configuration file and/or command line arguments to specify the paths to the input/output files.
If possible, create a Dockerfile/docker-compose recipe. It is even better if you publish it on a public repository like Docker Hub.
If your predictor requires additional inputs in addition to the protein sequence, make it possible to pass them as command line arguments (e.g., path to a file or string). For example, Psi-Blast or HHBlits results are often required by many predictors, and we can pre-compute them once for all predictors. Similarly, large ML models can be stored in a different location compared to the code.
Make it possible to specify the number of cores/threads to use (if applicable).
Ensure that the program works on an Linux machine that is not the one used for developing the software.
If your method uses any protein language models, do not include the model or its weights inside the container. We will precompute the embeddings and provide them to your software. Your software should preferably accept the path to the input embedding file as a command-line argument. The embeddings will be provided in either .npy or .h5 format.
If your method relies on structures/MSAs generated using ColabFold/AlphaFold, do not include ColabFold/AlphaFold or its dependencies in the container. Instead, please provide the exact command required to generate the structures. We will run this command and precompute the structures/MSAs for you.
Your method should take care of ambiguous chars in the FASTA input not crashing and provide a score.
* B - Asparagine (N) or Aspartic acid (D)
* Z - Glutamine (Q) or Glutamic acid (E)
* J - Leucine (L) or Isoleucine (I)
* U - Selenocysteine
* O - Pyrrolysine
* X - Any amino acid