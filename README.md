# AutoSafe Project

Building LLM Safety using an embedding based filtering mechanism for post-hoc detection and flagging.
Built in lieu of the LLM Agents MOOC Berkeley course.

## Demo app

  The demo app is a Chainlit application hosted in app.py. The core Autosafe filter model is loaded in autosafe.py which
  flags the responses in real time as interacted through the application. The application is an NVIDIA finance expert
  and is given the latest NVIDIA quarterly results report. Thus, the goal of Autosafe is to ensure that the LLM is not
  used to answer questions "outside" this domain.

## Autosafe Model.

 The notebook Autosafe_1.ipynb is used to graph and build a model for autosafe for the NVIDIA utterances. 

## Benchmarking

  We benchmark against the Best-of-N datase from the recent paper https://jplhughes.github.io/bon-jailbreaking
  The benchmarking code and results are in Autosafe_Benchmark.ipynb and show the effectiveness of our system to be
  close to 99%.
