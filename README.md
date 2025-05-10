# accounting-bot
This repository is used in the paper "Developing an accounting virtual assistant through Supervised Fine-Tuning (SFT) of a Small Language Model (SLM)" which is under review for the wiley publisher.



### train.py
is the code for LoRA https://unsloth.ai/ finetuning qwencoder 7B language model on bookkeeping double journal entries from 2007-2023 database. You can find the constrained and anonymized version of database on the link https://huggingface.co/datasets/mariozupan/bookkeeping-posting-schemes-2007-2023. While the model described in a paper has been trained on unmasked and unconstrained dataset, this version has been used only for proofing the concept of the paper and understanding the concept which is reproducable.

### requirements.txt
represents neccessary libraries installed inside nvidia apptainer which runs on https://www.srce.unizg.hr/en/advanced-computing

### ollama Modelfile
is configuration file for inference with the model in ollama llama.cpp wrapper.

