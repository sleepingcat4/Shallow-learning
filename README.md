This repository contains code to implement shallow deep learning models and calculate their complexity, conservation of law, and power of law. The purpose of this code is to provide an easy-to-use platform to experiment with the concepts and theories presented in recent papers related to efficient shallow DL models that work similarly to human brains with less computational power.

# Introduction
The code in this repository is my initiative to implement new research papers that come out of Nature, Google, and Meta to expand my understanding of computer science and artificial intelligence. The focus of this repository is on implementing shallow deep learning models and calculating their complexity, conservation of law, and power of law using TensorFlow.

## Purpose of the Paper
Recent papers have outlined the importance of efficient shallow DL models that perform extremely well with minimal computational resources. This paper is a significant milestone in deep learning architecture design, as it aims to minimize the amount of resources required while still achieving high performance, similar to that of the human brain.

## Links
The research paper for this code can be found at https://www.nature.com/articles/s41598-023-32559-8#Sec1.

The Google Colab Notebook containing the code can be accessed at https://colab.research.google.com/drive/1iVNQSg994SOTTSaxYMufupLa5Lv924bD?usp=sharing.

## LeNet-5 Calculations
```python
Power Law for LeNet-5: d2/d1 = 10.0 , constant = -0.9242792860618816
Complexity for LeNet-5 using power law as decay error: [3072.0, 897.9212113772354, 224.48030284430885, 123.36069990688432, 30.84017497672108, 1.4369348847391794, 1.398645979800433, 1.1904761904761907]
Conservation law for LeNet-5: depthi × mi = 232.08061813867866
```
