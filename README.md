# UD_PerDT_DependencyModel
A UDPipe2 trained model on UD_PerDT (Universal Persian Dependency Treebank).
This is a UDPipe2 model trained on UD_PerDT which is a Persian Universal Dependency Treebank. The corpus is the result of automatic conversion of Dadegan(Persian DT) to its universal version.
it contains nearly 30,000 sentences labeled with lamatization, POS tagges and dependency relations. 

## Steps to run model

1. Setting up an environment for udpip
     ```
     conda create --name udpip_env
     ```
      or 
     ```
      pip3 install virtualenv
      virtualenv udpip_env
      ```
      ```
      pip install -r requirements.txt
      ```
2. Setting up an environment in webembedding_service folder 
      ```
      virtualenv venv
      pip install -r requirements.txt
      ```
3. run script for producing BERT embeddings:
   ```
   bash scripts/compute_embeddings.sh test_data
   ```
   or<br/>
   sequentially run compute_embeddings file for each of your conllu file:
   ```
   python3 --format conllu path/to/input_file/test_data.conllu path/to/output_file/test_data.conllu.npz
   ```
   Noted that npz output file should be in the same folder as input conllu. 
4. download it from [here](https://drive.google.com/file/d/1AyLjszRgq0lhAk3p55DH_e0-o4wd6IXx/view?usp=sharing)
5. run trained model 
   ```
   python3 udpipe2.py uni_PerDT_model --predict --predict_input path/to/input --predict_output path/to/output
   ```
## Input Format
the input file should be prepared in conllu format. you can fill format just with tokens and leave the other tags as blank (_) so trained model will fill them for you.
(refer to the sample file in test_data/test/test_data.conllu)

If your file is in raw text format (.txt), first of all install [hazm](https://pypi.org/project/hazm/) library and then you can use the following script to convert it to conllu:
```
python3 convert_rawTxt_to_conllu.py --input_file path/to/input_txt_file --output_file path/to/save/output_conllu_file
```

---
the result of model on test set of UD_PerDT corpus:

|Metric     | Precision |    Recall |  F1 Score | AligndAcc|
|-----------|-----------|-----------|-----------|----------|
|Tokens     |    100.00 |    100.00 |    100.00 |          |
|Sentences  |    100.00 |    100.00 |    100.00 |          |
|Words      |    100.00 |    100.00 |    100.00 |          |
|UPOS       |     97.55 |     97.55 |     97.55 |     97.55|
|XPOS       |     97.30 |     97.30 |     97.30 |     97.30|
|UFeats     |     97.61 |     97.61 |     97.61 |     97.61|
|AllTags    |     95.28 |     95.28 |     95.28 |     95.28|
|Lemmas     |     98.98 |     98.98 |     98.98 |     98.98|
|UAS        |     93.62 |     93.62 |     93.62 |     93.62|
|LAS        |     90.96 |     90.96 |     90.96 |     90.96|
|CLAS       |     88.97 |     88.73 |     88.85 |     88.73|
|MLAS       |     85.00 |     84.77 |     84.89 |     84.77|
|BLEX       |     87.75 |     87.51 |     87.63 |     87.51|


## Reference
 Rasooli, Mohammad Sadegh, Pegah Safari, Amirsaeid Moloodi, and Alireza Nourian. "The Persian Dependency Treebank Made Universal." arXiv preprint arXiv:2009.10205 (2020).


## Resources
* [UD_PerDT Treebank](https://github.com/phsfr/UD_Persian-PerDT)
* [UD_PerDT Article](https://arxiv.org/pdf/2009.10205.pdf)

#### Contact Info:
pegh.safari@gmail.com