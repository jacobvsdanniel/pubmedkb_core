# R-BERT Update Notes

- Support multiple mentions per entity
- Support the control of learning rate schedule and pretrained tokenizer
- Adding various training, self-training, evaluation, prediction scripts and improvements
- Preprocessed datasets: *snpphena* (SNPPhenA), *anndisge* (my annotated parts of ClinVar+DisGeNET)
- Pre-trained model: a self-learning model combining biobert, snpphena, and ClinVar+DisGeNET
- Pre-trained model: a self-learning model combining biobert, anndisge, and unannotated parts of ClinVar+DisGeNET

## How to Run Prediction

### 1. Download pre-trained models

Git lfs is needed to clone large **.bin* files in *model/biobert-snpphena-disgevar1-disgevar2* and *model/biobert-nonanndisge3*

### 2. Preprocess your own input sentences

#### 2.1 Mask entity mentions by VARIANT and DISEASE

Although prefrontal cortical and ventral striatal activity are highly relevant for **addictive behavior**, and under partial control of COMT **rs4680** genotype, no association between **COMT** and **smoking behavior** was observed.

-> ```Although prefrontal cortical and ventral striatal activity are highly relevant for DISEASE, and under partial control of VARIANT genotype, no association between VARIANT and DISEASE was observed.```

#### 2.2 Mark targeted entity pairs by *\<e1\>* (for variants) and *\<e2\>* (for diseases) tags

(One possible pair) -> ```Although prefrontal cortical and ventral striatal activity are highly relevant for <e2> DISEASE </e2>, and under partial control of <e1> VARIANT </e1> genotype, no association between VARIANT and DISEASE was observed.```

Note that if a targeted entity has multiple mentions, all of them should be marked. For example:

In this report, we present an asymptomatic infant, seen for a second opinion, who was given the diagnosis of **cystic fibrosis** (**CF**) as a neonate based on the presence of two mutant alleles, DeltaF508 and R117H.

(One possible pair) -> ```In this report, we present an asymptomatic infant, seen for a second opinion, who was given the diagnosis of <e2> DISEASE </e2> (<e2> DISEASE </e2>) as a neonate based on the presence of two mutant alleles, VARIANT and <e1> VARIANT </e1>.```

### 3. Generate predictions

```bash
$ time CUDA_VISIBLE_DEVICES=0 python predict.py \
--model model/biobert-snpphena-disgevar1-disgevar2 \
--input_file sample_pred_in.txt \
--output_file output.txt
```

- *output.txt*: n lines of predicted labels
- *output.txt.npy*: n*m numpy float array of label scores for an m-way classifier

### 4. Prediction outputs

#### 4.1 Model *biobert-snpphena-disgevar1-disgevar2*

3-way V-D association:
- *Other*
- *VPOS-D(e1,e2)*: there is a statistical association between V and D
- *VNEG-D(e1,e2)*: there is no statistical association between V and D

```Although prefrontal cortical and ventral striatal activity are highly relevant for <e2> DISEASE </e2>, and under partial control of <e1> VARIANT </e1> genotype, no association between VARIANT and DISEASE was observed.```
-> prediction should be *VPOS-D(e1,e2)*

```Although prefrontal cortical and ventral striatal activity are highly relevant for DISEASE, and under partial control of <e1> VARIANT </e1> genotype, no association between VARIANT and <e2> DISEASE </e2> was observed.```
-> prediction should be *Other*

```Although prefrontal cortical and ventral striatal activity are highly relevant for DISEASE, and under partial control of VARIANT genotype, no association between <e1> VARIANT </e1> and <e2> DISEASE </e2> was observed.```
-> prediction should be *VNEG-D(e1,e2)*

#### 4.2 Model *biobert-nonanndisge3*

4-way V-D association
- *Other*
- *Vpositive-D(e1,e2)*: There is a positive association, e.g., causality, between V and D.
- *Vappositive-D(e1,e2)*: An appositive construct, at-gene, or a be-verb links V and D.
- *Vpatient-D(e1,e2)*: A prevalence rate or cohort is described for V and D.

```A novel <e2> DISEASE </e2> with DISEASE and DISEASE is caused by a <e1> VARIANT </e1> mutation in the fibroblast growth factor receptor 3 gene.```
-> prediction should be *Vpositive-D(e1,e2)*

```We describe a new missense mutation (<e1> VARIANT </e1>, VARIANT) in the <e2> DISEASE </e2> gene.```
-> prediction should be *Vappositive-D(e1,e2)*

```The <e1> VARIANT </e1> mutation accounts for approximately 48% of <e2> DISEASE </e2> alleles in African Americans and has been found to account for about 91% of <e2> DISEASE </e2> alleles in negroid South African patients which suggested that the mutation had an African origin.```
-> prediction should be *Vpatient-D(e1,e2)*

### 5. Speed Profile

Setup

- Hardware: Using one RTX 2080 Ti on node lab3-g3
- Dataset: 13,937 sentences from ClinVar+DisGeNET
- Batch size: 32

Result

- 2m14s for prediction
- 104 sent/sec

# Original R-BERT README

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enriching-pre-trained-language-model-with/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=enriching-pre-trained-language-model-with)

(Unofficial) Pytorch implementation of `R-BERT`: [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png" />  
</p>

### **Method**

1. **Get three vectors from BERT.**
   - [CLS] token vector
   - averaged entity_1 vector
   - averaged entity_2 vector
2. **Pass each vector to the fully-connected layers.**
   - dropout -> tanh -> fc-layer
3. **Concatenate three vectors.**
4. **Pass the concatenated vector to fully-connect layer.**
   - dropout -> fc-layer

- **_Exactly the SAME conditions_** as written in paper.
  - **Averaging** on `entity_1` and `entity_2` hidden state vectors, respectively. (including \$, # tokens)
  - **Dropout** and **Tanh** before Fully-connected layer.
  - **No [SEP] token** at the end of sequence. (If you want add [SEP] token, give `--add_sep_token` option)

## Dependencies

- perl (For evaluating official f1 score)
- python>=3.6
- torch==1.6.0
- transformers==3.3.1

## How to run

```bash
$ python3 main.py --do_train --do_eval
```

- Prediction will be written on `proposed_answers.txt` in `eval` directory.

## Official Evaluation

```bash
$ python3 official_eval.py
# macro-averaged F1 = 88.29%
```

- Evaluate based on the official evaluation perl script.
  - MACRO-averaged f1 score (except `Other` relation)
- You can see the detailed result on `result.txt` in `eval` directory.

## Prediction

```bash
$ python3 predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}
```

## References

- [Semeval 2010 Task 8 Dataset](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?sort=name&layout=list&num=50)
- [Semeval 2010 Task 8 Paper](https://www.aclweb.org/anthology/S10-1006.pdf)
- [NLP-progress Relation Extraction](http://nlpprogress.com/english/relationship_extraction.html)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)
